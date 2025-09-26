import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
import pandas as pd
import pandas_ta as ta
import pytz

from services.signal_parser import SignalParser

class SignalProcessor:
    """
    The central brain of the trading bot. This class orchestrates the entire
    process from signal detection to trade execution and management.
    """
    def __init__(self, config, ib_interface, telegram_interface, discord_interface, 
                 state_manager, sentiment_analyzer, initial_positions, initial_processed_ids):
        self.config = config
        self.ib_interface = ib_interface
        self.telegram_interface = telegram_interface
        self.discord_interface = discord_interface
        self.state_manager = state_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_parser = SignalParser(config)

        # Live operational state
        self.open_positions = initial_positions
        self.processed_message_ids = deque(initial_processed_ids, maxlen=config.processed_message_cache_size)
        self.channel_cooldowns = {}
        
        # Real-time data management
        self.position_data_cache = {} # {conId: pd.DataFrame}
        self.tick_buffer = {} # {conId: [ticks]}
        self.last_bar_timestamp = {} # {conId: datetime}
        
        # Graceful exit state
        self.trailing_highs = {} # {conId: float} for pullback stop
        self.atr_stop_prices = {} # {conId: float} for ATR trail
        self.breakeven_activated = {} # {conId: bool} to track breakeven state

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """The main entry point. Sets up concurrent tasks for all bot operations."""
        logging.info("Starting Signal Processor...")
        
        self.ib_interface.set_order_filled_callback(self._on_order_filled)
        
        await self._resubscribe_to_open_positions()

        tasks = [
            self._poll_discord_for_signals(),
            self._process_market_data_stream(),
        ]
        await asyncio.gather(*tasks)
        await self.shutdown()

    async def shutdown(self):
        """Gracefully shuts down all bot components."""
        if not self._shutdown_event.is_set():
            logging.info("Initiating graceful shutdown...")
            self._shutdown_event.set()

    # =================================================================
    # --- CORE LOGIC: Signal Processing and Trade Execution ---
    # =================================================================

    async def _poll_discord_for_signals(self):
        """Task to continuously poll Discord for new signals."""
        while not self._shutdown_event.is_set():
            if self._is_eod():
                await self.flatten_all_positions()
                await self.shutdown()
                break

            for profile in self.config.profiles:
                if not profile['enabled']:
                    continue

                channel_id = profile['channel_id']
                now = datetime.now()
                if now < self.channel_cooldowns.get(channel_id, now):
                    continue

                raw_messages = await self.discord_interface.poll_for_new_messages(channel_id, self.processed_message_ids)
                if raw_messages:
                    await self._process_new_signals(raw_messages, profile)

                self.channel_cooldowns[channel_id] = now + timedelta(seconds=self.config.delay_between_channels)

            await asyncio.sleep(self.config.delay_after_full_cycle)

    async def _process_new_signals(self, messages, profile):
        """Processes a batch of new messages for a given profile."""
        stale_message_count = 0

        for msg_id, msg_content, msg_timestamp in messages:
            if msg_id in self.processed_message_ids:
                continue
            
            # --- THE "ONE AND DONE" FIX ---
            # Mark the message as processed IMMEDIATELY to prevent re-processing on failure.
            self.processed_message_ids.append(msg_id)

            # --- THE STALE SIGNAL CHECK ---
            now_utc = datetime.now(timezone.utc)
            signal_age = now_utc - msg_timestamp
            if signal_age.total_seconds() > self.config.signal_max_age_seconds:
                stale_message_count += 1
                continue # Silently continue to the next message

            logging.info(f"Processing new message {msg_id} from '{profile['channel_name']}'")

            if any(word.lower() in msg_content.lower() for word in self.config.buzzwords_ignore):
                logging.debug(f"Message {msg_id} ignored due to buzzword.") # Demoted to DEBUG
                continue

            parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
            
            if not isinstance(parsed_signal, dict):
                logging.debug(f"Message {msg_id} did not parse into a valid signal dictionary.") # Demoted to DEBUG
                continue

            if self.config.sentiment_filter['enabled']:
                sentiment_score = self.sentiment_analyzer.get_sentiment_score(msg_content)
                is_call = parsed_signal['contract_type'].upper() == 'CALL'
                
                threshold = self.config.sentiment_filter['sentiment_threshold'] if is_call else self.config.sentiment_filter['put_sentiment_threshold']
                
                if (is_call and sentiment_score < threshold) or (not is_call and sentiment_score > threshold):
                    veto_reason = f"Sentiment score {sentiment_score:.4f} is outside threshold {threshold} for a {parsed_signal['contract_type']}."
                    logging.warning(f"Trade VETOED for {parsed_signal['ticker']}. Reason: {veto_reason}")
                    trade_info = {
                        'ticker': parsed_signal['ticker'],
                        'option': f"{parsed_signal['strike']}{parsed_signal['contract_type'][0].upper()}",
                        'expiry': parsed_signal['expiry_date'],
                        'source': profile['channel_name'],
                        'reason': veto_reason
                    }
                    await self.telegram_interface.send_trade_notification(trade_info, "VETOED")
                    continue
            
            await self._execute_trade_from_signal(parsed_signal, profile)
        
        # --- THE "SILENT SENTRY" FIX ---
        if stale_message_count > 0:
            logging.debug(f"Ignored {stale_message_count} stale message(s) from this poll cycle.") # Demoted to DEBUG

    async def _execute_trade_from_signal(self, signal, profile):
        """Validates and executes a single trade, aware of its origin."""
        try:
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], signal['expiry_date'], signal['strike'], signal['contract_type']
            )
            if not contract:
                return

            ticker = await self.ib_interface.get_live_ticker(contract)
            if not ticker or pd.isna(ticker.ask) or ticker.ask <= 0:
                logging.error(f"Could not get a valid ask price for {contract.localSymbol}. Cannot size position.")
                return
            
            ask_price = ticker.ask
            if not (profile['trading']['min_price_per_contract'] <= ask_price <= profile['trading']['max_price_per_contract']):
                logging.warning(f"Trade for {contract.localSymbol} vetoed. Ask price ${ask_price} is outside limits.")
                return
            
            quantity = int(profile['trading']['funds_allocation'] / (ask_price * 100))
            if quantity == 0:
                logging.warning(f"Trade for {contract.localSymbol} vetoed. Not enough funds to purchase a single contract at ${ask_price}.")
                return

            logging.info(f"Calculated quantity: {quantity} for {contract.localSymbol} at ask price ${ask_price}")
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity)
            if order:
                order.channel_id = profile['channel_id']
                logging.info(f"Successfully placed order for {quantity} of {contract.localSymbol} from channel {profile['channel_name']}")
                trade_info = {
                    'ticker': signal['ticker'],
                    'option': f"{signal['strike']}{signal['contract_type'][0].upper()}",
                    'expiry': signal['expiry_date'],
                    'source': profile['channel_name'],
                }
                await self.telegram_interface.send_trade_notification(trade_info, "OPENED")

        except Exception as e:
            logging.error(f"An error occurred during trade execution: {e}", exc_info=True)

    def _on_order_filled(self, trade):
        """Callback executed by IBInterface when an order is filled."""
        contract = trade.contract
        order = trade.order
        channel_id = getattr(order, 'channel_id', None)

        if channel_id is None:
            logging.warning(f"Could not determine originating channel for fill of {contract.localSymbol}. Using first enabled profile.")
            for profile in self.config.profiles:
                if profile['enabled']:
                    channel_id = profile['channel_id']
                    break
        
        fill_price = trade.execution.avgPrice
        quantity = trade.execution.shares
        
        logging.info(f"Order filled: {quantity} of {contract.localSymbol} at ${fill_price}")

        position_details = {
            'contract': contract,
            'entry_price': fill_price,
            'quantity': quantity,
            'entry_time': datetime.now(),
            'channel_id': channel_id
        }
        self.open_positions[contract.conId] = position_details
        self.trailing_highs[contract.conId] = fill_price 
        self.breakeven_activated[contract.conId] = False

        asyncio.create_task(self._post_fill_actions(trade, position_details))

    async def _post_fill_actions(self, trade, position_details):
        """Actions to take after an order is confirmed filled."""
        contract = trade.contract
        profile = self._get_profile_by_channel_id(position_details['channel_id'])

        if profile and profile['safety_net']['enabled']:
            trail_percent = profile['safety_net']['native_trail_percent']
            await self.ib_interface.attach_native_trail(trade.order, trail_percent)

        subscription_successful = await self.ib_interface.subscribe_to_market_data(contract)
        if subscription_successful:
            historical_data = await self.ib_interface.get_historical_data(contract)
            if historical_data is not None and not historical_data.empty:
                self.position_data_cache[contract.conId] = historical_data
                logging.info(f"Initialized historical data cache for {contract.localSymbol}")
            else:
                logging.warning(f"Could not fetch initial historical data for {contract.localSymbol}")
        else:
            logging.error(f"Failed to subscribe to market data for {contract.localSymbol}. Dynamic exits will be disabled.")
        
        self.state_manager.save_state(self.open_positions, self.processed_message_ids)

    # =================================================================
    # --- CORE LOGIC: Real-time Position Management ---
    # =================================================================
    
    async def _process_market_data_stream(self):
        """Task to continuously process real-time market data from the queue."""
        while not self._shutdown_event.is_set():
            try:
                ticker = await asyncio.wait_for(self.ib_interface.market_data_queue.get(), timeout=1.0)
                if ticker.contract.conId in self.open_positions:
                    await self._resample_ticks_to_bar(ticker)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing market data stream: {e}", exc_info=True)

    async def _resample_ticks_to_bar(self, ticker):
        """Collects ticks and resamples them into time-based bars for analysis."""
        conId = ticker.contract.conId
        now = datetime.now()
        
        if conId not in self.tick_buffer:
            self.tick_buffer[conId] = []
            self.last_bar_timestamp[conId] = now.replace(second=0, microsecond=0)

        if ticker.last > 0:
            self.tick_buffer[conId].append((now, ticker.last))

        if now >= self.last_bar_timestamp[conId] + timedelta(minutes=1):
            profile = self._get_profile_by_channel_id(self.open_positions[conId]['channel_id'])
            min_ticks = profile['exit_strategy'].get('min_ticks_per_bar', 1)

            if len(self.tick_buffer[conId]) >= min_ticks:
                
                prices = [p for t, p in self.tick_buffer[conId]]
                new_bar = {
                    'open': prices[0], 'high': max(prices),
                    'low': min(prices), 'close': prices[-1],
                    'volume': len(prices)
                }
                
                new_bar_df = pd.DataFrame([new_bar], index=[self.last_bar_timestamp[conId]])
                
                if conId in self.position_data_cache:
                    self.position_data_cache[conId] = pd.concat([self.position_data_cache[conId], new_bar_df])
                else:
                    self.position_data_cache[conId] = new_bar_df
                
                await self._evaluate_dynamic_exit(conId)

            self.tick_buffer[conId] = []
            self.last_bar_timestamp[conId] = now.replace(second=0, microsecond=0)


    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        if conId not in self.open_positions: return

        position = self.open_positions[conId]
        profile = self._get_profile_by_channel_id(position['channel_id'])
        data = self.position_data_cache.get(conId)

        if profile is None or data is None or data.empty or len(data) < 2: return

        is_call = position['contract'].right == 'C'
        exit_reason = None
        
        # This is where the breakeven, ATR, pullback, RSI, and PSAR checks happen.
        # This section is a placeholder for the fully implemented logic.
        pass

        if exit_reason:
            logging.info(f"Dynamic exit triggered for {position['contract'].localSymbol}. Reason: {exit_reason}")
            await self._execute_close_trade(conId, exit_reason)


    async def _execute_close_trade(self, conId, reason):
        """Closes a position and updates the state."""
        if conId in self.open_positions:
            position_to_close = self.open_positions.pop(conId)
            contract = position_to_close['contract']
            quantity = position_to_close['quantity']
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity, action='SELL')

            if order:
                logging.info(f"Successfully placed closing order for {quantity} of {contract.localSymbol}")
                
                await self.ib_interface.unsubscribe_from_market_data(contract)
                
                self._cleanup_position_data(conId)

                self.state_manager.save_state(self.open_positions, self.processed_message_ids)

                profile = self._get_profile_by_channel_id(position_to_close['channel_id'])
                trade_info = {
                    'ticker': contract.symbol,
                    'option': f"{contract.strike}{contract.right[0]}",
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'source': profile['channel_name'] if profile else 'N/A',
                    'pnl': "N/A - Fill price not yet available",
                    'exit_reason': reason
                }
                await self.telegram_interface.send_trade_notification(trade_info, "CLOSED")

    def _cleanup_position_data(self, conId):
        """Helper to remove all data associated with a closed/ghost position."""
        self.position_data_cache.pop(conId, None)
        self.tick_buffer.pop(conId, None)
        self.last_bar_timestamp.pop(conId, None)
        self.trailing_highs.pop(conId, None)
        self.atr_stop_prices.pop(conId, None)
        self.breakeven_activated.pop(conId, None)

    # =================================================================
    # --- UTILITY AND HELPER METHODS ---
    # =================================================================

    async def flatten_all_positions(self):
        """Closes all open positions. Triggered at EOD."""
        if not self.open_positions:
            logging.info("EOD reached. No open positions to flatten.")
            return

        logging.warning("EOD reached. Flattening all open positions.")
        await self.telegram_interface.send_message("🚨 EOD reached. Flattening all positions. 🚨")
        
        position_ids_to_close = list(self.open_positions.keys())
        
        for conId in position_ids_to_close:
            await self._execute_close_trade(conId, "EOD Flatten")

    def _is_eod(self):
        """Checks if the current time is past the EOD close time, using timezone-aware logic."""
        eod_config = self.config.eod_close
        if not e_config['enabled']:
            return False
        
        try:
            market_tz = pytz.timezone(self.config.MARKET_TIMEZONE)
            now_in_market_tz = datetime.now(market_tz)
            eod_in_market_tz = now_in_market_tz.replace(
                hour=eod_config['hour'], minute=eod_config['minute'], second=0, microsecond=0
            )
            return now_in_market_tz >= eod_in_market_tz
        except pytz.UnknownTimeZoneError:
            logging.error(f"FATAL: Unknown timezone in config: '{self.config.MARKET_TIMEZONE}'. EOD check disabled.")
            return False
        except Exception as e:
            logging.error(f"A critical error occurred in the EOD check: {e}", exc_info=True)
            return False

    def _get_profile_by_channel_id(self, channel_id):
        """Finds the correct profile for a given channel ID."""
        for profile in self.config.profiles:
            if profile['channel_id'] == str(channel_id):
                return profile
        logging.warning(f"Could not find a profile for channel ID {channel_id}")
        return None
    
    async def _resubscribe_to_open_positions(self):
        """Resubscribes to market data for all positions loaded from state."""
        if not self.open_positions:
            return
            
        logging.info(f"Resubscribing to market data for {len(self.open_positions)} loaded position(s)...")
        for conId, position in self.open_positions.items():
            await self.ib_interface.subscribe_to_market_data(position['contract'])
            historical_data = await self.ib_interface.get_historical_data(position['contract'])
            if historical_data is not None and not historical_data.empty:
                self.position_data_cache[conId] = historical_data

