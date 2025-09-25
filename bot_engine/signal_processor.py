import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta

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

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """The main entry point. Sets up concurrent tasks for all bot operations."""
        logging.info("Starting Signal Processor...")
        await self.telegram_interface.send_message("🤖 Bot is starting up...")

        # Initialize and login to Discord
        await self.discord_interface.initialize_and_login()

        # Resubscribe to market data for any positions loaded from state
        await self._resubscribe_to_open_positions()

        # Define and run all concurrent tasks
        tasks = [
            self._poll_discord_for_signals(),
            self._process_market_data_stream(),
            # self._monitor_for_shutdown_command() # Optional: A task to listen for a shutdown command
        ]
        await asyncio.gather(*tasks)
        await self.shutdown()

    async def shutdown(self):
        """Gracefully shuts down all bot components."""
        logging.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
        # The main 'start' method will handle the rest
        
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
        for msg_id, msg_content, msg_timestamp in messages:
            if msg_id in self.processed_message_ids:
                continue
            
            self.processed_message_ids.append(msg_id)
            logging.info(f"Processing new message {msg_id} from '{profile['channel_name']}'")

            # 1. Pre-flight check: Ignore buzzwords
            if any(word in msg_content for word in self.config.buzzwords_ignore):
                logging.info(f"Message {msg_id} ignored due to buzzword. Content: {msg_content}")
                continue

            # 2. Parse the signal
            parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
            if not parsed_signal:
                continue

            # 3. Pre-flight check: Sentiment Analysis
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
            
            # 4. Execute the trade
            await self._execute_trade_from_signal(parsed_signal, profile)

    async def _execute_trade_from_signal(self, signal, profile):
        """Validates and executes a single trade based on a parsed signal."""
        try:
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], signal['expiry_date'], signal['strike'], signal['contract_type']
            )
            if not contract:
                logging.error(f"Could not create or find unique contract for signal: {signal}")
                return
            
            # --- Intelligent Sizing ---
            ticker = await self.ib_interface.get_live_ticker(contract)
            if not ticker or pd.isna(ticker.ask):
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
            
            # --- Place Order ---
            order = await self.ib_interface.place_order(contract, 'MKT', quantity)
            if order:
                logging.info(f"Successfully placed order for {quantity} of {contract.localSymbol}")
                # The _on_order_filled callback will handle the rest
                trade_info = {
                    'ticker': signal['ticker'],
                    'option': f"{signal['strike']}{signal['contract_type'][0].upper()}",
                    'expiry': signal['expiry_date'],
                    'source': profile['channel_name'],
                }
                # Call the new structured notification method
                await self.telegram_interface.send_trade_notification(trade_info, "OPENED")

        except Exception as e:
            logging.error(f"An error occurred during trade execution: {e}", exc_info=True)

    def _on_order_filled(self, trade):
        """Callback executed by IBInterface when an order is filled."""
        contract = trade.contract
        fill_price = trade.execution.avgPrice
        quantity = trade.execution.shares
        
        logging.info(f"Order filled: {quantity} of {contract.localSymbol} at ${fill_price}")

        # Store position details
        position_details = {
            'contract': contract,
            'entry_price': fill_price,
            'quantity': quantity,
            'entry_time': datetime.now(),
            'profile_name': self.get_profile_for_position(contract) # Placeholder for multi-profile logic
        }
        self.open_positions[contract.conId] = position_details

        asyncio.create_task(self._post_fill_actions(trade, position_details))

    async def _post_fill_actions(self, trade, position_details):
        """Actions to take after an order is confirmed filled."""
        contract = trade.contract
        profile = self._get_profile_by_name(position_details['profile_name'])

        # 1. Attach Native Trail
        if profile and profile['safety_net']['enabled']:
            trail_percent = profile['safety_net']['native_trail_percent']
            await self.ib_interface.attach_native_trail(trade.order, trail_percent)

        # 2. Subscribe to market data for dynamic exits
        subscription_successful = await self.ib_interface.subscribe_to_market_data(contract)
        if subscription_successful:
            # 3. Fetch initial historical data to build the first bar set
            historical_data = await self.ib_interface.get_historical_data(contract)
            if historical_data is not None and not historical_data.empty:
                self.position_data_cache[contract.conId] = historical_data
                logging.info(f"Initialized historical data cache for {contract.localSymbol}")
            else:
                 logging.warning(f"Could not fetch initial historical data for {contract.localSymbol}")
        else:
            logging.error(f"Failed to subscribe to market data for {contract.localSymbol}. Dynamic exits will be disabled.")
        
        # 4. Save the new state
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
        
        # Initialize buffers if not present
        if conId not in self.tick_buffer:
            self.tick_buffer[conId] = []
            self.last_bar_timestamp[conId] = now.replace(second=0, microsecond=0)

        # Append the latest tick data (timestamp, price)
        # Using last price, but could use bid/ask for more precision
        if ticker.last > 0:
            self.tick_buffer[conId].append((now, ticker.last))

        # Check if a minute has passed
        if now >= self.last_bar_timestamp[conId] + timedelta(minutes=1):
            profile = self._get_profile_by_name(self.open_positions[conId]['profile_name'])
            min_ticks = profile['exit_strategy'].get('min_ticks_per_bar', 1)

            # Quality Control: Only form a bar if we have enough ticks
            if len(self.tick_buffer[conId]) >= min_ticks:
                
                # Create a new bar (OHLC) from the buffered ticks
                prices = [p for t, p in self.tick_buffer[conId]]
                new_bar = {
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': len(prices) # Use tick count as a proxy for volume
                }
                
                new_bar_df = pd.DataFrame([new_bar], index=[self.last_bar_timestamp[conId]])
                
                # Append the new bar to our historical data cache
                if conId in self.position_data_cache:
                    self.position_data_cache[conId] = pd.concat([self.position_data_cache[conId], new_bar_df])
                else:
                    self.position_data_cache[conId] = new_bar_df
                
                # Run exit logic on the updated data
                await self._evaluate_dynamic_exit(conId)

            # Reset for the next bar
            self.tick_buffer[conId] = []
            self.last_bar_timestamp[conId] = now.replace(second=0, microsecond=0)


    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        if conId not in self.open_positions:
            return

        position = self.open_positions[conId]
        profile = self._get_profile_by_name(position['profile_name'])
        data = self.position_data_cache.get(conId)

        if profile is None or data is None or data.empty:
            return
        
        is_call = position['contract'].right == 'C'
        exit_reason = None

        # --- Breakeven Logic ---
        # TODO: This logic needs to be more robust. It should probably set a stop order.
        current_price = data['close'].iloc[-1]
        pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
        if pnl_percent >= profile['exit_strategy']['breakeven_trigger_percent']:
            # This is a conceptual implementation. Real implementation would manage a stop order.
            logging.info(f"Position {position['contract'].localSymbol} hit breakeven trigger.")


        # --- Configurable Exit Hierarchy ---
        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_reason: break # Exit if a reason has been found

            if exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                # ATR logic would be here. It's complex as it needs to maintain a trailing value.
                pass # Placeholder

            elif exit_type == "pullback_stop" and profile['exit_strategy']['trail_method'] == 'pullback_percent':
                # Pullback logic would be here.
                pass # Placeholder

            elif exit_type == "rsi_hook" and profile['exit_strategy']['momentum_exits']['rsi_hook_enabled']:
                rsi_settings = profile['exit_strategy']['momentum_exits']['rsi_settings']
                data.ta.rsi(length=rsi_settings['period'], append=True)
                last_rsi = data[f'RSI_{rsi_settings["period"]}'].iloc[-1]
                
                if is_call and last_rsi > rsi_settings['overbought_level']:
                    exit_reason = f"RSI crossed overbought level ({last_rsi:.2f} > {rsi_settings['overbought_level']})"
                elif not is_call and last_rsi < rsi_settings['oversold_level']:
                    exit_reason = f"RSI crossed oversold level ({last_rsi:.2f} < {rsi_settings['oversold_level']})"

            elif exit_type == "psar_flip" and profile['exit_strategy']['momentum_exits']['psar_enabled']:
                psar_settings = profile['exit_strategy']['momentum_exits']['psar_settings']
                data.ta.psar(initial=psar_settings['start'], increment=psar_settings['increment'], maximum=psar_settings['max'], append=True)
                last_close = data['close'].iloc[-1]
                # Check for PSARl (long) or PSARs (short) columns
                if f'PSARl_{psar_settings["start"]}_{psar_settings["max"]}' in data.columns:
                    last_psar = data[f'PSARl_{psar_settings["start"]}_{psar_settings["max"]}'].iloc[-1]
                    if is_call and last_close < last_psar:
                        exit_reason = f"Price crossed below PSAR ({last_close:.2f} < {last_psar:.2f})"
                if f'PSARs_{psar_settings["start"]}_{psar_settings["max"]}' in data.columns:
                    last_psar = data[f'PSARs_{psar_settings["start"]}_{psar_settings["max"]}'].iloc[-1]
                    if not is_call and last_close > last_psar:
                        exit_reason = f"Price crossed above PSAR ({last_close:.2f} > {last_psar:.2f})"

        if exit_reason:
            logging.info(f"Dynamic exit triggered for {position['contract'].localSymbol}. Reason: {exit_reason}")
            await self._execute_close_trade(conId, exit_reason)


    async def _execute_close_trade(self, conId, reason):
        """Closes a position and updates the state."""
        if conId in self.open_positions:
            position_to_close = self.open_positions[conId]
            contract = position_to_close['contract']
            quantity = position_to_close['quantity']
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity, action='SELL')

            if order:
                logging.info(f"Successfully placed closing order for {quantity} of {contract.localSymbol}")
                
                # Unsubscribe from market data
                await self.ib_interface.unsubscribe_from_market_data(contract)
                
                # Remove from live management
                del self.open_positions[conId]
                self.position_data_cache.pop(conId, None)
                self.tick_buffer.pop(conId, None)
                self.last_bar_timestamp.pop(conId, None)

                # SURGICAL FIX: Save the state after closing a position
                self.state_manager.save_state(self.open_positions, self.processed_message_ids)

                # TODO: We need the fill price to calculate P/L for the notification
                trade_info = {
                    'ticker': contract.symbol,
                    'option': f"{contract.strike}{contract.right[0]}",
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'source': self._get_profile_by_name(position_to_close['profile_name'])['channel_name'],
                    'pnl': "N/A - Fill price not yet available",
                    'exit_reason': reason
                }
                await self.telegram_interface.send_trade_notification(trade_info, "CLOSED")


    # =================================================================
    # --- UTILITY AND HELPER METHODS ---
    # =================================================================

    async def flatten_all_positions(self):
        """Closes all open positions. Triggered at EOD."""
        logging.warning("EOD reached. Flattening all open positions.")
        await self.telegram_interface.send_message("🚨 EOD reached. Flattening all positions. 🚨")
        
        # Create a copy of the keys to avoid issues with modifying dict during iteration
        position_ids_to_close = list(self.open_positions.keys())
        
        for conId in position_ids_to_close:
            await self._execute_close_trade(conId, "EOD Flatten")

    def _is_eod(self):
        """Checks if the current time is past the EOD close time."""
        eod_config = self.config.eod_close
        if not eod_config['enabled']:
            return False
        
        now = datetime.now() # TODO: Make this timezone-aware
        eod_time = now.replace(hour=eod_config['hour'], minute=eod_config['minute'], second=0, microsecond=0)
        
        return now >= eod_time

    def get_profile_for_position(self, contract):
        # This is a placeholder. A real implementation would need a way
        # to know which profile initiated which trade.
        # For now, we'll assume the first enabled profile.
        for profile in self.config.profiles:
            if profile['enabled']:
                return profile['channel_name']
        return "default"

    def _get_profile_by_name(self, profile_name):
        for profile in self.config.profiles:
            if profile['channel_name'] == profile_name:
                return profile
        return None
    
    async def _resubscribe_to_open_positions(self):
        """Resubscribes to market data for all positions loaded from state."""
        if not self.open_positions:
            return
            
        logging.info(f"Resubscribing to market data for {len(self.open_positions)} loaded position(s)...")
        for conId, position in self.open_positions.items():
            await self.ib_interface.subscribe_to_market_data(position['contract'])
            # Also re-initialize the data cache if needed
            historical_data = await self.ib_interface.get_historical_data(position['contract'])
            if historical_data is not None and not historical_data.empty:
                self.position_data_cache[conId] = historical_data