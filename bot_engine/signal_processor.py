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
        self.global_cooldown_until = datetime.now() 

        # Real-time data management
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        
        # Graceful exit state
        self.trailing_highs_and_lows = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """
        The main entry point. Sets up and runs all concurrent tasks 
        for the bot's operations correctly.
        """
        logging.info("Starting Signal Processor...")
        
        self.ib_interface.set_order_filled_callback(self._on_order_filled)
        
        await self._reconcile_state_with_broker()

        await self._resubscribe_to_open_positions()

        tasks = [
            asyncio.create_task(self._poll_discord_for_signals()),
            asyncio.create_task(self._process_market_data_stream()),
            asyncio.create_task(self._reconcile_positions_periodically())
        ]
        
        await asyncio.gather(*tasks)
        await self.shutdown()

    async def shutdown(self):
        """Gracefully shuts down all bot components."""
        if not self._shutdown_event.is_set():
            logging.info("Initiating graceful shutdown...")
            self._shutdown_event.set()

    async def _reconcile_state_with_broker(self):
        """
        Compares the bot's internal state with the broker's actual portfolio
        at startup to eliminate ghost positions and adopt untracked ones.
        """
        logging.info("Performing initial state reconciliation with broker...")
        broker_positions = await self.ib_interface.get_open_positions()
        
        broker_positions = [p for p in broker_positions if p.position != 0]

        broker_conIds = {pos.contract.conId for pos in broker_positions}
        internal_conIds = set(self.open_positions.keys())

        ghost_positions = internal_conIds - broker_conIds
        if ghost_positions:
            logging.warning(f"Reconciliation: Found {len(ghost_positions)} ghost position(s) in state file. Removing.")
            for conId in list(ghost_positions):
                self._cleanup_position_data(conId)
        
        untracked_positions = broker_conIds - internal_conIds
        if untracked_positions:
            logging.info(f"Reconciliation: Found {len(untracked_positions)} untracked position(s) at broker. Adopting them.")
            for pos in broker_positions:
                if pos.contract.conId in untracked_positions:
                    entry_price = pos.avgCost / 100 if pos.contract.secType == 'OPT' else pos.avgCost
                    position_details = {
                        'contract': pos.contract, 'entry_price': entry_price,
                        'quantity': pos.position, 'entry_time': datetime.now(),
                        'channel_id': self._get_fallback_channel_id()
                    }
                    self.open_positions[pos.contract.conId] = position_details
                    self.trailing_highs_and_lows[pos.contract.conId] = {'high': entry_price, 'low': entry_price}
                    self.breakeven_activated[pos.contract.conId] = False
                    logging.info(f"Adopted position: {pos.position} of {pos.contract.localSymbol}")

        self.state_manager.save_state(self.open_positions, self.processed_message_ids)
        logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")

    async def _reconcile_positions_periodically(self):
        """
        Periodically syncs the bot's internal state with the broker's actual portfolio
        to prevent state desynchronization (managing "ghost" trades).
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.reconciliation_interval_seconds)
                logging.info("--- Starting periodic position reconciliation ---")
                
                broker_positions = await self.ib_interface.get_open_positions()
                broker_conIds = {pos.contract.conId for pos in broker_positions if pos.position != 0}
                internal_conIds = set(self.open_positions.keys())

                ghost_positions = internal_conIds - broker_conIds
                if ghost_positions:
                    logging.warning(f"Reconciliation: Found {len(ghost_positions)} ghost position(s). Removing from internal state.")
                    for conId in ghost_positions:
                        await self.ib_interface.unsubscribe_from_market_data(self.open_positions[conId]['contract'])
                        self._cleanup_position_data(conId)
                    self.state_manager.save_state(self.open_positions, self.processed_message_ids)
                
                logging.info("--- Position reconciliation complete ---")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error during position reconciliation: {e}", exc_info=True)

    async def _poll_discord_for_signals(self):
        """Task to continuously poll Discord for new signals."""
        while not self._shutdown_event.is_set():
            now = datetime.now()
            if now < self.global_cooldown_until:
                await asyncio.sleep(1)
                continue

            if self._is_eod():
                await self.flatten_all_positions()
                await self.shutdown()
                break

            for profile in self.config.profiles:
                if not profile['enabled']: continue
                channel_id = profile['channel_id']
                if now < self.channel_cooldowns.get(channel_id, now): continue
                raw_messages = await self.discord_interface.poll_for_new_messages(channel_id, self.processed_message_ids)
                if raw_messages:
                    await self._process_new_signals(raw_messages, profile)
                self.channel_cooldowns[channel_id] = now + timedelta(seconds=self.config.delay_between_channels)

            await asyncio.sleep(self.config.delay_after_full_cycle)

    async def _process_new_signals(self, messages, profile):
        """Processes a batch of new messages for a given profile."""
        processed_something_new = False
        for msg_id, msg_content, msg_timestamp in messages:
            if msg_id in self.processed_message_ids: continue
            processed_something_new = True
            self.processed_message_ids.append(msg_id)
            
            now_utc = datetime.now(timezone.utc)
            signal_age = now_utc - msg_timestamp
            if signal_age.total_seconds() > self.config.signal_max_age_seconds:
                logging.debug(f"Message {msg_id} is stale. Ignoring.")
                continue

            logging.info(f"Processing new message {msg_id} from '{profile['channel_name']}'")
            parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
            if not isinstance(parsed_signal, dict):
                logging.debug(f"Message {msg_id} did not parse into a valid signal.")
                continue
            
            await self._execute_trade_from_signal(parsed_signal, profile, None)
        
        if processed_something_new:
            self.state_manager.save_state(self.open_positions, self.processed_message_ids)
            logging.debug("Updated processed message ID cache to state file.")

    async def _execute_trade_from_signal(self, signal, profile, sentiment_score):
        """Validates and executes a single trade."""
        try:
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], signal['expiry_date'], signal['strike'], signal['contract_type']
            )
            if not contract: return

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
                order.sentiment_score = sentiment_score
                logging.info(f"Successfully placed order for {quantity} of {contract.localSymbol}")

        except Exception as e:
            logging.error(f"An error occurred during trade execution: {e}", exc_info=True)

    async def _on_order_filled(self, trade):
        """Callback executed by IBInterface when an order is filled."""
        contract = trade.contract
        order = trade.order
        fill_price = trade.orderStatus.avgFillPrice
        quantity = trade.orderStatus.filled

        if order.action == "BUY":
            channel_id = getattr(order, 'channel_id', self._get_fallback_channel_id())
            sentiment_score = getattr(order, 'sentiment_score', None)
            
            logging.info(f"Entry fill: {quantity} of {contract.localSymbol} at ${fill_price}")

            position_details = {
                'contract': contract, 'entry_price': fill_price,
                'quantity': quantity, 'entry_time': datetime.now(),
                'channel_id': channel_id
            }
            self.open_positions[contract.conId] = position_details
            self.trailing_highs_and_lows[contract.conId] = {'high': fill_price, 'low': fill_price}
            self.breakeven_activated[contract.conId] = False

            await self._post_fill_actions(trade, position_details, sentiment_score)
        
        elif order.action == "SELL":
            if contract.conId in self.open_positions:
                position_to_close = self.open_positions.pop(contract.conId)
                logging.info(f"Exit fill: {quantity} of {contract.localSymbol} at ${fill_price}")

                pnl = (fill_price - position_to_close['entry_price']) * quantity * 100
                profile = self._get_profile_by_channel_id(position_to_close['channel_id'])
                
                trade_info = {
                    'contract_details': contract.localSymbol,
                    'exit_price': fill_price,
                    'pnl': f"${pnl:.2f}",
                    'reason': getattr(order, 'exit_reason', 'Manual/Unknown')
                }
                await self.telegram_interface.send_trade_notification(trade_info, "CLOSED")
                self._cleanup_position_data(contract.conId)
                self.state_manager.save_state(self.open_positions, self.processed_message_ids)
            else:
                logging.warning(f"Received a SELL fill for an untracked position: {contract.localSymbol}")

    async def _post_fill_actions(self, trade, position_details, sentiment_score):
        """Actions to take after an ENTRY order is confirmed filled."""
        contract = trade.contract
        profile = self._get_profile_by_channel_id(position_details['channel_id'])

        if profile:
            momentum_exits = []
            if profile['exit_strategy']['momentum_exits'].get('psar_enabled'):
                momentum_exits.append("PSAR")
            if profile['exit_strategy']['momentum_exits'].get('rsi_hook_enabled'):
                momentum_exits.append("RSI")
            
            trade_info = {
                'source_channel': profile['channel_name'],
                'contract_details': contract.localSymbol,
                'quantity': position_details['quantity'],
                'entry_price': position_details['entry_price'],
                'sentiment_score': sentiment_score,
                'trail_method': profile['exit_strategy']['trail_method'].upper(),
                'momentum_exit': ", ".join(momentum_exits) if momentum_exits else "None"
            }
            await self.telegram_interface.send_trade_notification(trade_info, "OPENED")

        if profile and profile['safety_net']['enabled']:
            trail_percent = profile['safety_net']['native_trail_percent']
            await self.ib_interface.attach_native_trail(trade.order, trail_percent)

        subscription_successful = await self.ib_interface.subscribe_to_market_data(contract)
        if subscription_successful:
            historical_data = await self.ib_interface.get_historical_data(contract)
            if historical_data is not None and not historical_data.empty:
                self.position_data_cache[contract.conId] = historical_data
        
        self.state_manager.save_state(self.open_positions, self.processed_message_ids)

    async def _process_market_data_stream(self):
        """Task to continuously process real-time market data from the queue."""
        while not self._shutdown_event.is_set():
            try:
                ticker = await asyncio.wait_for(self.ib_interface.market_data_queue.get(), timeout=1.0)
                if ticker.contract.conId in self.open_positions:
                    await self._resample_ticks_to_bar(ticker)
            except asyncio.TimeoutError:
                continue

    async def _resample_ticks_to_bar(self, ticker):
        """Collects ticks and resamples them into time-based bars for analysis."""
        conId = ticker.contract.conId
        now = datetime.now()
        
        if conId not in self.tick_buffer:
            self.tick_buffer[conId] = []
            self.last_bar_timestamp[conId] = now.replace(second=0, microsecond=0)

        if ticker.last > 0:
            self.tick_buffer[conId].append(ticker.last)

        if now >= self.last_bar_timestamp[conId] + timedelta(minutes=1):
            profile = self._get_profile_by_channel_id(self.open_positions[conId]['channel_id'])
            min_ticks = profile['exit_strategy']['min_ticks_per_bar'] if profile else 1

            if len(self.tick_buffer[conId]) >= min_ticks:
                prices = self.tick_buffer[conId]
                new_bar = {'open': prices[0], 'high': max(prices), 'low': min(prices), 'close': prices[-1]}
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
        if not all([profile, data is not None, not data.empty, len(data) >= 2]): return

        is_call = position['contract'].right == 'C'
        exit_reason = None
        current_price = data['close'].iloc[-1]
        
        high_low = self.trailing_highs_and_lows[conId]
        high_low['high'] = max(high_low['high'], current_price)
        high_low['low'] = min(high_low['low'], current_price)

        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_reason: break
            
            if exit_type == "breakeven" and not self.breakeven_activated.get(conId):
                pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
                trigger_pct = profile['exit_strategy']['breakeven_trigger_percent']
                if (is_call and pnl_percent >= trigger_pct) or (not is_call and pnl_percent <= -trigger_pct):
                    logging.info(f"Breakeven triggered for {position['contract'].localSymbol}.")
                    self.breakeven_activated[conId] = True
            
            if self.breakeven_activated.get(conId):
                if (is_call and current_price <= position['entry_price']) or \
                   (not is_call and current_price >= position['entry_price']):
                    exit_reason = "Breakeven Stop Hit"

            if exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                settings = profile['exit_strategy']['trail_settings']
                data.ta.atr(length=settings['atr_period'], append=True)
                last_atr = data.get(f'ATRr_{settings["atr_period"]}', pd.Series([0])).iloc[-1]
                if pd.isna(last_atr): continue
                
                if is_call:
                    stop_price = current_price - (last_atr * settings['atr_multiplier'])
                    self.atr_stop_prices[conId] = max(self.atr_stop_prices.get(conId, 0), stop_price)
                    if current_price < self.atr_stop_prices[conId]:
                        exit_reason = f"ATR Trail ({current_price:.2f} < {self.atr_stop_prices[conId]:.2f})"
                else:
                    stop_price = current_price + (last_atr * settings['atr_multiplier'])
                    self.atr_stop_prices[conId] = min(self.atr_stop_prices.get(conId, float('inf')), stop_price)
                    if current_price > self.atr_stop_prices[conId]:
                        exit_reason = f"ATR Trail ({current_price:.2f} > {self.atr_stop_prices[conId]:.2f})"

            elif exit_type == "pullback_stop" and profile['exit_strategy']['trail_method'] == 'pullback_percent':
                pullback_pct = profile['exit_strategy']['trail_settings']['pullback_percent']
                if is_call:
                    stop_price = high_low['high'] * (1 - (pullback_pct / 100))
                    if current_price < stop_price: exit_reason = f"Pullback Stop ({pullback_pct}%)"
                else:
                    stop_price = high_low['low'] * (1 + (pullback_pct / 100))
                    if current_price > stop_price: exit_reason = f"Pullback Stop ({pullback_pct}%)"

            elif exit_type == "rsi_hook" and profile['exit_strategy']['momentum_exits']['rsi_hook_enabled']:
                settings = profile['exit_strategy']['momentum_exits']['rsi_settings']
                data.ta.rsi(length=settings['period'], append=True)
                last_rsi = data.get(f'RSI_{settings["period"]}', pd.Series([0])).iloc[-1]
                if pd.isna(last_rsi): continue

                prev_rsi = data[f'RSI_{settings["period"]}'].iloc[-2]
                if is_call and prev_rsi > settings['overbought_level'] and last_rsi <= settings['overbought_level']:
                    exit_reason = f"RSI Hook from Overbought ({prev_rsi:.2f} -> {last_rsi:.2f})"
                elif not is_call and prev_rsi < settings['oversold_level'] and last_rsi >= settings['oversold_level']:
                    exit_reason = f"RSI Hook from Oversold ({prev_rsi:.2f} -> {last_rsi:.2f})"
            
            elif exit_type == "psar_flip" and profile['exit_strategy']['momentum_exits']['psar_enabled']:
                settings = profile['exit_strategy']['momentum_exits']['psar_settings']
                data.ta.psar(initial=settings['start'], increment=settings['increment'], maximum=settings['max'], append=True)
                psar_long_col = f'PSARl_{settings["start"]}_{settings["max"]}'
                psar_short_col = f'PSARs_{settings["start"]}_{settings["max"]}'
                if psar_long_col in data.columns and not pd.isna(data[psar_long_col].iloc[-1]):
                    if is_call and current_price < data[psar_long_col].iloc[-1]:
                        exit_reason = "PSAR Flip"
                if psar_short_col in data.columns and not pd.isna(data[psar_short_col].iloc[-1]):
                    if not is_call and current_price > data[psar_short_col].iloc[-1]:
                        exit_reason = "PSAR Flip"

        if exit_reason:
            logging.info(f"Dynamic exit for {position['contract'].localSymbol}. Reason: {exit_reason}")
            await self._execute_close_trade(conId, exit_reason)

    async def _execute_close_trade(self, conId, reason):
        """Places a SELL order for a position and attaches the exit reason."""
        if conId in self.open_positions:
            position = self.open_positions[conId]
            contract = position['contract']
            quantity = position['quantity']
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity, action='SELL')

            if order:
                order.exit_reason = reason
                logging.info(f"Placed closing order for {quantity} of {contract.localSymbol}")
            else:
                logging.error(f"Failed to place closing order for {contract.localSymbol}")

    def _cleanup_position_data(self, conId):
        """Helper to remove all data associated with a closed/ghost position."""
        self.open_positions.pop(conId, None)
        self.position_data_cache.pop(conId, None)
        self.tick_buffer.pop(conId, None)
        self.last_bar_timestamp.pop(conId, None)
        self.trailing_highs_and_lows.pop(conId, None)
        self.atr_stop_prices.pop(conId, None)
        self.breakeven_activated.pop(conId, None)

    async def flatten_all_positions(self):
        """Closes all open positions. Triggered at EOD."""
        logging.warning("EOD reached. Flattening all open positions.")
        await self.telegram_interface.send_message("🚨 EOD reached. Flattening all positions. 🚨")
        
        position_ids_to_close = list(self.open_positions.keys())
        
        for conId in position_ids_to_close:
            await self._execute_close_trade(conId, "EOD Flatten")

    def _is_eod(self):
        """Checks if the current time is past the EOD close time."""
        eod_config = self.config.eod_close
        if not eod_config['enabled']:
            return False
        
        try:
            market_tz = pytz.timezone(self.config.MARKET_TIMEZONE)
            now_in_market_tz = datetime.now(market_tz)
            eod_in_market_tz = now_in_market_tz.replace(
                hour=eod_config['hour'], minute=eod_config['minute'], second=0, microsecond=0
            )
            return now_in_market_tz >= eod_in_market_tz
        except pytz.UnknownTimeZoneError:
            logging.error(f"FATAL: Unknown timezone in config: '{self.config.MARKET_TIMEZONE}'.")
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
    
    def _get_fallback_channel_id(self):
        """Finds the first enabled profile to use as a fallback for adopted positions."""
        for profile in self.config.profiles:
            if profile['enabled']:
                return profile['channel_id']
        return self.config.profiles[0]['channel_id'] if self.config.profiles else "unknown"

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

