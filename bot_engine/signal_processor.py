import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
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
        
        # Link the fill handler from the interface to our processor method
        self.ib_interface.set_order_filled_callback(self._on_order_filled)
        
        await self._resubscribe_to_open_positions()

        tasks = [
            self._poll_discord_for_signals(),
            self._process_market_data_stream(),
            self._reconcile_positions_periodically()
        ]
        await asyncio.gather(*tasks)
        await self.shutdown()

    async def shutdown(self):
        """Gracefully shuts down all bot components."""
        if not self._shutdown_event.is_set():
            logging.info("Initiating graceful shutdown...")
            self._shutdown_event.set()

    # =================================================================
    # --- CORE LOGIC: Reconciliation ---
    # =================================================================
    async def _reconcile_positions_periodically(self):
        """Periodically syncs the bot's internal state with the broker's portfolio."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.reconciliation_interval_seconds)
                logging.info("--- Starting periodic position reconciliation ---")
                
                broker_positions = await self.ib_interface.get_open_positions()
                broker_conIds = {pos.contract.conId for pos in broker_positions}
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
            
            # ... (rest of signal processing logic is unchanged)

            # 4. Execute the trade, now passing the originating channel_id
            await self._execute_trade_from_signal(parsed_signal, profile)


    async def _execute_trade_from_signal(self, signal, profile):
        """Validates and executes a single trade, aware of its origin."""
        try:
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], signal['expiry_date'], signal['strike'], signal['contract_type']
            )
            # ... (sizing logic is unchanged)
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity)
            if order:
                # Associate the order with its originating channel profile
                order.channel_id = profile['channel_id']
                logging.info(f"Successfully placed order for {quantity} of {contract.localSymbol} from channel {profile['channel_name']}")
                # ... (Telegram notification is unchanged)
        except Exception as e:
            logging.error(f"An error occurred during trade execution: {e}", exc_info=True)

    def _on_order_filled(self, trade):
        """Callback executed by IBInterface when an order is filled."""
        contract = trade.contract
        order = trade.order
        channel_id = getattr(order, 'channel_id', None)

        if channel_id is None:
            logging.warning(f"Could not determine originating channel for fill of {contract.localSymbol}. Using first enabled profile.")
            # Fallback for manually placed trades or other edge cases
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
            'channel_id': channel_id # NON-NEGOTIABLE FIX: Store the origin
        }
        self.open_positions[contract.conId] = position_details
        self.trailing_highs[contract.conId] = fill_price 
        self.breakeven_activated[contract.conId] = False

        asyncio.create_task(self._post_fill_actions(trade, position_details))

    async def _post_fill_actions(self, trade, position_details):
        """Actions to take after an order is confirmed filled."""
        # ... (post-fill actions are largely unchanged)
        self.state_manager.save_state(self.open_positions, self.processed_message_ids)

    # =================================================================
    # --- CORE LOGIC: Real-time Position Management (HEAVILY UPGRADED) ---
    # =================================================================
    
    async def _process_market_data_stream(self):
        # ... (unchanged)
        pass

    async def _resample_ticks_to_bar(self, ticker):
        # ... (unchanged)
        pass

    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        if conId not in self.open_positions: return

        position = self.open_positions[conId]
        profile = self._get_profile_by_channel_id(position['channel_id'])
        data = self.position_data_cache.get(conId)

        if profile is None or data is None or data.empty or len(data) < 2: return

        is_call = position['contract'].right == 'C'
        exit_reason = None
        current_price = data['close'].iloc[-1]

        # Update trailing highs/lows for pullback stops
        if is_call:
            self.trailing_highs[conId] = max(self.trailing_highs.get(conId, 0), current_price)
        else: # Is Put
            self.trailing_highs[conId] = min(self.trailing_highs.get(conId, float('inf')), current_price)
        
        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_reason: break

            if exit_type == "breakeven" and not self.breakeven_activated.get(conId):
                pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
                if (is_call and pnl_percent >= profile['exit_strategy']['breakeven_trigger_percent']) or \
                   (not is_call and pnl_percent <= -profile['exit_strategy']['breakeven_trigger_percent']):
                    logging.info(f"Breakeven triggered for {position['contract'].localSymbol}. Logic to place stop order would go here.")
                    self.breakeven_activated[conId] = True # Fire once
                    # This is where you would place a STP order at entry price
            
            if exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                atr_settings = profile['exit_strategy']['trail_settings']
                data.ta.atr(length=atr_settings['atr_period'], append=True)
                last_atr = data[f'ATRr_{atr_settings["atr_period"]}'].iloc[-1]
                
                if is_call:
                    stop_price = current_price - (last_atr * atr_settings['atr_multiplier'])
                    self.atr_stop_prices[conId] = max(self.atr_stop_prices.get(conId, 0), stop_price)
                    if current_price < self.atr_stop_prices[conId]:
                        exit_reason = f"ATR Trailing Stop Hit ({current_price:.2f} < {self.atr_stop_prices[conId]:.2f})"
                else: # Is Put
                    stop_price = current_price + (last_atr * atr_settings['atr_multiplier'])
                    self.atr_stop_prices[conId] = min(self.atr_stop_prices.get(conId, float('inf')), stop_price)
                    if current_price > self.atr_stop_prices[conId]:
                        exit_reason = f"ATR Trailing Stop Hit ({current_price:.2f} > {self.atr_stop_prices[conId]:.2f})"

            elif exit_type == "pullback_stop" and profile['exit_strategy']['trail_method'] == 'pullback_percent':
                pullback_pct = profile['exit_strategy']['trail_settings']['pullback_percent']
                
                if is_call:
                    trailing_high = self.trailing_highs[conId]
                    stop_price = trailing_high * (1 - (pullback_pct / 100))
                    if current_price < stop_price:
                        exit_reason = f"Pullback Stop Hit ({pullback_pct}%)"
                else: # Is Put
                    trailing_low = self.trailing_highs[conId] # Note: still using trailing_highs for this logic
                    stop_price = trailing_low * (1 + (pullback_pct / 100))
                    if current_price > stop_price:
                        exit_reason = f"Pullback Stop Hit ({pullback_pct}%)"

            # ... (RSI and PSAR logic as before)
            pass

        if exit_reason:
            logging.info(f"Dynamic exit triggered for {position['contract'].localSymbol}. Reason: {exit_reason}")
            await self._execute_close_trade(conId, exit_reason)

    async def _execute_close_trade(self, conId, reason):
        # ... (unchanged)
        pass

    def _cleanup_position_data(self, conId):
        """Helper to remove all data associated with a closed/ghost position."""
        self.open_positions.pop(conId, None)
        self.position_data_cache.pop(conId, None)
        self.tick_buffer.pop(conId, None)
        self.last_bar_timestamp.pop(conId, None)
        self.trailing_highs.pop(conId, None)
        self.atr_stop_prices.pop(conId, None)
        self.breakeven_activated.pop(conId, None)

    # =================================================================
    # --- UTILITY AND HELPER METHODS (COMPLETE) ---
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
        # ... (logic is unchanged and complete)
        pass

    def _get_profile_by_channel_id(self, channel_id):
        """NON-NEGOTIABLE FIX: Finds the correct profile for a given channel ID."""
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