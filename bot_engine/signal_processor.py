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

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """The main entry point. Sets up concurrent tasks for all bot operations."""
        logging.info("Starting Signal Processor...")
        
        await self._resubscribe_to_open_positions()

        tasks = [
            self._poll_discord_for_signals(),
            self._process_market_data_stream(),
            self._reconcile_positions_periodically() # NEW: The "Reality Check" loop
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
        """
        Periodically syncs the bot's internal state with the broker's actual portfolio
        to prevent state desynchronization (managing "ghost" trades).
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.reconciliation_interval_seconds)
                logging.info("--- Starting periodic position reconciliation ---")
                
                broker_positions = await self.ib_interface.get_open_positions()
                broker_conIds = {pos.contract.conId for pos in broker_positions}
                internal_conIds = set(self.open_positions.keys())

                # Positions broker has but bot doesn't (manual trade, etc.)
                for pos in broker_positions:
                    if pos.contract.conId not in internal_conIds:
                        logging.warning(f"Reconciliation: Found untracked position for {pos.contract.localSymbol}. Bot is not managing this position.")
                        # Future enhancement: decide whether to add it to the bot's management.

                # Positions bot has but broker doesn't (manual close, etc.)
                ghost_positions = internal_conIds - broker_conIds
                if ghost_positions:
                    logging.warning(f"Reconciliation: Found {len(ghost_positions)} ghost position(s). Removing from internal state.")
                    for conId in ghost_positions:
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
            # ... (rest of the polling logic is unchanged)
            await asyncio.sleep(self.config.delay_after_full_cycle)

    async def _process_new_signals(self, messages, profile):
        """Processes a batch of new messages for a given profile."""
        for msg_id, msg_content, msg_timestamp in messages:
            # ... (rest of signal processing logic is unchanged)
            # 4. Execute the trade, now passing the originating channel_id
            await self._execute_trade_from_signal(parsed_signal, profile, msg_id, profile['channel_id'])

    async def _execute_trade_from_signal(self, signal, profile, msg_id, channel_id):
        """Validates and executes a single trade, now aware of its origin."""
        # ... (sizing logic is unchanged)
            
            order = await self.ib_interface.place_order(contract, 'MKT', quantity)
            if order:
                # The _on_order_filled callback will handle the rest, but we need
                # a temporary way to link the fill to the profile.
                self.ib_interface.ib.pendingTradesEvent += self._create_fill_handler(channel_id)
                
                # ... (Telegram notification is unchanged)

    def _create_fill_handler(self, channel_id):
        """Creates a temporary event handler to link a fill to its originating channel."""
        def handler(trade):
            if not trade.isDone():
                return
            # Now we can pass the channel_id to the main fill handler
            self._on_order_filled(trade, channel_id)
            # Clean up the handler to prevent it from firing on other fills
            self.ib_interface.ib.pendingTradesEvent -= handler
        return handler

    def _on_order_filled(self, trade, channel_id):
        """Callback executed when an order is filled, now with channel context."""
        contract = trade.contract
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
        self.trailing_highs[contract.conId] = fill_price # Initialize for pullback stop

        asyncio.create_task(self._post_fill_actions(trade, position_details))

    # ... (post_fill_actions, market data processing, resampling are largely unchanged)

    # =================================================================
    # --- CORE LOGIC: Dynamic Exit Evaluation (HEAVILY UPGRADED) ---
    # =================================================================

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

        # Update trailing high for pullback stops
        self.trailing_highs[conId] = max(self.trailing_highs.get(conId, 0), current_price)
        
        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_reason: break

            # --- WIRED IN: Breakeven Logic ---
            if exit_type == "breakeven":
                pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
                if pnl_percent >= profile['exit_strategy']['breakeven_trigger_percent']:
                    # This is a conceptual implementation of a software-based stop.
                    if current_price <= position['entry_price']:
                        exit_reason = "Breakeven Stop Hit"
            
            # --- WIRED IN: Graceful Exits ---
            if exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                atr_settings = profile['exit_strategy']['trail_settings']
                data.ta.atr(length=atr_settings['atr_period'], append=True)
                last_atr = data[f'ATRr_{atr_settings["atr_period"]}'].iloc[-1]
                
                # Calculate and update the ATR stop price
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
                trailing_high = self.trailing_highs[conId]
                
                if is_call:
                    stop_price = trailing_high * (1 - (pullback_pct / 100))
                    if current_price < stop_price:
                        exit_reason = f"Pullback Stop Hit ({pullback_pct}%)"
                else: # Is Put
                    # For puts, we trail from the low
                    # This logic would need to track trailing_low instead of trailing_high
                    pass # Placeholder for Put pullback logic

            # --- Momentum Exits (logic remains the same) ---
            elif exit_type == "rsi_hook" and profile['exit_strategy']['momentum_exits']['rsi_hook_enabled']:
                # ... (RSI logic as before)
                pass
            elif exit_type == "psar_flip" and profile['exit_strategy']['momentum_exits']['psar_enabled']:
                # ... (PSAR logic as before)
                pass

        if exit_reason:
            logging.info(f"Dynamic exit triggered for {position['contract'].localSymbol}. Reason: {exit_reason}")
            await self._execute_close_trade(conId, exit_reason)

    # ... (execute_close_trade and flatten_all_positions are unchanged)

    def _cleanup_position_data(self, conId):
        """Helper to remove all data associated with a closed/ghost position."""
        self.open_positions.pop(conId, None)
        self.position_data_cache.pop(conId, None)
        self.tick_buffer.pop(conId, None)
        self.last_bar_timestamp.pop(conId, None)
        self.trailing_highs.pop(conId, None)
        self.atr_stop_prices.pop(conId, None)

    # =================================================================
    # --- UTILITY AND HELPER METHODS (UPGRADED) ---
    # =================================================================

    def _get_profile_by_channel_id(self, channel_id):
        """NON-NEGOTIABLE FIX: Finds the correct profile for a given channel ID."""
        for profile in self.config.profiles:
            if profile['channel_id'] == str(channel_id): # Compare as strings for safety
                return profile
        logging.warning(f"Could not find a profile for channel ID {channel_id}")
        return None

    # ... (other helpers are unchanged)