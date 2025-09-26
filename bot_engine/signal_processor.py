import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
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
        
        # --- THE "STATE OF THE UNION" PROTOCOL ---
        # Reconcile state with the broker BEFORE subscribing to data for old positions.
        await self._reconcile_state_with_broker()

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

    async def _reconcile_state_with_broker(self):
        """
        Compares the bot's internal state with the broker's actual portfolio
        at startup to eliminate ghost positions from the state file.
        """
        logging.info("Performing initial state reconciliation with broker...")
        broker_positions = await self.ib_interface.get_open_positions()
        broker_conIds = {pos.contract.conId for pos in broker_positions}
        internal_conIds = set(self.open_positions.keys())

        ghost_positions = internal_conIds - broker_conIds
        if ghost_positions:
            logging.warning(f"Reconciliation: Found {len(ghost_positions)} ghost position(s) in state file. Removing.")
            for conId in ghost_positions:
                self._cleanup_position_data(conId)
        
        # Filter the internal positions to only keep ones the broker confirms exist.
        valid_positions = internal_conIds.intersection(broker_conIds)
        self.open_positions = {conId: pos for conId, pos in self.open_positions.items() if conId in valid_positions}

        # Save the cleaned state immediately
        self.state_manager.save_state(self.open_positions, self.processed_message_ids)
        logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")

    async def _poll_discord_for_signals(self):
        """Task to continuously poll Discord for new signals."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    async def _process_new_signals(self, messages, profile):
        """Processes a batch of new messages for a given profile."""
        stale_message_count = 0

        for msg_id, msg_content, msg_timestamp in messages:
            if msg_id in self.processed_message_ids:
                continue
            
            # --- THE REAL "AMNESIA VACCINE" ---
            # Mark the message as processed IMMEDIATELY. This is non-negotiable.
            self.processed_message_ids.append(msg_id)

            # --- THE STALE SIGNAL CHECK ---
            now_utc = datetime.now(timezone.utc)
            signal_age = now_utc - msg_timestamp
            if signal_age.total_seconds() > self.config.signal_max_age_seconds:
                stale_message_count += 1
                continue # Silently continue to the next message

            logging.info(f"Processing new message {msg_id} from '{profile['channel_name']}'")

            # ... (rest of the processing logic is correct and unchanged)

        if stale_message_count > 0:
            logging.debug(f"Ignored {stale_message_count} stale message(s) from this poll cycle.")

    async def _execute_trade_from_signal(self, signal, profile, sentiment_score):
        """Validates and executes a single trade, now with sentiment score for reporting."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    async def _on_order_filled(self, trade):
        """
        Callback executed by IBInterface when an order is filled.
        This is now a correctly defined async function.
        """
        contract = trade.contract
        order = trade.order
        channel_id = getattr(order, 'channel_id', None)
        sentiment_score = getattr(order, 'sentiment_score', None)

        if channel_id is None:
            logging.warning(f"Could not determine originating channel for fill of {contract.localSymbol}. Using first enabled profile.")
            for profile in self.config.profiles:
                if profile['enabled']:
                    channel_id = profile['channel_id']
                    break
        
        # --- THE API MISMATCH FIX ---
        fill_price = trade.orderStatus.avgFillPrice
        quantity = trade.orderStatus.filled
        
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

        await self._post_fill_actions(trade, position_details, sentiment_score)

    async def _post_fill_actions(self, trade, position_details, sentiment_score):
        """Actions to take after an order is confirmed filled, with full reporting data."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    # =================================================================
    # --- CORE LOGIC: Real-time Position Management ---
    # =================================================================
    
    async def _process_market_data_stream(self):
        """Task to continuously process real-time market data from the queue."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    async def _resample_ticks_to_bar(self, ticker):
        """Collects ticks and resamples them into time-based bars for analysis."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    async def _execute_close_trade(self, conId, reason):
        """Closes a position and updates the state."""
        # ... (This function's internal logic is correct and unchanged)
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
    # --- UTILITY AND HELPER METHODS ---
    # =================================================================

    async def flatten_all_positions(self):
        """Closes all open positions. Triggered at EOD."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    def _is_eod(self):
        """Checks if the current time is past the EOD close time, using timezone-aware logic."""
        # ... (This function's internal logic is correct and unchanged)
        pass

    def _get_profile_by_channel_id(self, channel_id):
        """Finds the correct profile for a given channel ID."""
        # ... (This function's internal logic is correct and unchanged)
        pass
    
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