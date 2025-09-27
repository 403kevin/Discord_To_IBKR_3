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
        self.global_cooldown_until = datetime.now() # The "Pause Button" timestamp

        # Real-time data management
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        
        # Graceful exit state
        self.trailing_highs = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """The main entry point. Sets up concurrent tasks for all bot operations."""
        logging.info("Starting Signal Processor...")
        
        self.ib_interface.set_order_filled_callback(self._on_order_filled)
        
        await self._reconcile_state_with_broker()

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
                    entry_price = pos.avgCost
                    if pos.contract.secType == 'OPT':
                        entry_price /= 100

                    position_details = {
                        'contract': pos.contract,
                        'entry_price': entry_price,
                        'quantity': pos.position,
                        'entry_time': datetime.now(),
                        'channel_id': self._get_fallback_channel_id()
                    }
                    self.open_positions[pos.contract.conId] = position_details
                    self.trailing_highs[pos.contract.conId] = entry_price
                    self.breakeven_activated[pos.contract.conId] = False
                    logging.info(f"Adopted position: {pos.position} of {pos.contract.localSymbol}")

        self.state_manager.save_state(self.open_positions, self.processed_message_ids)
        logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")

    async def _poll_discord_for_signals(self):
        """Task to continuously poll Discord for new signals."""
        while not self._shutdown_event.is_set():
            now = datetime.now()
            
            if now < self.global_cooldown_until:
                await asyncio.sleep(self.config.delay_after_full_cycle)
                continue

            if self._is_eod():
                await self.flatten_all_positions()
                await self.shutdown()
                break

            for profile in self.config.profiles:
                if not profile['enabled']:
                    continue

                channel_id = profile['channel_id']
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
        processed_something_new = False

        for msg_id, msg_content, msg_timestamp in messages:
            if msg_id in self.processed_message_ids:
                continue
            
            processed_something_new = True
            self.processed_message_ids.append(msg_id)

            now_utc = datetime.now(timezone.utc)
            signal_age = now_utc - msg_timestamp
            if signal_age.total_seconds() > self.config.signal_max_age_seconds:
                stale_message_count += 1
                continue

            logging.info(f"Processing new message {msg_id} from '{profile['channel_name']}'")

            if any(word.lower() in msg_content.lower() for word in self.config.buzzwords_ignore):
                logging.debug(f"Message {msg_id} ignored due to buzzword.")
                continue

            parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
            
            if not isinstance(parsed_signal, dict):
                logging.debug(f"Message {msg_id} did not parse into a valid signal dictionary.")
                continue

            sentiment_score = None
            if self.config.sentiment_filter['enabled']:
                sentiment_score = self.sentiment_analyzer.get_sentiment_score(msg_content)
                is_call = parsed_signal['contract_type'].upper() == 'CALL'
                
                threshold = self.config.sentiment_filter['sentiment_threshold'] if is_call else self.config.sentiment_filter['put_sentiment_threshold']
                
                if (is_call and sentiment_score < threshold) or (not is_call and sentiment_score > threshold):
                    veto_reason = f"Sentiment score {sentiment_score:.4f} is outside threshold {threshold} for a {parsed_signal['contract_type']}."
                    logging.warning(f"Trade VETOED for {parsed_signal['ticker']}. Reason: {veto_reason}")
                    trade_info = {
                        'source_channel': profile['channel_name'],
                        'contract_details': f"{parsed_signal['ticker']} {parsed_signal['expiry_date']} {parsed_signal['strike']}{parsed_signal['contract_type'][0].upper()}",
                        'reason': veto_reason
                    }
                    await self.telegram_interface.send_trade_notification(trade_info, "VETOED")
                    continue
            
            await self._execute_trade_from_signal(parsed_signal, profile, sentiment_score)
        
        if stale_message_count > 0:
            logging.debug(f"Ignored {stale_message_count} stale message(s) from this poll cycle.")

        if processed_something_new:
            self.state_manager.save_state(self.open_positions, self.processed_message_ids)
            logging.debug("Updated processed message ID cache to state file.")

    async def _execute_trade_from_signal(self, signal, profile, sentiment_score):
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
                order.sentiment_score = sentiment_score
                logging.info(f"Successfully placed order for {quantity} of {contract.localSymbol} from channel {profile['channel_name']}")

        except Exception as e:
            logging.error(f"An error occurred during trade execution: {e}", exc_info=True)

    async def _on_order_filled(self, trade):
        """
        Callback executed by IBInterface when an order is filled.
        """
        contract = trade.contract
        order = trade.order
        channel_id = getattr(order, 'channel_id', None)
        sentiment_score = getattr(order, 'sentiment_score', None)

        if channel_id is None:
            logging.warning(f"Could not determine originating channel for fill of {contract.localSymbol}. Using fallback.")
            channel_id = self._get_fallback_channel_id()
        
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
        cooldown_seconds = self.config.cooldown_after_trade_seconds
        if cooldown_seconds > 0:
            self.global_cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
            logging.info(f"Global trade cooldown initiated. No new signals will be processed for {cooldown_seconds} seconds.")
        
        # ... (Rest of post-fill actions are unchanged)
        pass

    async def _process_market_data_stream(self):
        """Task to continuously process real-time market data from the queue."""
        pass

    async def _resample_ticks_to_bar(self, ticker):
        """Collects ticks and resamples them into time-based bars for analysis."""
        pass

    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        pass

    async def _execute_close_trade(self, conId, reason):
        """Closes a position and updates the state."""
        pass

    def _cleanup_position_data(self, conId):
        """Helper to remove all data associated with a closed/ghost position."""
        pass

    async def flatten_all_positions(self):
        """Closes all open positions. Triggered at EOD."""
        pass

    def _is_eod(self):
        """Checks if the current time is past the EOD close time, using timezone-aware logic."""
        pass

    def _get_profile_by_channel_id(self, channel_id):
        """Finds the correct profile for a given channel ID."""
        pass
    
    def _get_fallback_channel_id(self):
        """Finds the first enabled profile to use as a fallback for adopted positions."""
        pass

    async def _resubscribe_to_open_positions(self):
        """Resubscribes to market data for all positions loaded from state."""
        pass