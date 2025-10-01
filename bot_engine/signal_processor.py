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

        # Create each loop as a separate, concurrent background task.
        tasks = [
            asyncio.create_task(self._poll_discord_for_signals()),
            asyncio.create_task(self._process_market_data_stream()),
            asyncio.create_task(self._reconcile_positions_periodically())
        ]
        
        # This will now run forever until a task finishes or is cancelled.
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

    async def _execute_trade_from_signal(self, signal, profile, sentiment_score):
        """Validates and executes a single trade."""
        pass # Placeholder

    async def _on_order_filled(self, trade):
        """Callback executed by IBInterface when an order is filled."""
        pass # Placeholder

    async def _post_fill_actions(self, trade, position_details, sentiment_score, profile):
        """Actions to take after an order is confirmed filled."""
        pass # Placeholder

    async def _process_market_data_stream(self):
        """Task to continuously process real-time market data from the queue."""
        pass # Placeholder

    async def _resample_ticks_to_bar(self, ticker):
        """Collects ticks and resamples them into time-based bars for analysis."""
        pass # Placeholder

    async def _evaluate_dynamic_exit(self, conId):
        """Evaluates all configured dynamic exit strategies for a position."""
        pass # Placeholder

    async def _execute_close_trade(self, conId, reason):
        """Closes a position and updates the state."""
        pass # Placeholder

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
        pass # Placeholder

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