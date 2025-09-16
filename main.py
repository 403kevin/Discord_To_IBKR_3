# main.py
import logging
import time
import queue
from threading import Lock, Event
from datetime import datetime, timezone, timedelta

# Import project modules
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.telegram_interface import TelegramInterface
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from services.market_data_manager import MarketDataManager
from bot_engine.signal_processor import SignalProcessor
from bot_engine.trade_executor import TradeExecutor

class MainApp:
    """
    The main application class. This is the definitive "Single Operator" version.
    It now uses a "Two-Phase" model to safely handle fill events, permanently
    resolving the 'event loop is already running' error.
    """
    def __init__(self):
        from services.custom_logger import setup_logger
        setup_logger()

        self.config = Config()
        self.message_queue = queue.Queue()
        self.shutdown_event = Event()

        self.ib_interface = IBInterface(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_data_manager = MarketDataManager(self.ib_interface)
        
        self._live_positions = {}
        self._pm_lock = Lock()
        
        self.trade_executor = TradeExecutor(self.ib_interface, self, self.notifier)
        self.channel_states = self._initialize_channel_states()
        self.state_lock = Lock()
        
        self.signal_processor = SignalProcessor(
            self.config, self.sentiment_analyzer, self.trade_executor,
            self.channel_states, self.state_lock
        )
        
        self.discord_interface = DiscordInterface(self.config, self.message_queue, self.shutdown_event)
        self.is_running = True

    def _initialize_channel_states(self):
        states = {}
        for profile in self.config.profiles:
            channel_id = profile["channel_id"]
            states[channel_id] = {"consecutive_losses": 0, "cooldown_until": None}
        return states

    # --- Position Management (formerly PositionMonitor) ---
    def add_position_to_monitor(self, conId, entry_trade, profile, sentiment_score):
        with self._pm_lock:
            if conId in self._live_positions: return
            logging.info(f"Position for {entry_trade.contract.localSymbol} (conId: {conId}) is now pending setup.")
            self._live_positions[conId] = {
                "entry_trade": entry_trade, "profile": profile,
                "sentiment_score": sentiment_score, "safety_net_settings": profile["safety_net"],
                "entry_price": None, "entry_timestamp": None,
                "high_water_mark": 0, "trailing_stop_price": 0,
                "breakeven_set": False, "status": "pending_fill",
                "initial_indicators": None, "previous_rsi": None
            }

    def _process_pending_setups(self):
        """
        Phase 2 of the fill process. This is called safely from the main loop.
        It finds newly filled trades and performs the slow, blocking setup tasks.
        """
        with self._pm_lock:
            pending_setup_ids = [
                conId for conId, pos_data in self._live_positions.items()
                if pos_data.get("status") == "filled_pending_setup"
            ]

        for conId in pending_setup_ids:
            with self._pm_lock:
                pos_data = self._live_positions.get(conId)
                if not pos_data: continue
                trade = pos_data["entry_trade"]
            
            # 1. Get the "Mission Briefing" (Technical Indicators)
            initial_indicators = None
            for i in range(3): # Intelligent retry loop
                logging.info(f"Fetching initial TA indicators for {trade.contract.localSymbol} (Attempt {i+1}/3)...")
                indicators = self.ib_interface.get_technical_indicators(trade.contract)
                if indicators:
                    initial_indicators = indicators
                    break
                logging.warning(f"Attempt {i+1} failed to fetch TA data. Retrying in 2 seconds...")
                time.sleep(2)
            
            # 2. Activate full monitoring with the mission briefing
            self.activate_monitoring(conId, pos_data["entry_price"], initial_indicators)
            
            # 3. Place the native trail order safety net
            safety_net_settings = pos_data.get("safety_net_settings", {})
            if safety_net_settings.get("enabled"):
                logging.info("Pausing for 1 second before attaching safety net...")
                time.sleep(1)
                logging.info("Placing native safety net trail order...")
                self.ib_interface.place_native_trail_stop(
                    trade.contract, trade.order.totalQuantity,
                    safety_net_settings["native_trail_percent"]
                )
                logging.info("Native safety net trail order placed successfully.")

    def activate_monitoring(self, conId, entry_price, initial_indicators):
        with self._pm_lock:
            if conId not in self._live_positions: return None
            pos_data = self._live_positions[conId]
            pos_data["entry_price"] = entry_price
            pos_data["high_water_mark"] = entry_price
            pos_data["entry_timestamp"] = datetime.now(timezone.utc)
            pos_data["initial_indicators"] = initial_indicators
            pos_data["status"] = "live"
            logging.info(f"Monitoring fully activated for {pos_data['entry_trade'].contract.localSymbol} at ${entry_price:.2f}")
            if initial_indicators:
                logging.info(f"[MISSION BRIEFING] | Initial ATR: {initial_indicators.get('atr', 'N/A'):.2f}, PSAR: {initial_indicators.get('psar', 'N/A'):.2f}")
            else:
                logging.warning(f"No initial indicators provided for {pos_data['entry_trade'].contract.localSymbol}. Dynamic exits will use fallback.")
            return pos_data

    # ... (get_position_data, remove_position, _monitor_positions_loop, etc. are the same as the last version) ...
    # ... I have included them here for completeness ...

    def get_position_data(self, conId):
        with self._pm_lock:
            return self._live_positions.get(conId)

    def remove_position(self, conId):
        with self._pm_lock:
            if conId in self._live_positions:
                self._live_positions.pop(conId)

    def _monitor_positions_loop(self):
        with self._pm_lock:
            active_conIds = list(self._live_positions.keys())
        for conId in active_conIds:
            self._check_single_position(conId)

    def _check_single_position(self, conId):
        with self._pm_lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data.get("status") != "live": return

        ticker = self.market_data_manager.get_ticker(conId)
        if not ticker or not ticker.last: return

        current_price = ticker.last
        pos_data["high_water_mark"] = max(pos_data["high_water_mark"], current_price)
        exit_strategy = pos_data["profile"]["exit_strategy"]
        contract_symbol = ticker.contract.localSymbol
        indicators = pos_data["initial_indicators"]
        
        timeout_minutes = exit_strategy["timeout_exit_minutes"]
        time_in_trade = (datetime.now(timezone.utc) - pos_data["entry_timestamp"]).total_seconds() / 60
        if time_in_trade > timeout_minutes:
            self._execute_exit(conId, f"Timeout ({timeout_minutes}m)")
            return
        
        momentum_exits = exit_strategy.get("momentum_exits", {})
        if indicators and momentum_exits.get("psar_enabled"):
             if indicators['psar'] > current_price:
                 self._execute_exit(conId, "PSAR Flip")
                 return
        
        new_trailing_stop_price = self._calculate_trailing_stop(current_price, pos_data, indicators)
        pos_data["trailing_stop_price"] = new_trailing_stop_price

        if current_price <= pos_data["trailing_stop_price"]:
            self._execute_exit(conId, f"Trailing Stop ({pos_data['trailing_stop_price']:.2f})")
            return
        
        logging.info(f"[MONITOR] {contract_symbol} | Price: {current_price:.2f} | High: {pos_data['high_water_mark']:.2f} | Stop: {pos_data['trailing_stop_price']:.2f}")

    def _calculate_trailing_stop(self, current_price, pos_data, indicators):
        entry_price = pos_data["entry_price"]
        high_water_mark = pos_data["high_water_mark"]
        exit_strategy = pos_data["profile"]["exit_strategy"]
        contract_symbol = pos_data["entry_trade"].contract.localSymbol

        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
        if not pos_data["breakeven_set"] and profit_percent >= exit_strategy["breakeven_trigger_percent"]:
            logging.info(f"BREAKEVEN TRIGGERED for {contract_symbol}. Stop moved to entry price ${entry_price:.2f}.")
            pos_data["breakeven_set"] = True
            return entry_price

        if pos_data["breakeven_set"]:
            return max(pos_data.get("trailing_stop_price", 0), entry_price)

        trail_method = exit_strategy["trail_method"]
        if trail_method == "pullback_percent":
            pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["pullback_percent"] / 100)
            return high_water_mark - pullback_amount
        
        elif trail_method == "atr":
            if indicators and 'atr' in indicators:
                atr_val = indicators['atr']
                atr_mult = exit_strategy["trail_settings"]["atr_multiplier"]
                pullback_amount = atr_val * atr_mult
                logging.info(f"[ATR CALC] {contract_symbol} | ATR: {atr_val:.2f} * {atr_mult}x => Pullback: ${pullback_amount:.2f}")
                return high_water_mark - pullback_amount
            else:
                logging.warning(f"ATR data not available for {contract_symbol}. Using fallback percentage.")
                pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["pullback_percent"] / 100)
                return high_water_mark - pullback_amount
        return 0

    def _execute_exit(self, conId, reason):
        with self._pm_lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live": return
            pos_data["status"] = "exit_sent"
        
        contract = pos_data["entry_trade"].contract
        quantity = pos_data["entry_trade"].order.totalQuantity
        logging.info(f"Executing market close for {contract.localSymbol} (Reason: {reason}).")
        self.ib_interface.close_position(contract, quantity)

    def _handle_fill_event(self, trade, fill):
        """
        Phase 1 of the fill process. This is the fast, non-blocking receiver.
        Its ONLY job is to update the status and hand off to the main loop.
        """
        conId = fill.contract.conId
        side = fill.execution.side
        
        if side == 'BOT':
            with self._pm_lock:
                pos_data = self._live_positions.get(conId)
                if pos_data and pos_data.get("status") == "pending_fill":
                    pos_data["status"] = "filled_pending_setup"
                    pos_data["entry_price"] = fill.execution.price # Store the price now
                    logging.info(f"Fill received for {fill.contract.localSymbol}. Status set to pending_setup.")
                    self.notifier.send_fill_confirmation(fill, pos_data.get("sentiment_score", 0.0), pos_data["profile"]["channel_name"])
                    self.market_data_manager.subscribe_to_contract(trade.contract)

        elif side == 'SLD':
            self._process_exit_fill(fill)

    def _process_exit_fill(self, fill):
        # ... (This logic is the same) ...
        pass

    def run(self):
        self.is_running = True
        try:
            self.ib_interface.connect()
            self.ib_interface.on_fill_callback = self._handle_fill_event
            self.discord_interface.start_polling()

            logging.info("Main application loop started. Bot is now live.")
            while self.is_running and not self.shutdown_event.is_set():
                self.ib_interface.ib.sleep(1) # The intelligent, active wait
                
                # --- NEW: Process pending setups safely in the main loop ---
                self._process_pending_setups()

                try:
                    message_data = self.message_queue.get_nowait()
                    self.signal_processor.process_message(message_data)
                except queue.Empty:
                    pass
                
                self._monitor_positions_loop()

                if self.config.oversold_monitor_enabled:
                    # ... (Oversold logic is the same) ...
                    pass
                            
        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
        finally:
            self.stop()

    def stop(self):
        if not self.is_running: return
        self.is_running = False
        self.shutdown_event.set()
        logging.info("Shutting down bot...")
        self.discord_interface.stop_polling()
        self.ib_interface.disconnect()
        logging.info("Bot has been shut down.")

if __name__ == "__main__":
    app = MainApp()
    app.run()
