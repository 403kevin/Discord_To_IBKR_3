# main.py
import logging
import time
import queue
from threading import Lock, Event
from datetime import datetime, timezone

# Import project modules
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.telegram_notifier import TelegramNotifier
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from services.market_data_manager import MarketDataManager
from bot_engine.signal_processor import SignalProcessor
from bot_engine.trade_executor import TradeExecutor
# NOTE: We NO LONGER import PositionMonitor, as its logic is now inside MainApp

class MainApp:
    """
    The main application class. This definitive "Single Operator" version
    absorbs the role of the PositionMonitor, running all critical IBKR
    interactions (monitoring, TA checks) in the main thread to prevent
    event loop conflicts, based on the proven architecture of the original script.
    """
    def __init__(self):
        from services.custom_logger import setup_logger
        setup_logger()

        self.config = Config()
        self.message_queue = queue.Queue()
        self.shutdown_event = Event() # The shared emergency cord

        self.ib_interface = IBInterface(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_data_manager = MarketDataManager(self.ib_interface)
        
        # --- The Position Monitor logic is now part of the MainApp ---
        self._live_positions = {}
        self._pm_lock = Lock()
        
        self.trade_executor = TradeExecutor(self.ib_interface, self, self.notifier) # Passes itself (MainApp)
        self.channel_states = self._initialize_channel_states()
        self.state_lock = Lock()
        
        self.signal_processor = SignalProcessor(
            self.config, self.sentiment_analyzer, self.trade_executor,
            self.channel_states, self.state_lock
        )
        
        self.discord_interface = DiscordInterface(self.config, self.message_queue, self.shutdown_event)
        self.is_running = True # Controlled by the shutdown_event now

    def _initialize_channel_states(self):
        states = {}
        for profile in self.config.profiles:
            channel_id = profile["channel_id"]
            states[channel_id] = {"consecutive_losses": 0, "cooldown_until": None}
        return states

    # --- Position Monitor methods are now inside MainApp ---
    def add_position_to_monitor(self, conId, entry_trade, profile, sentiment_score):
        with self._pm_lock:
            if conId in self._live_positions: return
            logging.info(f"Position for {entry_trade.contract.localSymbol} (conId: {conId}) is now pending, awaiting mission briefing.")
            self._live_positions[conId] = {
                "entry_trade": entry_trade, "profile": profile,
                "sentiment_score": sentiment_score, "safety_net_settings": profile["safety_net"],
                "entry_price": None, "entry_timestamp": None,
                "high_water_mark": 0, "trailing_stop_price": 0,
                "breakeven_set": False, "status": "pending_fill",
                "initial_indicators": None, "previous_rsi": None
            }

    def activate_monitoring(self, conId, entry_price, initial_indicators):
        with self._pm_lock:
            if conId not in self._live_positions: return None
            pos_data = self._live_positions[conId]
            pos_data["entry_price"] = entry_price
            pos_data["high_water_mark"] = entry_price
            pos_data["entry_timestamp"] = datetime.now(timezone.utc)
            pos_data["initial_indicators"] = initial_indicators
            pos_data["status"] = "live"
            logging.info(f"Monitoring fully activated for conId {conId} at entry price ${entry_price:.2f}")
            if initial_indicators:
                logging.info(f"[MISSION BRIEFING] {pos_data['entry_trade'].contract.localSymbol} | Initial ATR: {initial_indicators.get('atr', 'N/A'):.2f}, PSAR: {initial_indicators.get('psar', 'N/A'):.2f}")
            else:
                logging.warning(f"No initial indicators provided for {pos_data['entry_trade'].contract.localSymbol}. Dynamic exits will use fallback.")
            return pos_data

    def get_position_data(self, conId):
        with self._pm_lock:
            return self._live_positions.get(conId)

    def remove_position(self, conId):
        with self._pm_lock:
            if conId in self._live_positions:
                self._live_positions.pop(conId)

    def _monitor_positions_loop(self):
        """The new monitoring logic, running safely in the main thread."""
        with self._pm_lock:
            active_conIds = list(self._live_positions.keys())
        
        for conId in active_conIds:
            self._check_single_position(conId)

    def _check_single_position(self, conId):
        with self._pm_lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live": return

        ticker = self.market_data_manager.get_ticker(conId)
        if not ticker or not ticker.last: return

        current_price = ticker.last
        pos_data["high_water_mark"] = max(pos_data["high_water_mark"], current_price)
        exit_strategy = pos_data["profile"]["exit_strategy"]
        contract_symbol = ticker.contract.localSymbol
        indicators = pos_data["initial_indicators"]
        
        # Exit Logic...
        # Time-Based Exit
        timeout_minutes = exit_strategy["timeout_exit_minutes"]
        time_in_trade = (datetime.now(timezone.utc) - pos_data["entry_timestamp"]).total_seconds() / 60
        if time_in_trade > timeout_minutes:
            self._execute_exit(conId, f"Timeout ({timeout_minutes}m)")
            return
        
        # Momentum-Based Early Exits
        momentum_exits = exit_strategy.get("momentum_exits", {})
        if indicators and momentum_exits.get("psar_enabled"):
             if indicators['psar'] > current_price:
                 self._execute_exit(conId, "PSAR Flip")
                 return
        
        # Trailing Stop
        new_trailing_stop_price = self._calculate_trailing_stop(current_price, pos_data, indicators)
        pos_data["trailing_stop_price"] = new_trailing_stop_price

        if current_price <= pos_data["trailing_stop_price"]:
            self._execute_exit(conId, f"Trailing Stop ({pos_data['trailing_stop_price']:.2f})")
            return
        
        logging.info(f"[MONITOR] {contract_symbol} | Price: {current_price:.2f} | High: {pos_data['high_water_mark']:.2f} | Stop: {pos_data['trailing_stop_price']:.2f}")

    def _calculate_trailing_stop(self, current_price, pos_data, indicators):
        # ... (This logic is identical to the old position_monitor) ...
        pass

    def _execute_exit(self, conId, reason):
        # ... (This logic is identical to the old position_monitor) ...
        pass

    def _handle_fill_event(self, trade, fill):
        # ... (This logic is identical to the last version) ...
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

