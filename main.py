# main.py
import logging
import time
import queue
from threading import Lock
from datetime import datetime, timedelta, timezone

# Import project modules
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.telegram_notifier import TelegramNotifier
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from services.market_data_manager import MarketDataManager
from bot_engine.signal_processor import SignalProcessor
from bot_engine.position_monitor import PositionMonitor
from bot_engine.trade_executor import TradeExecutor

class MainApp:
    def __init__(self):
        from services.custom_logger import setup_logger
        setup_logger()

        self.config = Config()
        self.message_queue = queue.Queue()

        self.ib_interface = IBInterface(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_data_manager = MarketDataManager(self.ib_interface)
        self.position_monitor = PositionMonitor(self.ib_interface, self.notifier, self.market_data_manager)
        self.trade_executor = TradeExecutor(self.ib_interface, self.position_monitor, self.notifier)
        self.channel_states = self._initialize_channel_states()
        self.state_lock = Lock()
        
        self.signal_processor = SignalProcessor(
            self.config, self.sentiment_analyzer, self.trade_executor,
            self.channel_states, self.state_lock
        )
        
        self.discord_interface = DiscordInterface(self.config, self.message_queue)
        self.is_running = False

    def _initialize_channel_states(self):
        states = {}
        for profile in self.config.profiles:
            channel_id = profile["channel_id"]
            states[channel_id] = {"consecutive_losses": 0, "cooldown_until": None}
        logging.info(f"Initialized channel state tracker: {states}")
        return states

    def _handle_fill_event(self, trade, fill):
        """
        The central switchboard. This is the definitive, battle-hardened version
        that uses an intelligent retry loop to fetch TA data.
        """
        conId = fill.contract.conId
        side = fill.execution.side
        
        if side == 'BOT':
            pos_data = self.position_monitor.get_position_data(conId)
            if not pos_data or pos_data.get("status") != "pending_fill": return

            # --- THIS IS THE CRITICAL, "INTELLIGENT RETRY" FIX ---
            initial_indicators = None
            for i in range(3): # Try up to 3 times
                logging.info(f"Fill confirmed for {trade.contract.localSymbol}. Fetching initial TA indicators (Attempt {i+1}/3)...")
                initial_indicators = self.ib_interface.get_technical_indicators(trade.contract)
                if initial_indicators:
                    break # Success, exit the loop
                logging.warning(f"Attempt {i+1} failed to fetch TA data. Retrying in 2 seconds...")
                time.sleep(2)
            
            self.position_monitor.activate_monitoring(conId, fill.execution.price, initial_indicators)
            
            self.notifier.send_fill_confirmation(fill, pos_data.get("sentiment_score", 0.0), pos_data["profile"]["channel_name"])
            self.market_data_manager.subscribe_to_contract(trade.contract)

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

        elif side == 'SLD':
            self._process_exit_fill(fill)

    def _process_exit_fill(self, fill):
        # ... (This function is the same) ...
        conId = fill.contract.conId
        pos_data = self.position_monitor.get_position_data(conId)
        if not pos_data or not pos_data.get("entry_price"): return

        profit_loss = fill.execution.price - pos_data["entry_price"]
        channel_id = pos_data["profile"]["channel_id"]
        monitor_config = pos_data["profile"]["consecutive_loss_monitor"]
        
        if not monitor_config["enabled"]:
            self.position_monitor.remove_position(conId)
            return

        with self.state_lock:
            state = self.channel_states[channel_id]
            if profit_loss <= 0:
                state["consecutive_losses"] += 1
            else:
                state["consecutive_losses"] = 0
            if state["consecutive_losses"] >= monitor_config["max_losses"]:
                cooldown_minutes = monitor_config["cooldown_minutes"]
                state["cooldown_until"] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)
                self.notifier.send_message(f"🚨 *Kill Switch Activated* 🚨\n\nChannel: `{pos_data['profile']['channel_name']}`\nReason: Reached max consecutive losses ({monitor_config['max_losses']}).\nCooldown: `{cooldown_minutes}` minutes.")
                state["consecutive_losses"] = 0
        self.position_monitor.remove_position(conId)

    def run(self):
        # ... (This function is the same) ...
        self.is_running = True
        try:
            self.ib_interface.connect()
            self.ib_interface.on_fill_callback = self._handle_fill_event
            self.position_monitor.start_monitoring()
            self.discord_interface.start_polling()
            logging.info("Main application loop started. Bot is now live.")
            while self.is_running:
                self.ib_interface.ib.sleep(1)
                try:
                    message_data = self.message_queue.get_nowait()
                    self.signal_processor.process_message(message_data)
                except queue.Empty:
                    pass
                if self.config.oversold_monitor_enabled:
                    positions = self.ib_interface.get_all_positions()
                    for pos in positions:
                        if pos.position < 0:
                            logging.critical(f"OVERSOLD CONDITION DETECTED: {pos.contract.localSymbol} shows position of {pos.position}.")
                            self.notifier.send_message(f"🆘 *OVERSOLD POSITION DETECTED* 🆘\n\nEmergency flatten initiated.")
                            self.stop()
                            self.ib_interface.flatten_all_positions()
                            return
        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
        except Exception as e:
            logging.error(f"A critical error occurred in the main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        # ... (This function is the same) ...
        if not self.is_running: return
        logging.info("Shutting down bot...")
        self.is_running = False
        self.discord_interface.stop_polling()
        self.position_monitor.stop_monitoring()
        self.ib_interface.disconnect()
        logging.info("Bot has been shut down.")

if __name__ == "__main__":
    app = MainApp()
    app.run()

