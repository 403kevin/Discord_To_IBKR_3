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
from services.market_data_manager import MarketDataManager # NEW: The Master Watchmaker
from bot_engine.signal_processor import SignalProcessor
from bot_engine.position_monitor import PositionMonitor
from bot_engine.trade_executor import TradeExecutor

class MainApp:
    def __init__(self):
        # Setup logging first, so all components can use it
        from services.custom_logger import setup_logger
        setup_logger()

        self.config = Config()
        self.message_queue = queue.Queue()

        self.ib_interface = IBInterface(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # --- NEW: Hire the Master Watchmaker ---
        self.market_data_manager = MarketDataManager(self.ib_interface)

        # The Position Monitor now gets the data manager to read prices from
        self.position_monitor = PositionMonitor(self.ib_interface, self.notifier, self.market_data_manager)
        
        self.trade_executor = TradeExecutor(
            self.ib_interface, 
            self.position_monitor, 
            self.notifier
        )

        self.channel_states = self._initialize_channel_states()
        self.state_lock = Lock()
        
        self.signal_processor = SignalProcessor(
            self.config,
            self.sentiment_analyzer,
            self.trade_executor,
            self.channel_states,
            self.state_lock
        )
        
        self.discord_interface = DiscordInterface(self.config, self.message_queue)
        self.is_running = False

    def _initialize_channel_states(self):
        """Creates the initial state tracker for each channel profile."""
        states = {}
        for profile in self.config.profiles:
            channel_id = profile["channel_id"]
            states[channel_id] = {
                "consecutive_losses": 0,
                "cooldown_until": None
            }
        logging.info(f"Initialized channel state tracker: {states}")
        return states

    def _handle_fill_event(self, trade, fill):
        """
        The central switchboard. This is the final, intelligent version that
        subscribes to market data and places the native trail order ONLY
        after a fill is confirmed.
        """
        conId = fill.contract.conId
        side = fill.execution.side
        
        if side == 'BOT': # --- Handle Entry Fills ---
            pos_data = self.position_monitor.update_position_on_fill(conId, fill.execution.price)
            if pos_data:
                self.notifier.send_fill_confirmation(fill, pos_data.get("sentiment_score", 0.0), pos_data["profile"]["channel_name"])

                # --- NEW: Tell the Watchmaker to subscribe to the data stream ---
                self.market_data_manager.subscribe_to_contract(trade.contract)

                # --- The "Listen, Then Act" fix for the safety net ---
                safety_net_settings = pos_data.get("safety_net_settings", {})
                if safety_net_settings.get("enabled"):
                    logging.info("Entry fill confirmed. Placing native safety net trail order...")
                    self.ib_interface.place_native_trail_stop(
                        trade.contract,
                        trade.order.totalQuantity,
                        safety_net_settings["native_trail_percent"]
                    )
                    logging.info("Native safety net trail order placed successfully.")

        elif side == 'SLD': # --- Handle Exit Fills ---
            self._process_exit_fill(fill)

    def _process_exit_fill(self, fill):
        """Contains the logic for the consecutive loss kill switch."""
        conId = fill.contract.conId
        pos_data = self.position_monitor.get_position_data(conId)
        if not pos_data or not pos_data.get("entry_price"):
            return

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
        """Starts all components and the main application loop."""
        self.is_running = True
        try:
            self.ib_interface.connect()
            self.ib_interface.on_fill_callback = self._handle_fill_event
            
            self.position_monitor.start_monitoring()
            self.discord_interface.start_polling()

            logging.info("Main application loop started. Bot is now live.")
            while self.is_running:
                try:
                    message_data = self.message_queue.get(timeout=1)
                    self.signal_processor.process_message(message_data)
                except queue.Empty:
                    pass

                # The oversold check should be less frequent to avoid spamming IBKR.
                # A proper implementation would use a timer. For now, it runs in the loop.
                if self.config.oversold_monitor_enabled:
                    positions = self.ib_interface.get_all_positions()
                    for pos in positions:
                        if pos.position < 0:
                            logging.critical(f"OVERSOLD CONDITION DETECTED: {pos.contract.localSymbol} shows position of {pos.position}. INITIATING EMERGENCY FLATTEN.")
                            self.notifier.send_message(f"🆘 *OVERSOLD POSITION DETECTED* 🆘\n\nEmergency flatten initiated for all positions.")
                            self.stop()
                            self.ib_interface.flatten_all_positions()
                            return
                
                time.sleep(0.1)

        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
        except Exception as e:
            logging.error(f"A critical error occurred in the main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Gracefully stops all components."""
        if not self.is_running:
            return
        logging.info("Shutting down bot...")
        self.is_running = False
        self.discord_interface.stop_polling()
        self.position_monitor.stop_monitoring()
        self.ib_interface.disconnect()
        logging.info("Bot has been shut down.")

if __name__ == "__main__":
    app = MainApp()
    app.run()

