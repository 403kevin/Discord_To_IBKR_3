# main.py
import logging
import time
from threading import Lock
from datetime import datetime, timedelta, timezone

# Import project modules
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.telegram_notifier import TelegramNotifier
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from bot_engine.signal_processor import SignalProcessor
from bot_engine.position_monitor import PositionMonitor
from bot_engine.trade_executor import TradeExecutor


class MainApp:
    def __init__(self):
        self.config = Config()
        self.ib_interface = IBInterface(self.config)
        self.notifier = TelegramNotifier(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()

        # --- NEW: State tracker for the kill switch ---
        self.channel_states = self._initialize_channel_states()
        self.state_lock = Lock()

        self.position_monitor = PositionMonitor(self.ib_interface, self.notifier)
        self.trade_executor = TradeExecutor(self.ib_interface, self.position_monitor)

        self.signal_processor = SignalProcessor(
            self.config,
            self.sentiment_analyzer,
            self.trade_executor,
            self.channel_states,  # Pass the state tracker to the decision maker
            self.state_lock
        )

        self.discord_interface = DiscordInterface(self.config, self.signal_processor.process_message)
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
        The central switchboard for all confirmed fills from the broker.
        This now handles both entries and exits.
        """
        conId = fill.contract.conId
        side = fill.execution.side

        if side == 'BOT':  # --- Handle Entry Fills ---
            if self.position_monitor.is_position_active(conId):
                entry_price = fill.execution.price
                pos_data = self.position_monitor.update_position_on_fill(conId, entry_price)
                if pos_data:
                    self.notifier.send_fill_confirmation(fill, pos_data["sentiment_score"],
                                                         pos_data["profile"]["channel_name"])

        elif side == 'SLD':  # --- NEW: Handle Exit Fills ---
            self._process_exit_fill(fill)

    def _process_exit_fill(self, fill):
        """Contains the logic for the consecutive loss kill switch."""
        conId = fill.contract.conId
        exit_price = fill.execution.price

        # Get the original entry data from the position monitor
        pos_data = self.position_monitor.get_position_data(conId)
        if not pos_data or not pos_data.get("entry_price"):
            logging.warning(
                f"Received an exit fill for an untracked or unconfirmed position (conId: {conId}). Cannot process P/L.")
            return

        entry_price = pos_data["entry_price"]
        profit_loss = exit_price - entry_price

        # Now, update the channel state
        channel_id = pos_data["profile"]["channel_id"]
        monitor_config = pos_data["profile"]["consecutive_loss_monitor"]

        if not monitor_config["enabled"]:
            # Remove the position from monitoring if the kill switch is off for this channel
            self.position_monitor.remove_position(conId)
            return

        with self.state_lock:
            state = self.channel_states[channel_id]
            if profit_loss < 0:  # It's a loss
                state["consecutive_losses"] += 1
                logging.warning(
                    f"Loss recorded for channel {channel_id}. Consecutive losses: {state['consecutive_losses']}")
            else:  # It's a win or breakeven
                if state["consecutive_losses"] > 0:
                    logging.info(
                        f"Win recorded for channel {channel_id}. Resetting consecutive loss count from {state['consecutive_losses']} to 0.")
                    state["consecutive_losses"] = 0

            # Check if the kill switch is triggered
            if state["consecutive_losses"] >= monitor_config["max_losses"]:
                cooldown_minutes = monitor_config["cooldown_minutes"]
                state["cooldown_until"] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)

                logging.critical(
                    f"INTERNAL KILL SWITCH TRIGGERED for channel {channel_id} ({pos_data['profile']['channel_name']}). Max losses of {monitor_config['max_losses']} reached. On cooldown for {cooldown_minutes} minutes.")
                self.notifier.send_message(
                    f"🚨 *Kill Switch Activated* 🚨\n\nChannel: `{pos_data['profile']['channel_name']}`\nReason: Reached max consecutive losses ({monitor_config['max_losses']}).\nCooldown: `{cooldown_minutes}` minutes.")

                # Reset counter after triggering
                state["consecutive_losses"] = 0

        # Finally, tell the position monitor this trade is officially over.
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
                # The main loop's primary job is now just to keep the app alive
                # and handle the emergency flatten check.

                if self.config.oversold_monitor_enabled:
                    positions = self.ib_interface.get_all_positions()
                    for pos in positions:
                        if pos.position < 0:
                            logging.critical(
                                f"OVERSOLD CONDITION DETECTED: {pos.contract.localSymbol} shows position of {pos.position}. INITIATING EMERGENCY FLATTEN.")
                            self.notifier.send_message(
                                f"🆘 *OVERSOLD POSITION DETECTED* 🆘\n\nEmergency flatten initiated for all positions.")
                            self.stop()  # Gracefully shut down normal operations
                            self.ib_interface.flatten_all_positions()
                            return  # Exit the application

                time.sleep(self.config.polling_interval_seconds)

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

