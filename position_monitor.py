# bot_engine/position_monitor.py
import logging
import time
from threading import Thread, Lock
from datetime import datetime, timezone


class PositionMonitor:
    """
    The guardian of our live trades. This class runs in a dedicated thread
    to monitor all active positions, manage their state, and execute exit
    logic based on the strategy defined in the config.
    """

    def __init__(self, ib_interface, notifier):
        self.ib_interface = ib_interface
        self.notifier = notifier
        self._live_positions = {}
        self._lock = Lock()
        self._monitoring_thread = None
        self._is_monitoring = False

    def add_position_to_monitor(self, conId, entry_trade, profile, sentiment_score):
        """Called by the trade_executor to put a new position under watch."""
        with self._lock:
            if conId in self._live_positions:
                logging.warning(f"Attempted to add an already monitored position with conId: {conId}")
                return

            logging.info(
                f"Position for {entry_trade.contract.localSymbol} (conId: {conId}) is now pending, awaiting fill.")
            self._live_positions[conId] = {
                "entry_trade": entry_trade,
                "profile": profile,
                "sentiment_score": sentiment_score,
                "entry_price": None,
                "entry_timestamp": None,
                "high_water_mark": 0,
                "trailing_stop_price": 0,
                "breakeven_set": False,
                "status": "pending_fill"  # NEW: Status tracking
            }

    def update_position_on_fill(self, conId, entry_price):
        """Called by main.py's fill handler to activate full monitoring."""
        with self._lock:
            if conId not in self._live_positions:
                return None

            pos_data = self._live_positions[conId]
            pos_data["entry_price"] = entry_price
            pos_data["high_water_mark"] = entry_price
            pos_data["entry_timestamp"] = datetime.now(timezone.utc)
            pos_data["status"] = "live"  # NEW: Status is now live
            logging.info(f"Monitoring fully activated for conId {conId} at entry price ${entry_price:.2f}")
            return pos_data

    # --- NEW: Helper functions for main.py ---
    def get_position_data(self, conId):
        """Safely gets the data for a given position."""
        with self._lock:
            return self._live_positions.get(conId)

    def remove_position(self, conId):
        """Safely removes a position from monitoring after P/L is processed."""
        with self._lock:
            if conId in self._live_positions:
                logging.info(f"Removing conId {conId} from monitoring. Trade lifecycle complete.")
                self._live_positions.pop(conId)

    def _monitor_loop(self):
        """The main loop that runs in a separate thread to check positions."""
        logging.info("Position monitor thread started.")
        while self._is_monitoring:
            try:
                with self._lock:
                    active_conIds = list(self._live_positions.keys())

                if not active_conIds:
                    time.sleep(2)
                    continue

                for conId in active_conIds:
                    self._check_single_position(conId)

                time.sleep(2)

            except Exception as e:
                logging.error(f"Error in position monitor loop: {e}", exc_info=True)
                time.sleep(15)
        logging.info("Position monitor thread stopped.")

    def _check_single_position(self, conId):
        """Contains all exit logic for a single monitored position."""
        with self._lock:
            pos_data = self._live_positions.get(conId)
            # Skip if the fill isn't confirmed or an exit order is already sent
            if not pos_data or pos_data["status"] != "live":
                return

        # --- Get Live Data ---
        contract = pos_data["entry_trade"].contract
        ticker = self.ib_interface.get_live_ticker(contract)
        if not ticker or not ticker.last:
            logging.warning(f"Could not get live price for {contract.localSymbol}. Skipping check.")
            return

        current_price = ticker.last
        exit_strategy = pos_data["profile"]["exit_strategy"]

        # --- Update High-Water Mark ---
        pos_data["high_water_mark"] = max(pos_data["high_water_mark"], current_price)

        # --- Execute Exit Logic (in order of priority) ---

        # 1. Time-Based Exit
        timeout_minutes = exit_strategy["timeout_exit_minutes"]
        time_in_trade = (datetime.now(timezone.utc) - pos_data["entry_timestamp"]).total_seconds() / 60
        if time_in_trade > timeout_minutes:
            logging.info(f"TIMEOUT EXIT for {contract.localSymbol}. Trade exceeded {timeout_minutes} minutes.")
            self._execute_exit(conId, "Timeout")
            return

        # 2. Calculate current trailing stop price
        new_trailing_stop_price = self._calculate_trailing_stop(current_price, pos_data)
        pos_data["trailing_stop_price"] = new_trailing_stop_price

        # 3. Trailing Stop Exit
        if current_price <= pos_data["trailing_stop_price"]:
            logging.info(
                f"TRAILING STOP EXIT for {contract.localSymbol}. Price {current_price:.2f} hit stop at {pos_data['trailing_stop_price']:.2f}.")
            self._execute_exit(conId, "Trailing Stop")
            return

    def _calculate_trailing_stop(self, current_price, pos_data):
        """Calculates the new trailing stop price based on the configured strategy."""
        entry_price = pos_data["entry_price"]
        high_water_mark = pos_data["high_water_mark"]
        exit_strategy = pos_data["profile"]["exit_strategy"]

        # Check for Breakeven first
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        if not pos_data["breakeven_set"] and profit_percent >= exit_strategy["breakeven_trigger_percent"]:
            logging.info(
                f"BREAKEVEN TRIGGERED for {pos_data['entry_trade'].contract.localSymbol}. Locking stop at entry price.")
            pos_data["breakeven_set"] = True
            return entry_price

        if pos_data["breakeven_set"]:
            return max(pos_data["trailing_stop_price"], entry_price)

        # Main Trailing Logic
        trail_method = exit_strategy["trail_method"]
        if trail_method == "percentage":
            pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["percentage"] / 100)
            return high_water_mark - pullback_amount

        elif trail_method == "atr":
            atr_value = self.ib_interface.get_atr(pos_data["entry_trade"].contract)
            if atr_value:
                pullback_amount = atr_value * exit_strategy["trail_settings"]["atr_multiplier"]
                return high_water_mark - pullback_amount
            else:
                # Fallback to percentage if ATR fails
                pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["percentage"] / 100)
                return high_water_mark - pullback_amount

        return 0

    def _execute_exit(self, conId, reason):
        """Sends the market close order and marks the position as exiting."""
        with self._lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live":
                return  # Already exiting

            # Mark as exiting to prevent duplicate orders
            pos_data["status"] = "exit_sent"

        contract = pos_data["entry_trade"].contract
        quantity = pos_data["entry_trade"].order.totalQuantity

        logging.info(f"Executing market close for {contract.localSymbol} (Reason: {reason}).")
        self.ib_interface.close_position(contract, quantity)
        # We NO LONGER send a Telegram message here. The main fill handler does it.

    def start_monitoring(self):
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitoring_thread = Thread(target=self._monitor_loop, name="PositionMonitorThread", daemon=True)
            self._monitoring_thread.start()

    def stop_monitoring(self):
        logging.info("Stopping position monitor.")
        self._is_monitoring = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)

