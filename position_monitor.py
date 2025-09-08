# bot_engine/position_monitor.py
import logging
import time
from threading import Thread, Lock
from datetime import datetime, timezone

class PositionMonitor:
    """
    The "Watchtower." This is the definitive, stream-aware version. It no longer
    fetches its own market data. Instead, it gets instantaneous price updates
    from the central MarketDataManager, making it faster, more reliable, and
    truly professional.
    """

    def __init__(self, ib_interface, notifier, market_data_manager):
        self.ib_interface = ib_interface
        self.notifier = notifier
        self.market_data_manager = market_data_manager  # NEW: Gets the Master Watchmaker
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

            logging.info(f"Position for {entry_trade.contract.localSymbol} (conId: {conId}) is now pending, awaiting fill.")
            self._live_positions[conId] = {
                "entry_trade": entry_trade,
                "profile": profile,
                "sentiment_score": sentiment_score,
                "safety_net_settings": profile["safety_net"],
                "entry_price": None,
                "entry_timestamp": None,
                "high_water_mark": 0,
                "trailing_stop_price": 0,
                "breakeven_set": False,
                "status": "pending_fill",
                "previous_rsi": None
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
            pos_data["status"] = "live"
            logging.info(f"Monitoring fully activated for conId {conId} at entry price ${entry_price:.2f}")
            return pos_data

    def get_position_data(self, conId):
        with self._lock:
            return self._live_positions.get(conId)

    def remove_position(self, conId):
        with self._lock:
            if conId in self._live_positions:
                logging.info(f"Removing conId {conId} from monitoring. Trade lifecycle complete.")
                self._live_positions.pop(conId)

    def _monitor_loop(self):
        logging.info("Position monitor thread started.")
        while self._is_monitoring:
            try:
                with self._lock:
                    active_conIds = list(self._live_positions.keys())

                if not active_conIds:
                    time.sleep(1)
                    continue

                for conId in active_conIds:
                    self._check_single_position(conId)

                time.sleep(1)  # Can check very frequently now

            except Exception as e:
                logging.error(f"Error in position monitor loop: {e}", exc_info=True)
                time.sleep(15)
        logging.info("Position monitor thread stopped.")

    def _check_single_position(self, conId):
        with self._lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live":
                return

        # --- THIS IS THE CRITICAL FIX ---
        # Get the latest price from the central data manager's price board.
        # This is an instantaneous lookup, not a slow API call.
        ticker = self.market_data_manager.get_ticker(conId)
        if not ticker or not ticker.last:
            logging.debug(f"No live price data yet for conId {conId}. Waiting for stream to start.")
            return

        current_price = ticker.last
        pos_data["high_water_mark"] = max(pos_data["high_water_mark"], current_price)
        exit_strategy = pos_data["profile"]["exit_strategy"]

        # --- Execute Exit Logic (in order of priority) ---
        # 1. Time-Based Exit
        timeout_minutes = exit_strategy["timeout_exit_minutes"]
        time_in_trade = (datetime.now(timezone.utc) - pos_data["entry_timestamp"]).total_seconds() / 60
        if time_in_trade > timeout_minutes:
            reason = f"Timeout ({timeout_minutes}m)"
            logging.info(f"EXIT TRIGGER: {reason} for {ticker.contract.localSymbol}.")
            self._execute_exit(conId, reason)
            return
        
        # 2. Momentum-Based Early Exits
        # Note: This logic now needs to be adapted to get TA data from a different source
        # as the position_monitor no longer fetches historical bars. For now, it's disabled.
        # We can build a TA service later if desired.

        # 3. Trailing Stop
        new_trailing_stop_price = self._calculate_trailing_stop(current_price, pos_data)
        pos_data["trailing_stop_price"] = new_trailing_stop_price

        if current_price <= pos_data["trailing_stop_price"]:
            reason = f"Trailing Stop ({pos_data['trailing_stop_price']:.2f})"
            logging.info(f"EXIT TRIGGER: {reason} for {ticker.contract.localSymbol}. Price hit {current_price:.2f}.")
            self._execute_exit(conId, reason)
            return
        
        # Log the check for visibility, similar to your old script
        logging.info(f"[TRAIL CHECK] {ticker.contract.localSymbol} | Price: {current_price:.2f} | High: {pos_data['high_water_mark']:.2f} | Stop: {pos_data['trailing_stop_price']:.2f}")


    def _calculate_trailing_stop(self, current_price, pos_data):
        entry_price = pos_data["entry_price"]
        high_water_mark = pos_data["high_water_mark"]
        exit_strategy = pos_data["profile"]["exit_strategy"]

        # Check for Breakeven first
        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
        if not pos_data["breakeven_set"] and profit_percent >= exit_strategy["breakeven_trigger_percent"]:
            logging.info(f"BREAKEVEN TRIGGERED for {pos_data['entry_trade'].contract.localSymbol}.")
            pos_data["breakeven_set"] = True
            return entry_price

        if pos_data["breakeven_set"]:
            return max(pos_data.get("trailing_stop_price", 0), entry_price)

        # Main Trailing Logic (currently only percentage, as ATR requires historical data)
        trail_method = exit_strategy["trail_method"]
        if trail_method == "percentage":
            pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["percentage"] / 100)
            return high_water_mark - pullback_amount
        
        # NOTE: ATR logic is removed from here because this monitor no longer fetches
        # historical data bars. It would need a separate TA service to work.
        # For now, it defaults to a wide percentage stop if ATR is selected.
        else: # Default/fallback for ATR
            logging.debug("ATR trail method selected, but not implemented in this version. Using 50% fallback trail.")
            pullback_amount = high_water_mark * 0.50
            return high_water_mark - pullback_amount

    def _execute_exit(self, conId, reason):
        with self._lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live":
                return
            pos_data["status"] = "exit_sent"
        
        contract = pos_data["entry_trade"].contract
        quantity = pos_data["entry_trade"].order.totalQuantity
        
        logging.info(f"Executing market close for {contract.localSymbol} (Reason: {reason}).")
        self.ib_interface.close_position(contract, quantity)
        
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

