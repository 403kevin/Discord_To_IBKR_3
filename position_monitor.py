# bot_engine/position_monitor.py
import logging
import time
from threading import Thread, Lock
from datetime import datetime, timezone

class PositionMonitor:
    """
    The "Watchtower." This is the definitive, fully transparent version.
    It now provides rich, detailed logging for every aspect of its
    decision-making process, from breakeven triggers to live indicator values.
    """
    def __init__(self, ib_interface, notifier, market_data_manager):
        self.ib_interface = ib_interface
        self.notifier = notifier
        self.market_data_manager = market_data_manager
        self._live_positions = {}
        self._lock = Lock()
        self._monitoring_thread = None
        self._is_monitoring = False

    def add_position_to_monitor(self, conId, entry_trade, profile, sentiment_score):
        with self._lock:
            if conId in self._live_positions: return
            logging.info(f"Position for {entry_trade.contract.localSymbol} (conId: {conId}) is now pending, awaiting fill.")
            self._live_positions[conId] = {
                "entry_trade": entry_trade, "profile": profile,
                "sentiment_score": sentiment_score, "safety_net_settings": profile["safety_net"],
                "entry_price": None, "entry_timestamp": None,
                "high_water_mark": 0, "trailing_stop_price": 0,
                "breakeven_set": False, "status": "pending_fill", "previous_rsi": None
            }

    def update_position_on_fill(self, conId, entry_price):
        with self._lock:
            if conId not in self._live_positions: return None
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
                time.sleep(5) # Check positions every 5 seconds for clearer logs
            except Exception as e:
                logging.error(f"Error in position monitor loop: {e}", exc_info=True)
                time.sleep(15)
        logging.info("Position monitor thread stopped.")
    
    def _check_single_position(self, conId):
        with self._lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live": return

        ticker = self.market_data_manager.get_ticker(conId)
        if not ticker or not ticker.last: return

        current_price = ticker.last
        pos_data["high_water_mark"] = max(pos_data["high_water_mark"], current_price)
        exit_strategy = pos_data["profile"]["exit_strategy"]
        contract_symbol = ticker.contract.localSymbol

        # --- NEW: Full Instrument Panel Logging ---
        indicators = self.ib_interface.get_technical_indicators(ticker.contract)
        
        # 1. Time-Based Exit
        timeout_minutes = exit_strategy["timeout_exit_minutes"]
        time_in_trade = (datetime.now(timezone.utc) - pos_data["entry_timestamp"]).total_seconds() / 60
        if time_in_trade > timeout_minutes:
            reason = f"Timeout ({timeout_minutes}m)"
            logging.info(f"EXIT TRIGGER: {reason} for {contract_symbol}.")
            self._execute_exit(conId, reason)
            return
        
        # 2. Momentum-Based Early Exits
        momentum_exits = exit_strategy.get("momentum_exits", {})
        if indicators:
            if momentum_exits.get("psar_enabled"):
                psar_val = indicators['psar']
                logging.info(f"[PSAR CHECK] {contract_symbol} | Price: {current_price:.2f} vs PSAR: {psar_val:.2f}")
                if psar_val > current_price:
                    logging.info(f"EXIT TRIGGER: PSAR Flip for {contract_symbol}.")
                    self._execute_exit(conId, "PSAR Flip")
                    return

            if momentum_exits.get("rsi_hook_enabled"):
                # Logic for RSI hook can be added here if needed
                pass

        # 3. Trailing Stop
        new_trailing_stop_price = self._calculate_trailing_stop(current_price, pos_data, indicators)
        pos_data["trailing_stop_price"] = new_trailing_stop_price

        if current_price <= pos_data["trailing_stop_price"]:
            reason = f"Trailing Stop ({pos_data['trailing_stop_price']:.2f})"
            logging.info(f"EXIT TRIGGER: {reason} for {contract_symbol}. Price hit {current_price:.2f}.")
            self._execute_exit(conId, reason)
            return
        
        # Final consolidated log line
        logging.info(f"[MONITOR] {contract_symbol} | Price: {current_price:.2f} | High: {pos_data['high_water_mark']:.2f} | Stop: {pos_data['trailing_stop_price']:.2f}")

    def _calculate_trailing_stop(self, current_price, pos_data, indicators):
        entry_price = pos_data["entry_price"]
        high_water_mark = pos_data["high_water_mark"]
        exit_strategy = pos_data["profile"]["exit_strategy"]
        contract_symbol = pos_data["entry_trade"].contract.localSymbol

        # Check for Breakeven first
        profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
        if not pos_data["breakeven_set"] and profit_percent >= exit_strategy["breakeven_trigger_percent"]:
            logging.info(f"BREAKEVEN TRIGGERED for {contract_symbol}. Stop moved to entry price ${entry_price:.2f}.")
            pos_data["breakeven_set"] = True
            return entry_price

        if pos_data["breakeven_set"]:
            return max(pos_data.get("trailing_stop_price", 0), entry_price)

        # Main Trailing Logic
        trail_method = exit_strategy["trail_method"]
        if trail_method == "percentage":
            pullback_amount = high_water_mark * (exit_strategy["trail_settings"]["percentage"] / 100)
            return high_water_mark - pullback_amount
        
        elif trail_method == "atr":
            if indicators and 'atr' in indicators:
                atr_val = indicators['atr']
                atr_mult = exit_strategy["trail_settings"]["atr_multiplier"]
                pullback_amount = atr_val * atr_mult
                logging.info(f"[ATR CALC] {contract_symbol} | ATR: {atr_val:.2f} * {atr_mult}x => Pullback: ${pullback_amount:.2f}")
                return high_water_mark - pullback_amount
            else:
                logging.warning(f"ATR calculation failed for {contract_symbol}. Using fallback.")
                pullback_amount = high_water_mark * 0.50 # Fallback
                return high_water_mark - pullback_amount
        
        return 0

    def _execute_exit(self, conId, reason):
        # ... (This function is the same)
        with self._lock:
            pos_data = self._live_positions.get(conId)
            if not pos_data or pos_data["status"] != "live": return
            pos_data["status"] = "exit_sent"
        
        contract = pos_data["entry_trade"].contract
        quantity = pos_data["entry_trade"].order.totalQuantity
        
        logging.info(f"Executing market close for {contract.localSymbol} (Reason: {reason}).")
        self.ib_interface.close_position(contract, quantity)
        
    def start_monitoring(self):
        # ... (This function is the same)
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitoring_thread = Thread(target=self._monitor_loop, name="PositionMonitorThread", daemon=True)
            self._monitoring_thread.start()

    def stop_monitoring(self):
        # ... (This function is the same)
        logging.info("Stopping position monitor.")
        self._is_monitoring = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)

