import time
import logging
from typing import Dict

from ib_insync import Order

# We need to import the IBInterface class definition to avoid circular import issues
# but we'll get the actual connected instance from the main app.
from ib_interface import IBInterface
from trade_logger import log_trade

class TrailingStopManager:
    """
    Manages the lifecycle of active trades using dynamic, multi-stage exit logic.
    This runs in a separate thread and checks positions every few seconds.
    """
    def __init__(self, ib_interface: IBInterface):
        self.ib = ib_interface
        self.active_trails: Dict[str, dict] = {} # Key: contract.localSymbol

    def add_position(self, symbol: str, contract, entry_price: float, qty: int, rules: dict, trader_name: str, strategy_details: str):
        """
        Adds a new position to be managed by the dynamic trailing stop logic.
        The `rules` dictionary contains the specific exit strategy parameters for this trade.
        """
        # Only add the position if its strategy is 'dynamic_trail'
        if not rules or rules.get("type") != "dynamic_trail":
            return

        logging.info(f"[TRAIL ADD] Activating dynamic trail for {qty}x {symbol} for {trader_name}")
        self.active_trails[symbol] = {
            'contract': contract,
            'entry_price': entry_price,
            'qty': qty,
            'highest_price': entry_price,
            'breakeven_hit': False,
            'start_time': time.time(),
            'rules': rules,
            'trader_name': trader_name,
            'strategy_details': strategy_details
        }

    def check_trailing_stops(self):
        """
        This method is run in a loop to check the status of all active trails.
        It applies the multi-stage exit logic based on the rules for each trade.
        """
        if not self.active_trails:
            return

        # Iterate over a copy of the keys to safely modify the dictionary during the loop
        for symbol in list(self.active_trails.keys()):
            trail = self.active_trails[symbol]
            rules = trail['rules']

            # Fetch the current price for the contract
            current_price = self.ib.get_realtime_price(trail['contract'])
            if current_price is None:
                logging.warning(f"[TRAIL] No valid price for {symbol}, skipping check.")
                continue

            # --- 1. Hard Stop Loss Check (Highest Priority) ---
            hard_stop_percent = rules.get("hard_stop_loss_percent")
            if hard_stop_percent:
                hard_stop_price = trail['entry_price'] * (1 - hard_stop_percent / 100)
                if current_price <= hard_stop_price:
                    logging.info(f"[TRAIL EXIT] HARD STOP for {symbol}. Price {current_price:.2f} <= {hard_stop_price:.2f}")
                    self.execute_sell(symbol, trail, "hard_stop", current_price)
                    continue

            # --- 2. Breakeven Logic Check ---
            breakeven_percent = rules.get("breakeven_trigger_percent")
            if breakeven_percent and not trail['breakeven_hit']:
                breakeven_trigger_price = trail['entry_price'] * (1 + breakeven_percent / 100)
                if current_price >= breakeven_trigger_price:
                    trail['breakeven_hit'] = True
                    logging.info(f"[TRAIL] BREAKEVEN TRIGGERED for {symbol} at {breakeven_trigger_price:.2f}")

            # --- 3. Main Trailing Logic (only if breakeven has been hit) ---
            pullback_percent = rules.get("pullback_stop_percent")
            if pullback_percent and trail['breakeven_hit']:
                if current_price > trail['highest_price']:
                    trail['highest_price'] = current_price
                
                pullback_stop_price = trail['highest_price'] * (1 - pullback_percent / 100)
                if current_price < pullback_stop_price:
                    logging.info(f"[TRAIL EXIT] PULLBACK STOP for {symbol}. Price {current_price:.2f} < {pullback_stop_price:.2f}")
                    self.execute_sell(symbol, trail, "pullback_stop", current_price)
                    continue

            # --- 4. Timeout Check ---
            timeout_minutes = rules.get("timeout_exit_minutes")
            if timeout_minutes:
                elapsed_minutes = (time.time() - trail['start_time']) / 60
                if elapsed_minutes > timeout_minutes:
                    logging.info(f"[TRAIL EXIT] TIMEOUT for {symbol} after {elapsed_minutes:.1f} minutes.")
                    self.execute_sell(symbol, trail, "timeout", current_price)
                    continue

    def execute_sell(self, symbol: str, trail_info: dict, reason: str, exit_price: float):
        """
        Executes a market sell order and logs the trade with enhanced details.
        """
        logging.info(f"Executing sell for {symbol} for {trail_info['trader_name']} due to: {reason}")
        
        # Place a simple market sell order for the contract
        sell_order = Order(action="SELL", orderType="MKT", totalQuantity=trail_info['qty'])
        self.ib.ib.placeOrder(trail_info['contract'], sell_order)

        # Log the exit with the new enhanced details
        log_trade(
            symbol=symbol,
            qty=trail_info['qty'],
            price=exit_price,
            action="SELL",
            reason=reason,
            trader_name=trail_info['trader_name'],
            strategy_details=trail_info['strategy_details']
        )

        # Remove the position from active management
        del self.active_trails[symbol]

