# bot_engine/trade_executor.py
import logging
import math
from datetime import datetime
import pytz

class TradeExecutor:
    """
    The "Trader." This is the definitive, event-driven version. Its sole
    responsibility is to place the entry order. The responsibility for attaching
    the safety net trail order has been promoted to the main fill handler
    to create a more robust, "Listen, Then Act" system.
    """
    def __init__(self, ib_interface, position_monitor, notifier):
        self.ib_interface = ib_interface
        self.position_monitor = position_monitor
        self.notifier = notifier
        self.config = ib_interface.config

    def _is_pre_market(self):
        """Checks if the current time is within the global pre-market window."""
        pm_config = self.config.pre_market_trading
        if not pm_config.get("enabled"): return False
        try:
            eastern = pytz.timezone('US/Eastern')
            now_et = datetime.now(eastern).time()
            start_time = datetime.strptime(pm_config["start_time"], "%H:%M").time()
            end_time = datetime.strptime(pm_config["end_time"], "%H:%M").time()
            return start_time <= now_et < end_time
        except Exception as e:
            logging.error(f"Could not parse pre-market times: {e}")
            return False

    def execute_trade(self, signal, profile):
        """
        The main entry point for executing a trade, now simplified to only
        handle the entry order.
        """
        try:
            contract = self.ib_interface.get_option_contract(
                symbol=signal["symbol"], strike=signal["strike"],
                right=signal["right"], expiry=signal["expiry"]
            )
            if not contract: raise ValueError("No security definition found for the request.")
            logging.info(f"Retrieved contract for trade execution: {contract.localSymbol}")

            trade_quantity = 0
            time_in_force = profile["trading"]["time_in_force"] 
            
            is_pm = self._is_pre_market()
            pm_symbols = self.config.pre_market_trading.get("symbols", [])

            if is_pm and signal["symbol"] in pm_symbols:
                # --- Pre-Market Path ---
                pm_config = self.config.pre_market_trading
                trade_quantity = pm_config["trade_quantity"]
                time_in_force = 'OPG'
                logging.info(f"PRE-MARKET MODE: Using fixed quantity of {trade_quantity} and TIF='OPG' for {signal['symbol']}.")
            else:
                # --- Regular Hours Path ---
                trading_config = profile["trading"]
                ticker = self.ib_interface.get_live_ticker(contract)
                live_price = ticker.ask if ticker and ticker.ask > 0 else ticker.last if ticker else 0

                if not live_price or live_price <= 0:
                    self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Could not fetch a valid live price.")
                    return

                min_price = trading_config["min_price_per_contract"]
                max_price = trading_config["max_price_per_contract"]

                if not (min_price <= live_price <= max_price):
                    self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Live price ${live_price:.2f} is outside the allowed range.")
                    return

                funds_to_allocate = trading_config["funds_allocation"]
                cost_per_contract = live_price * 100
                calculated_quantity = math.floor(funds_to_allocate / cost_per_contract)

                if calculated_quantity == 0:
                    self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Not enough allocated funds to purchase one contract.")
                    return
                
                trade_quantity = calculated_quantity
                logging.info(f"REGULAR HOURS: Calculated trade quantity: {trade_quantity} for {contract.localSymbol}.")

            # --- Execute the trade ---
            if trade_quantity > 0:
                # The Trader's job is now complete after this one call.
                entry_trade = self.ib_interface.place_trade(
                    contract, trade_quantity, time_in_force=time_in_force
                )
                logging.info(f"Entry order for {trade_quantity} contracts placed. OrderId: {entry_trade.order.orderId}, TIF: {time_in_force}")

                # The "dumb pause" and direct call to place the trail are gone.
                # The main.py fill handler is now responsible for the safety net.

                # We still notify the position monitor to expect a fill.
                self.position_monitor.add_position_to_monitor(
                    conId=contract.conId, entry_trade=entry_trade, profile=profile,
                    sentiment_score=signal.get("sentiment_score", 0.0)
                )
        except Exception as e:
            logging.error(f"Failed to execute trade for signal {signal}. Error: {e}", exc_info=True)
            error_message = (f"🚨 *Trade Execution Error* 🚨\n\n*Ticker:* `{signal.get('symbol', 'N/A')}`\n*Option:* `{signal.get('strike', 'N/A')}{signal.get('right', 'N/A')}`\n\n*Error:* `{e}`")
            self.notifier.send_message(error_message)

