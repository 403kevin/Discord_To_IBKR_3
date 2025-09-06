# bot_engine/trade_executor.py
import logging

class TradeExecutor:
    """
    The "Trader." This specialist takes a fully validated signal and
    translates it into live orders with the broker. This is the corrected
    version with the proper class structure.
    """
    def __init__(self, ib_interface, position_monitor, notifier):
        self.ib_interface = ib_interface
        self.position_monitor = position_monitor
        self.notifier = notifier

    def execute_trade(self, signal, profile):
        """
        The main entry point for executing a trade. Follows the proven
        two-step entry logic.
        """
        try:
            # 1. Get the official contract from IBKR
            contract = self.ib_interface.get_option_contract(
                symbol=signal["symbol"],
                strike=signal["strike"],
                right=signal["right"],
                expiry=signal["expiry"]
            )
            logging.info(f"Executing trade for contract: {contract.localSymbol}")

            # 2. Place the primary Market Order to enter the position
            entry_trade = self.ib_interface.place_trade(
                contract,
                profile["trading"]["trade_quantity"],
                profile["entry_order_type"]
            )
            logging.info(f"Entry order placed for {contract.localSymbol}. OrderId: {entry_trade.order.orderId}")

            # 3. If enabled, place the wide native trail order as a safety net
            if profile["safety_net"]["enabled"]:
                self.ib_interface.place_native_trail_stop(
                    contract,
                    profile["trading"]["trade_quantity"],
                    profile["safety_net"]["native_trail_percent"]
                )
                logging.info(f"Native safety net trail order placed for {contract.localSymbol}.")

            # 4. Add the position to the monitor's watchlist
            self.position_monitor.add_position_to_monitor(
                conId=contract.conId,
                entry_trade=entry_trade,
                profile=profile,
                sentiment_score=signal.get("sentiment_score", 0.0)
            )

        except Exception as e:
            logging.error(f"Failed to execute trade for signal {signal}. Error: {e}", exc_info=True)
            self.notifier.send_message(f"🚨 *Trade Execution Error* 🚨\n\nSymbol: `{signal.get('symbol', 'N/A')}`\nError: `{e}`")

