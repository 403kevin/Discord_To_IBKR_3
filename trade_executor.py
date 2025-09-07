# bot_engine/trade_executor.py
import logging
import math

class TradeExecutor:
    """
    The "Trader." This is the final, intelligent version that calculates
    position size based on a fixed capital allocation and performs pre-trade
    price checks, rather than using a static trade quantity.
    """
    def __init__(self, ib_interface, position_monitor, notifier):
        self.ib_interface = ib_interface
        self.position_monitor = position_monitor
        self.notifier = notifier

    def execute_trade(self, signal, profile):
        """
        The main entry point for executing a trade. It now includes pre-trade
        checks and dynamic position sizing.
        """
        try:
            trading_config = profile["trading"]

            # 1. Get the official contract from IBKR
            contract = self.ib_interface.get_option_contract(
                symbol=signal["symbol"],
                strike=signal["strike"],
                right=signal["right"],
                expiry=signal["expiry"]
            )
            logging.info(f"Retrieved contract for trade execution: {contract.localSymbol}")

            # 2. Get the live market price for the contract
            ticker = self.ib_interface.get_live_ticker(contract)
            # Use the 'ask' price for a more realistic entry cost on a BUY order
            live_price = ticker.ask if ticker and ticker.ask > 0 else ticker.last if ticker else 0

            if not live_price or live_price <= 0:
                logging.warning(f"TRADE REJECTED for {contract.localSymbol}. Could not fetch a valid live price.")
                self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Could not fetch a valid live price (market may be closed).")
                return

            # 3. Apply Pre-Trade Price Filters
            min_price = trading_config["min_price_per_contract"]
            max_price = trading_config["max_price_per_contract"]

            if not (min_price <= live_price <= max_price):
                logging.warning(f"TRADE REJECTED for {contract.localSymbol}. Live price ${live_price:.2f} is outside the configured range (${min_price:.2f} - ${max_price:.2f}).")
                self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Live price ${live_price:.2f} is outside the allowed range.")
                return

            # 4. Calculate Position Size based on Capital Allocation
            funds_to_allocate = trading_config["funds_allocation"]
            cost_per_contract = live_price * 100
            
            # Use math.floor to ensure we don't try to buy a fraction of a contract
            trade_quantity = math.floor(funds_to_allocate / cost_per_contract)

            if trade_quantity == 0:
                logging.warning(f"TRADE REJECTED for {contract.localSymbol}. Allocated funds of ${funds_to_allocate} is not enough to buy a single contract at ${cost_per_contract:.2f}.")
                self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Not enough allocated funds to purchase one contract.")
                return

            logging.info(f"Calculated trade quantity: {trade_quantity} contracts based on ${funds_to_allocate} allocation and live price of ${live_price:.2f}.")

            # 5. Place the primary Market Order with the calculated quantity
            entry_trade = self.ib_interface.place_trade(
                contract,
                trade_quantity,
                trading_config["entry_order_type"]
            )
            logging.info(f"Entry order for {trade_quantity} contracts placed for {contract.localSymbol}. OrderId: {entry_trade.order.orderId}")

            # 6. If enabled, place the native safety net trail order
            if profile["safety_net"]["enabled"]:
                self.ib_interface.place_native_trail_stop(
                    contract,
                    trade_quantity,
                    profile["safety_net"]["native_trail_percent"]
                )
                logging.info(f"Native safety net trail order placed for {contract.localSymbol}.")

            # 7. Add the position to the monitor's watchlist
            self.position_monitor.add_position_to_monitor(
                conId=contract.conId,
                entry_trade=entry_trade,
                profile=profile,
                sentiment_score=signal.get("sentiment_score", 0.0)
            )

        except Exception as e:
            logging.error(f"Failed to execute trade for signal {signal}. Error: {e}", exc_info=True)
            self.notifier.send_message(f"🚨 *Trade Execution Error* 🚨\n\nSymbol: `{signal.get('symbol', 'N/A')}`\nError: `{e}`")

