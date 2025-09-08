# bot_engine/trade_executor.py
import logging
import math

class TradeExecutor:
    """
    The "Trader." This is the definitive, battle-hardened version. It calculates
    position size based on capital allocation, performs pre-trade price checks,
    and has a robust safety net to prevent any single trade failure from
    crashing the bot.
    """
    def __init__(self, ib_interface, position_monitor, notifier):
        self.ib_interface = ib_interface
        self.position_monitor = position_monitor
        self.notifier = notifier

    def execute_trade(self, signal, profile):
        """
        The main entry point for executing a trade, wrapped in a master
        error handler for maximum stability.
        """
        try:
            trading_config = profile["trading"]

            # 1. Get the official contract from IBKR using our new, smarter interface
            contract = self.ib_interface.get_option_contract(
                symbol=signal["symbol"],
                strike=signal["strike"],
                right=signal["right"],
                expiry=signal["expiry"]
            )
            
            # If the contract could not be found, raise an error to be caught below
            if not contract:
                raise ValueError(f"No security definition found for the request.")

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
            
            trade_quantity = math.floor(funds_to_allocate / cost_per_contract)

            if trade_quantity == 0:
                logging.warning(f"TRADE REJECTED for {contract.localSymbol}. Allocated funds of ${funds_to_allocate} is not enough to buy one contract at ${cost_per_contract:.2f}.")
                self.notifier.send_message(f"⚠️ *Trade Rejected* ⚠️\n\nSymbol: `{contract.localSymbol}`\nReason: Not enough allocated funds to purchase one contract.")
                return

            logging.info(f"Calculated trade quantity: {trade_quantity} contracts for {contract.localSymbol}.")

            # 5. Place the primary Market Order
            entry_trade = self.ib_interface.place_trade(
                contract, trade_quantity, trading_config["entry_order_type"]
            )
            logging.info(f"Entry order for {trade_quantity} contracts placed. OrderId: {entry_trade.order.orderId}")

            # 6. Place the native safety net trail order
            if profile["safety_net"]["enabled"]:
                self.ib_interface.place_native_trail_stop(
                    contract, trade_quantity, profile["safety_net"]["native_trail_percent"]
                )
                logging.info(f"Native safety net trail order placed.")

            # 7. Add the position to the monitor's watchlist
            self.position_monitor.add_position_to_monitor(
                conId=contract.conId,
                entry_trade=entry_trade,
                profile=profile,
                sentiment_score=signal.get("sentiment_score", 0.0)
            )

        except Exception as e:
            # This is the master safety net. It catches ANY error during execution.
            logging.error(f"Failed to execute trade for signal {signal}. Error: {e}", exc_info=True)
            # It then sends a detailed notification WITHOUT crashing the bot.
            error_message = (
                f"🚨 *Trade Execution Error* 🚨\n\n"
                f"*Ticker:* `{signal.get('symbol', 'N/A')}`\n"
                f"*Option:* `{signal.get('strike', 'N/A')}{signal.get('right', 'N/A')}`\n\n"
                f"*Error:* `{e}`"
            )
            self.notifier.send_message(error_message)

