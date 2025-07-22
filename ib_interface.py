# ib_interface.py

import logging
import ib_insync
from ib_insync import Option, Stock, Order


class IBInterface:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, clientId: int = 1, account_number: str = ""):
        """
        Initialize the IB connection.
        """
        self.ib = ib_insync.IB()
        self.account_number = account_number
        self.ib.connect(host, port, clientId)
        logging.info(f"[IB] Connected to {host}:{port} as clientId={clientId}")

    def create_contract(self, parsed_symbol) -> ib_insync.Contract:
        """
        Build and qualify a Stock or Option contract from parsed_symbol.
        """
        # Case 1: parsed_symbol is a simple stock ticker string
        if not hasattr(parsed_symbol, "underlying_symbol"):
            contract = Stock(parsed_symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            logging.info(f"[CONTRACT] Qualified Stock: {contract.localSymbol}")
            return contract

        # Case 2: parsed_symbol is an OptionSymbol-like object
        expiry_str = parsed_symbol.expiry.strftime("%Y%m%d")
        contract = Option(
            parsed_symbol.underlying_symbol,
            expiry_str,
            parsed_symbol.strike_price,
            parsed_symbol.call_or_put,
            "SMART",
            currency="USD"
        )

        # Special handling for SPX vs. SPXW
        if parsed_symbol.underlying_symbol.upper() == "SPX":
            contract.tradingClass = "SPXW"

        self.ib.qualifyContracts(contract)
        logging.info(f"[CONTRACT] Qualified Option: {contract.localSymbol}")
        return contract

    def get_realtime_price(
        self,
        contract: ib_insync.Contract,
        timeout: float = 3.0,
        use_snapshot: bool = False
    ) -> tuple[float, ib_insync.Contract]:
        """
        Fetch a single real-time price for a qualified IB Contract.
        """
        try:
            if not getattr(contract, "conId", None):
                self.ib.qualifyContracts(contract)

            if use_snapshot:
                ticker = self.ib.reqMktData(contract, "", False, True)
                self.ib.sleep(0.5)
                price = ticker.last if (ticker.last not in [None, 0.0]) else None
                try:
                    self.ib.cancelMktData(contract)
                except Exception:
                    pass
            else:
                ticker = self.ib.reqMktData(contract, "", False, False)
                self.ib.sleep(timeout)
                self.ib.cancelMktData(contract)
                price = ticker.last if ticker.last and (ticker.last not in [0.0]) else ticker.marketPrice()

            if price in [None, 0.0]:
                if hasattr(ticker, "bid") and hasattr(ticker, "ask") and ticker.bid and ticker.ask:
                    price = (ticker.bid + ticker.ask) / 2
                else:
                    price = None

            if price:
                logging.info(f"[PRICE FETCH] {contract.localSymbol}: {price}")
                return price, contract
            else:
                logging.warning(f"[PRICE FETCH FAIL] No valid price for {contract.localSymbol}")
                return -1.0, contract

        except Exception as exc:
            logging.error(f"[PRICE EXC] failed for {getattr(contract, 'localSymbol', contract)}: {exc}")
            return -1.0, contract

    def place_native_trail_stop(self, order: dict) -> ib_insync.Trade:
        """
        Place a native IB trailing-stop order.
        """
        contract = self.create_contract(order["parsed_symbol"])
        trailing_percent = order.get("trail_percent", 1.5) / 100.0
        ib_order = Order(
            orderType="TRAIL",
            totalQuantity=order["qty"],
            trailingPercent=trailing_percent,
            action="SELL",
            tif="GTC",
            account=self.account_number if self.account_number else None,
        )
        trade = self.ib.placeOrder(contract, ib_order)
        logging.info(
            f"[TRAIL STOP] Placed TRAIL order on {contract.localSymbol} "
            f"qty={order['qty']} trail%={trailing_percent*100}"
        )
        return trade

    def unsub_market_data(self, contract: ib_insync.Contract):
        """
        Cancel any ongoing market data subscription for the given contract.
        """
        try:
            self.ib.cancelMktData(contract)
            logging.info(f"[UNSUBSCRIBE] Cancelled market data for {contract.localSymbol}")
        except Exception as exc:
            logging.warning(f"[UNSUBSCRIBE FAIL] Could not cancel data for {contract}: {exc}")

    def disconnect(self):
        """
        Disconnect the IB session cleanly.
        """
        try:
            self.ib.disconnect()
            logging.info("[IB] Disconnected from Interactive Brokers")
        except Exception as exc:
            logging.error(f"[IB DISCONNECT ERROR] {exc}")
