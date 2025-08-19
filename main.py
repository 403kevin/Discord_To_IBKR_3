import logging
import requests
import time
from datetime import date
from typing import Dict, Union

import ib_insync
from ib_insync import Option, Stock, Order, Ticker
from ib_insync.util import isNan

import config


class DiscordScraper:
    BASE_URL = 'https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}'

    def __init__(self, auth_token: str):
        if not auth_token: raise ValueError("Discord auth token is required.")
        self.headers = {'authorization': auth_token,
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def poll_new_messages(self, channel_id: str, limit: int = 10) -> list:
        try:
            url = self.BASE_URL.format(channel_id=channel_id, limit=limit)
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Exception polling Discord channel {channel_id}: {e}")
            return []


class IBInterface:
    def __init__(self):
        self.ib = ib_insync.IB()
        self.connection_settings = (config.TWS_SETTINGS if config.USE_TWS else config.GATEWAY_SETTINGS)
        self.active_tickers: Dict[str, Ticker] = {}
        try:
            self.ib.connect(self.connection_settings['IP'], self.connection_settings['PORT'],
                            clientId=self.connection_settings['CLIENT_ID'])
            logging.info(f"[IB] Connected to {self.connection_settings['IP']}:{self.connection_settings['PORT']}")
        except Exception as e:
            logging.critical(f"[IB] Connection failed: {e}", exc_info=True)
            raise

    def create_contract_from_parsed_signal(self, parsed_signal: Dict) -> Union[Option, None]:
        try:
            symbol = parsed_signal.get("underlying")
            if not symbol: raise ValueError("Signal missing 'underlying' symbol.")
            exp_date = date(date.today().year, parsed_signal['exp_month'], parsed_signal['exp_day'])
            if exp_date < date.today(): exp_date = date(date.today().year + 1, parsed_signal['exp_month'],
                                                        parsed_signal['exp_day'])
            expiry_str = exp_date.strftime("%Y%m%d")
            right = "P" if parsed_signal["p_or_c"].upper() == "P" else "C"
            contract = Option(symbol, expiry_str, parsed_signal["strike"], right, "SMART", currency="USD")
            if symbol.upper() == "SPX": contract.tradingClass = "SPXW"
            self.ib.qualifyContracts(contract)
            logging.info(f"[CONTRACT] Qualified: {contract.localSymbol}")
            return contract
        except Exception as e:
            logging.error(f"[CONTRACT] Failed to create contract for {parsed_signal.get('underlying')}: {e}")
            return None

    def get_snapshot_price(self, contract: Option) -> float:
        if not contract or not contract.conId: return None
        ticker = self.ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        self.ib.sleep(2)
        price = ticker.last if ticker.last and not isNan(ticker.last) else ticker.marketPrice()
        if price and not isNan(price):
            logging.info(f"[PRICE] Fetched snapshot price for {contract.localSymbol}: {price}")
            return price
        else:
            logging.warning(f"[PRICE] Could not fetch valid snapshot price for {contract.localSymbol}.")
            return None

    def subscribe_to_streaming_data(self, contract: Option):
        if not contract or not contract.conId: return
        symbol = contract.localSymbol
        if symbol not in self.active_tickers:
            logging.info(f"[STREAM] Subscribing to market data for {symbol}")
            ticker = self.ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
            self.active_tickers[symbol] = ticker

    def get_price_from_stream(self, symbol: str) -> float:
        if symbol in self.active_tickers:
            ticker = self.active_tickers[symbol]
            price = ticker.last if ticker.last and not isNan(ticker.last) else ticker.marketPrice()
            return price if price and not isNan(price) else None
        return None

    def unsubscribe_from_streaming_data(self, symbol: str):
        if symbol in self.active_tickers:
            logging.info(f"[STREAM] Unsubscribing from market data for {symbol}")
            ticker = self.active_tickers.pop(symbol)
            self.ib.cancelMktData(ticker.contract)

    def submit_entry_order(self, order_details: Dict, profile: Dict) -> ib_insync.Trade:
        contract = self.create_contract_from_parsed_signal(order_details["parsed_symbol"])
        if not contract: return None
        order_type = profile.get("entry_order_type", "MKT")
        qty = order_details["qty"]

        if order_type == "PEG_MID":
            order = Order(action="BUY", orderType="PEG MID", totalQuantity=qty, account=config.ACCOUNT_NUMBER or "")
        elif order_type == "ADAPTIVE_URGENT":
            order = Order(action="BUY", orderType="MKT", algoStrategy="Adaptive",
                          algoParams=[ib_insync.TagValue("adaptivePriority", "Urgent")],
                          account=config.ACCOUNT_NUMBER or "")
        else:
            order = Order(action="BUY", orderType="MKT", totalQuantity=qty, account=config.ACCOUNT_NUMBER or "")

        logging.info(f"Submitting {order_type} BUY order for {qty}x {contract.localSymbol}")
        trade = self.ib.placeOrder(contract, order)

        if order_type != "MKT":
            timeout = profile.get("fill_timeout_seconds", 20)
            start_time = time.time()
            while trade.isActive() and (time.time() - start_time) < timeout:
                self.ib.sleep(1)

            if trade.isActive():
                logging.warning(f"[{contract.localSymbol}] Order did not fill within {timeout}s. Cancelling.")
                self.ib.cancelOrder(trade.order)
                return None

        return trade

    def submit_native_trail_order(self, order_details: Dict, trail_percent: float) -> ib_insync.Trade:
        contract = self.create_contract_from_parsed_signal(order_details["parsed_symbol"])
        if not contract: return None
        trail_order = Order(
            action="SELL", orderType="TRAIL", totalQuantity=order_details["qty"],
            trailingPercent=trail_percent, tif="GTC", account=config.ACCOUNT_NUMBER or ""
        )
        trade = self.ib.placeOrder(contract, trail_order)
        logging.info(f"Submitted NATIVE TRAIL for {order_details['qty']}x {contract.localSymbol} at {trail_percent}%")
        return trade

    def submit_bracket_order(self, order_details: Dict, base_price: float, exit_strategy: Dict):
        contract = self.create_contract_from_parsed_signal(order_details["parsed_symbol"])
        if not contract: return
        qty = order_details["qty"]

        take_profit_percent = exit_strategy.get("take_profit_percent", 20)
        stop_loss_percent = exit_strategy.get("stop_loss_percent", 20)

        take_profit_price = round(base_price * (1 + take_profit_percent / 100), 2)
        stop_loss_price = round(base_price * (1 - stop_loss_percent / 100), 2)

        parent = Order(action="BUY", orderType="LMT", totalQuantity=qty, lmtPrice=base_price, transmit=False,
                       account=config.ACCOUNT_NUMBER or "")
        takeProfit = Order(action="SELL", orderType="LMT", totalQuantity=qty, lmtPrice=take_profit_price,
                           parentId=parent.orderId, transmit=False, account=config.ACCOUNT_NUMBER or "")
        stopLoss = Order(action="SELL", orderType="STP", totalQuantity=qty, auxPrice=stop_loss_price,
                         parentId=parent.orderId, transmit=True, account=config.ACCOUNT_NUMBER or "")

        for ord in [parent, takeProfit, stopLoss]:
            self.ib.placeOrder(contract, ord)

        logging.info(
            f"Submitted BRACKET order for {qty}x {contract.localSymbol} | TP: {take_profit_price}, SL: {stop_loss_price}")

    def cancel_order(self, order_id: int):
        for o in self.ib.openOrders():
            if o.orderId == order_id:
                logging.info(f"[CANCEL] Cancelling safety net order ID: {order_id}")
                self.ib.cancelOrder(o)
                return

    def close_all_positions(self):
        positions = self.ib.positions(account=config.ACCOUNT_NUMBER or "")
        if not positions: logging.info("[EOD] No positions to close."); return
        logging.info(f"[EOD] Found {len(positions)} positions to close.")
        for p in positions:
            if p.position == 0: continue
            order = Order(action="SELL" if p.position > 0 else "BUY", orderType="MKT", totalQuantity=abs(p.position))
            self.ib.placeOrder(p.contract, order)
            logging.info(f"[EOD] Submitted closing order for {abs(p.position)}x {p.contract.localSymbol}")
            self.ib.sleep(0.5)

    def disconnect(self):
        logging.info("[IB] Disconnecting from Interactive Brokers.")
        self.ib.disconnect()
