import logging
import requests
from datetime import date
from typing import Dict, Union

import ib_insync
from ib_insync import Option, Stock, Order, Ticker

import config


# ==============================================================================
# DISCORD SCRAPER INTERFACE
# ==============================================================================

class DiscordScraper:
    """
    Handles the HTTP requests to scrape messages from a Discord channel.
    """
    BASE_URL = 'https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}'

    def __init__(self, auth_token: str):
        if not auth_token:
            raise ValueError("Discord authentication token is required.")

        self.headers = {
            'authorization': auth_token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def poll_new_messages(self, channel_id: str, limit: int = 10) -> list:
        """
        Polls a specific channel for the latest messages.
        """
        try:
            url = self.BASE_URL.format(channel_id=channel_id, limit=limit)
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Exception polling messages from Discord channel {channel_id}: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during Discord poll: {e}")
            return []


# ==============================================================================
# INTERACTIVE BROKERS INTERFACE
# ==============================================================================

class IBInterface:
    """
    Handles all communication with the Interactive Brokers TWS or Gateway.
    """

    def __init__(self):
        self.ib = ib_insync.IB()
        self.connection_settings = (
            config.TWS_SETTINGS if config.USE_TWS else config.GATEWAY_SETTINGS
        )
        try:
            self.ib.connect(
                self.connection_settings['IP'],
                self.connection_settings['PORT'],
                clientId=self.connection_settings['CLIENT_ID']
            )
            logging.info(f"[IB] Connected to {self.connection_settings['IP']}:{self.connection_settings['PORT']}")
        except Exception as e:
            logging.critical(f"[IB] Connection failed: {e}", exc_info=True)
            raise

    def create_contract_from_parsed_signal(self, parsed_signal: Dict) -> Union[Option, None]:
        """
        Creates and qualifies an IBKR Option contract from a parsed signal dictionary.
        """
        try:
            symbol = parsed_signal.get("underlying")
            if not symbol:
                raise ValueError("Parsed signal must contain an 'underlying' symbol.")

            exp_date = date(date.today().year, parsed_signal['exp_month'], parsed_signal['exp_day'])
            if exp_date < date.today():
                exp_date = date(date.today().year + 1, parsed_signal['exp_month'], parsed_signal['exp_day'])

            expiry_str = exp_date.strftime("%Y%m%d")
            right = "P" if parsed_signal["p_or_c"].upper() == "P" else "C"

            contract = Option(
                symbol,
                expiry_str,
                parsed_signal["strike"],
                right,
                "SMART",
                currency="USD"
            )

            if symbol.upper() == "SPX":
                contract.tradingClass = "SPXW"

            self.ib.qualifyContracts(contract)
            logging.info(f"[CONTRACT] Qualified: {contract.localSymbol}")
            return contract
        except Exception as e:
            logging.error(f"[CONTRACT] Failed to create or qualify contract for {parsed_signal.get('underlying')}: {e}")
            return None

    def get_realtime_price(self, contract: Option) -> float:
        if not contract or not contract.conId:
            logging.error(f"[PRICE] Cannot fetch price for an invalid or unqualified contract.")
            return None

        ticker = self.ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
        self.ib.sleep(2)
        price = ticker.last if Ticker.isne(ticker.last) else ticker.marketPrice()
        self.ib.cancelMktData(contract)

        if Ticker.isne(price):
            logging.info(f"[PRICE] Fetched price for {contract.localSymbol}: {price}")
            return price
        else:
            logging.warning(f"[PRICE] Could not fetch valid price for {contract.localSymbol}.")
            return None

    def submit_buy_market_order(self, order_details: Dict) -> ib_insync.Trade:
        contract = self.create_contract_from_parsed_signal(order_details["parsed_symbol"])
        if not contract: return None
        order = Order(
            action="BUY",
            orderType="MKT",
            totalQuantity=order_details["qty"],
            account=config.ACCOUNT_NUMBER or ""
        )
        trade = self.ib.placeOrder(contract, order)
        logging.info(f"Submitted market BUY order for {order_details['qty']}x {contract.localSymbol}")
        return trade

    def close_all_positions(self):
        positions = self.ib.positions(account=config.ACCOUNT_NUMBER or "")
        if not positions:
            logging.info("[EOD] No positions to close.")
            return
        logging.info(f"[EOD] Found {len(positions)} positions to close.")
        for position in positions:
            if position.position == 0:
                continue

            close_order = Order(
                action="SELL" if position.position > 0 else "BUY",
                orderType="MKT",
                totalQuantity=abs(position.position)
            )
            self.ib.placeOrder(position.contract, close_order)
            logging.info(f"[EOD] Submitted closing order for {abs(position.position)}x {position.contract.localSymbol}")
            self.ib.sleep(0.5)

    def disconnect(self):
        logging.info("[IB] Disconnecting from Interactive Brokers.")
        self.ib.disconnect()
