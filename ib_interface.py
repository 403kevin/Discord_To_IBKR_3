# interfaces/ib_interface.py
import logging
from ib_insync import IB, Stock, Option, MarketOrder, Order, util
from datetime import datetime
import pandas as pd
import technical_analysis as ta

from services.config import Config

class IBInterface:
    """
    The bot's "Hands." This is the final, battle-hardened version that
    correctly handles timeouts and shutdown events.
    """
    def __init__(self, config: Config):
        self.config = config
        self.ib = IB()
        self.is_connected = False
        self.on_fill_callback = None

        # --- CORRECTED EVENT HANDLERS ---
        # We bind the methods to the event loop to ensure 'self' is passed correctly.
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self.ib.execDetailsEvent += self._on_fill

    def connect(self):
        try:
            logging.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
            # util.run ensures this works correctly across different threads/event loops
            util.run(self.ib.connectAsync(self.config.ibkr_host, self.config.ibkr_port, clientId=self.config.ibkr_client_id))
            self.ib.reqMarketDataType(3)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to IBKR: {e}")
            return False

    def get_option_contract(self, symbol, strike, right, expiry):
        """Gets a qualified option contract object from IBKR."""
        contract = Option(symbol, expiry, strike, right, 'SMART', tradingClass=symbol)
        [qualified_contract] = self.ib.qualifyContracts(contract)
        return qualified_contract

    def place_trade(self, contract, quantity, order_type="MKT"):
        """Places a simple market order."""
        order = MarketOrder("BUY", quantity)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_native_trail_stop(self, contract, quantity, trail_percent):
        """Places a native, broker-side trailing stop order."""
        order = Order()
        order.orderType = "TRAIL"
        order.action = "SELL"
        order.totalQuantity = quantity
        order.trailStopPrice = 0
        order.trailingPercent = float(trail_percent)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def get_news_headlines(self, symbol):
        """Fetches recent news headlines for a given stock symbol."""
        stock_contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock_contract)
        
        # --- THIS IS THE CRITICAL FIX FOR THE CRASH ---
        # We use util.run() to handle the asynchronous call and its timeout gracefully.
        # This is the library's official, correct way to prevent crashes.
        try:
            headlines = util.run(
                self.ib.reqHistoricalNewsAsync(stock_contract.conId, "BRFG", "", "", 100, []),
                timeout=10
            )
        except TimeoutError:
            logging.warning(f"Could not fetch news for {symbol}. Request timed out (market closed?).")
            headlines = [] # Return an empty list on timeout

        return [h.headline for h in headlines]

    def get_live_ticker(self, contract):
        """Requests and returns a live ticker for a contract."""
        self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(0.5)
        ticker = self.ib.ticker(contract)
        return ticker

    def get_technical_indicators(self, contract):
        """Calculates ATR, PSAR, and RSI."""
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime='', durationStr='1 D', barSizeSetting='1 min',
                whatToShow='TRADES', useRTH=True, formatDate=1
            )
            if not bars:
                logging.warning(f"Could not fetch historical data for {contract.localSymbol} to calculate TA.")
                return None
            df = pd.DataFrame(bars)
            df.set_index('date', inplace=True)
            atr = ta.get_atr(df)['ATR'][-1]
            psar = ta.get_psar(df)['PSAR'][-1]
            rsi = ta.get_rsi(df)['RSI'][-1]
            return {'atr': atr, 'psar': psar, 'rsi': rsi}
        except Exception as e:
            logging.error(f"Error calculating technical indicators for {contract.localSymbol}: {e}", exc_info=True)
            return None

    def close_position(self, contract, quantity):
        order = MarketOrder("SELL", quantity)
        self.ib.placeOrder(contract, order)

    def get_all_positions(self):
        return self.ib.positions()

    def flatten_all_positions(self):
        positions = self.get_all_positions()
        if not positions:
            logging.info("Flatten command received, but no open positions to close.")
            return
        logging.warning(f"EMERGENCY FLATTEN: Closing all {len(positions)} positions.")
        for pos in positions:
            action = "SELL" if pos.position > 0 else "BUY"
            quantity = abs(pos.position)
            order = MarketOrder(action, quantity)
            self.ib.placeOrder(pos.contract, order)
        logging.warning("All flatten orders have been sent.")

    def disconnect(self):
        if self.is_connected:
            logging.info("Disconnecting from IBKR.")
            self.ib.disconnect()

    # --- Private Methods (Corrected) ---
    def _on_connected(self):
        logging.info("IBKR connection successful.")
        self.is_connected = True

    def _on_disconnected(self):
        # This method now correctly handles the event.
        logging.warning("IBKR connection lost.")
        self.is_connected = False

    def _on_error(self, reqId, errorCode, errorString, contract):
        if errorCode not in [2104, 2106, 2158, 2108]:
             logging.error(f"IBKR Error. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")

    def _on_fill(self, trade, fill):
        if self.on_fill_callback:
            self.on_fill_callback(trade, fill)

