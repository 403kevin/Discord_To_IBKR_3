# interfaces/ib_interface.py
import logging
import pandas as pd
import pandas_ta as ta  # <-- NEW: Import the technical analysis library
from ib_insync import IB, MarketOrder, Order, Contract, Stock

class IBInterface:
    """
    The Hands. This class manages all communication with the Interactive
    Brokers TWS or Gateway, including connections, order placement,
    and data fetching.
    """
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.on_fill_callback = None

    def connect(self):
        try:
            logging.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
            self.ib.connect(self.config.ibkr_host, self.config.ibkr_port, clientId=self.config.ibkr_client_id)
            # --- NEW: Set up the event handler for fills ---
            self.ib.execDetailsEvent += self._on_fill
            logging.info("Successfully connected to IBKR.")
        except Exception as e:
            logging.critical(f"Failed to connect to IBKR: {e}")
            raise ConnectionError("Could not connect to IBKR. Is TWS or Gateway running?") from e

    def disconnect(self):
        if self.ib.isConnected():
            logging.info("Disconnecting from IBKR.")
            self.ib.disconnect()

    def _on_fill(self, trade, fill):
        """Internal 'pager' that receives fill confirmations from the broker."""
        if self.on_fill_callback:
            self.on_fill_callback(trade, fill)

    def get_option_contract(self, symbol, strike, right, expiry):
        """Gets a qualified IBKR option contract object."""
        contract = Contract(secType='OPT', symbol=symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right=right, exchange='SMART', currency='USD')
        [qualified_contract] = self.ib.qualifyContracts(contract)
        return qualified_contract

    def place_market_order(self, contract, action, quantity):
        """Places a simple market order."""
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logging.info(f"Placed Market Order for {contract.localSymbol}: {action} {quantity}")
        return trade

    def place_native_trail_order(self, contract, action, quantity, trail_percent):
        """Places a broker-side native TRAIL order."""
        order = Order(
            action=action,
            orderType='TRAIL',
            totalQuantity=quantity,
            trailingPercent=trail_percent,
            transmit=True
        )
        trade = self.ib.placeOrder(contract, order)
        logging.info(f"Placed Native TRAIL Order for {contract.localSymbol} at {trail_percent}%")
        return trade

    def close_position(self, contract, quantity):
        """Closes a position with a market order."""
        action = 'SELL' if quantity > 0 else 'BUY'
        abs_quantity = abs(quantity)
        return self.place_market_order(contract, action, abs_quantity)

    def get_live_ticker(self, contract):
        """Gets a live market data ticker for a contract."""
        self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(0.5) # Allow time for data to arrive
        ticker = self.ib.ticker(contract)
        return ticker

    def get_news_headlines(self, symbol):
        """Fetches recent news headlines for a stock symbol."""
        stock_contract = Stock(symbol, 'SMART', 'USD')
        headlines = self.ib.reqNewsHeadlines(stock_contract.conId, '', 10)
        return headlines

    def get_all_positions(self):
        """Gets a list of all current portfolio positions."""
        return self.ib.positions()

    def flatten_all_positions(self):
        """Closes all open positions in the portfolio."""
        positions = self.get_all_positions()
        for pos in positions:
            if pos.position != 0:
                self.close_position(pos.contract, pos.position)

    # --- NEW: The Technical Analysis Toolkit ---
    def get_technical_indicators(self, contract, timeframe='1 min', duration='1 D'):
        """
        Fetches historical data and calculates key technical indicators.
        Returns a dictionary with the latest values for ATR, PSAR, and RSI.
        """
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            if not bars:
                logging.warning(f"Could not fetch historical bars for {contract.localSymbol}.")
                return None

            # Convert to a pandas DataFrame
            df = pd.DataFrame(bars)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            
            # Use pandas_ta to calculate indicators
            df.ta.atr(append=True)
            df.ta.psar(append=True)
            df.ta.rsi(append=True)

            # Get the latest values
            latest_indicators = {
                'atr': df['ATRr_14'].iloc[-1],
                'psar': df['PSARl_0.02_0.2'].iloc[-1], # Check long PSAR
                'rsi': df['RSI_14'].iloc[-1]
            }
            return latest_indicators

        except Exception as e:
            logging.error(f"Error calculating technical indicators for {contract.localSymbol}: {e}")
            return None

