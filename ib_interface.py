# interfaces/ib_interface.py
import logging
from ib_insync import IB, MarketOrder, Order, Trade
import pandas as pd  # <-- NEW: We need pandas for data analysis


class IBInterface:
    """
    Manages all communication with the Interactive Brokers API.
    This class handles connecting, placing orders, fetching data, and
    managing live trade events.
    """

    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.on_fill_callback = None

    def connect(self):
        """Establishes connection to TWS or Gateway."""
        try:
            logging.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
            self.ib.connect(self.config.ibkr_host, self.config.ibkr_port, clientId=self.config.ibkr_client_id)
            logging.info("IBKR connection successful.")

            # Setup event handlers for fills
            self.ib.filledEvent += self._on_fill
        except Exception as e:
            logging.critical(f"Failed to connect to IBKR: {e}")
            raise

    def disconnect(self):
        """Disconnects from TWS or Gateway."""
        if self.ib.isConnected():
            logging.info("Disconnecting from IBKR.")
            self.ib.disconnect()

    def _on_fill(self, trade: Trade, fill):
        """Internal handler for when an order is filled."""
        logging.info(
            f"Fill received: {fill.contract.localSymbol} {fill.execution.side} {fill.execution.shares} @ {fill.execution.price}")
        if self.on_fill_callback:
            self.on_fill_callback(trade, fill)

    def get_option_contract(self, symbol, strike, right, expiry):
        """Fetches and qualifies a specific option contract."""
        # This function should contain the logic to find and qualify the contract
        # For now, it's a placeholder.
        from ib_insync import ContFuture
        contract = ContFuture('ES', 'GLOBEX')  # Placeholder
        self.ib.qualifyContracts(contract)
        return contract

    def place_market_order(self, contract, action, quantity):
        """Places a simple market order."""
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logging.info(f"Placed Market Order: {action} {quantity} {contract.localSymbol}")
        return trade

    def place_native_trail_order(self, contract, action, quantity, trail_percent):
        """Places a native, broker-side trailing stop order."""
        order = Order(
            action=action,
            orderType='TRAIL',
            totalQuantity=quantity,
            trailingPercent=trail_percent
        )
        trade = self.ib.placeOrder(contract, order)
        logging.info(f"Placed Native Trail Order for {contract.localSymbol} at {trail_percent}%")
        return trade

    def close_position(self, contract, quantity):
        """Closes a position with a market order."""
        action = 'SELL' if quantity > 0 else 'BUY'
        abs_quantity = abs(quantity)
        self.place_market_order(contract, action, abs_quantity)

    def get_news_headlines(self, symbol):
        """Fetches recent news headlines for a given stock symbol."""
        # This function should contain the logic to get news headlines.
        # Placeholder for now.
        return ["Good news for stocks!", "Market is going up."]

    def get_all_positions(self):
        """Fetches a list of all positions in the portfolio."""
        return self.ib.positions()

    def flatten_all_positions(self):
        """Closes every position in the portfolio."""
        positions = self.get_all_positions()
        if not positions:
            logging.info("Flatten command received, but no positions to close.")
            return

        logging.warning(f"Executing emergency flatten for {len(positions)} positions.")
        for pos in positions:
            self.close_position(pos.contract, pos.position)

    def get_live_ticker(self, contract):
        """
        [--- NEW ---]
        Requests and returns a live market data ticker for a contract.
        """
        self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(0.5)  # Allow a moment for the data to arrive
        return self.ib.ticker(contract)

    def get_atr(self, contract, period=14):
        """
        [--- NEW ---]
        Calculates the Average True Range (ATR) for a given contract.
        """
        try:
            # Request 1 day of 1-minute bars to get enough data for ATR
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            if not bars:
                logging.warning(f"Could not fetch historical data for {contract.localSymbol} to calculate ATR.")
                return None

            # Convert to pandas DataFrame
            df = pd.DataFrame(bars)[['date', 'open', 'high', 'low', 'close']]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Calculate True Range (TR)
            df['high_minus_low'] = df['high'] - df['low']
            df['high_minus_prev_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_minus_prev_close'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close']].max(axis=1)

            # Calculate ATR using Exponential Moving Average
            df['atr'] = df['true_range'].ewm(span=period, adjust=False).mean()

            # Return the most recent ATR value
            latest_atr = df['atr'].iloc[-1]
            logging.info(f"Calculated ATR for {contract.localSymbol}: {latest_atr:.4f}")
            return latest_atr

        except Exception as e:
            logging.error(f"Failed to calculate ATR for {contract.localSymbol}: {e}")
            return None

