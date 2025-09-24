import asyncio
import logging
from ib_insync import IB, Stock, Option, util, Ticker, Order, MarketOrder

class IBInterface:
    """
    Manages the connection and all interactions with the IBKR API via ib_insync.
    This includes connecting, placing/canceling orders, and managing real-time
    market data streams for open positions.
    """
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.market_data_queue = asyncio.Queue()

    async def connect(self):
        """Establishes connection to TWS or Gateway."""
        try:
            if not self.ib.isConnected():
                await self.ib.connectAsync(
                    self.config.ibkr_host,
                    self.config.ibkr_port,
                    clientId=self.config.ibkr_client_id
                )
                logging.info("Successfully connected to IBKR.")
                self.ib.pendingTickersEvent += self._on_pending_tickers
        except Exception as e:
            logging.error("Failed to connect to IBKR: %s", e)
            raise ConnectionError("Could not connect to IBKR TWS/Gateway.")

    async def disconnect(self):
        """Disconnects from TWS or Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logging.info("Disconnected from IBKR.")

    def _on_pending_tickers(self, tickers):
        """
        Callback for real-time market data updates.
        Validates the ticker and puts it into the queue for the SignalProcessor.
        """
        for ticker in tickers:
            # --- SURGICAL UPGRADE: VALIDATION LAYER ---
            # Ensure the ticker has a valid last price before queuing
            if ticker and not util.isNan(ticker.last):
                logging.debug("Received tick: %s at %s", ticker.contract.symbol, ticker.last)
                self.market_data_queue.put_nowait(ticker)
            else:
                logging.warning("Received invalid or empty tick for %s. Discarding.", ticker.contract.symbol if ticker else "Unknown")


    async def get_contract_details(self, ticker, option_type, strike, expiry_str):
        """
        Fetches contract details from IBKR for a given signal.
        Returns a Contract object or None if not found.
        """
        try:
            contract = Option(ticker, expiry_str, strike, option_type[0].upper(), 'SMART')
            details = await self.ib.reqContractDetailsAsync(contract)
            if details:
                logging.info("Found contract details for %s", details[0].contract.localSymbol)
                return details[0].contract
            else:
                logging.warning("No contract details found for signal: %s %s %sC%s", ticker, expiry_str, strike, option_type)
                return None
        except Exception as e:
            logging.error("Error fetching contract details: %s", e)
            return None

    async def place_order(self, contract, action, quantity):
        """Places a market order for a given contract."""
        try:
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            logging.info("Placed order for %s %s of %s", action, quantity, contract.localSymbol)
            return trade
        except Exception as e:
            logging.error("Error placing order for %s: %s", contract.localSymbol, e)
            return None

    async def subscribe_to_market_data(self, contract):
        """Subscribes to real-time tick data for a contract."""
        try:
            self.ib.reqMktData(contract, '', False, False)
            logging.info("Subscribed to market data for %s.", contract.localSymbol)
            return True
        except Exception as e:
            logging.error("Failed to subscribe to market data for %s: %s", contract.localSymbol, e)
            return False

    async def unsubscribe_from_market_data(self, contract):
        """Unsubscribes from real-time tick data for a contract."""
        try:
            self.ib.cancelMktData(contract)
            logging.info("Unsubscribed from market data for %s.", contract.localSymbol)
        except Exception as e:
            logging.error("Failed to unsubscribe from market data for %s: %s", contract.localSymbol, e)

    async def get_historical_data(self, contract, duration, bar_size):
        """
        Fetches historical OHLC data for a contract.
        Returns a pandas DataFrame.
        """
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            if bars:
                df = util.df(bars)
                logging.info("Fetched %d bars of historical data for %s.", len(df), contract.localSymbol)
                return df
            else:
                logging.warning("No historical data returned for %s.", contract.localSymbol)
                return None
        except Exception as e:
            logging.error("Error fetching historical data for %s: %s", contract.localSymbol, e)
            return None
            
    async def get_live_ticker(self, contract):
        """
        Fetches a single snapshot of the current market data for a contract.
        Returns a Ticker object.
        """
        try:
            ticker = await self.ib.reqTickersAsync(contract)
            if ticker and ticker[0]:
                 # --- SURGICAL UPGRADE: VALIDATION LAYER ---
                 # Ensure the ticker has a valid price before returning
                if not util.isNan(ticker[0].last) and not util.isNan(ticker[0].ask):
                    return ticker[0]
                else:
                    logging.warning("Live ticker for %s has invalid price data.", contract.symbol)
                    return None
            return None
        except Exception as e:
            logging.error("Error fetching live ticker for %s: %s", contract.symbol, e)
            return None