"""
Bot_Engine/Interfaces/ib_interface.py

Author: 403-Forbidden
Purpose: Handles all communication with the Interactive Brokers (IBKR) TWS API.
         This module is responsible for connecting to TWS, placing/canceling orders,
         and managing real-time market data streams for open positions.
"""
import asyncio
import logging
from ib_insync import IB, Stock, Option, util, Ticker, Order

class IBInterface:
    """
    Manages the connection and all interactions with the IBKR API.
    """
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        # This queue is the mailbox for real-time market data.
        # The interface puts new price ticks here, and the signal processor reads them.
        self.market_data_queue = asyncio.Queue()

    def is_connected(self):
        """Checks if the connection to IBKR is active."""
        return self.ib.isConnected()

    async def connect(self):
        """Establishes a connection to the IBKR TWS/Gateway."""
        if self.is_connected():
            logging.info("IBKR connection is already active.")
            return

        try:
            logging.info(f"Connecting to IBKR at {self.config.ib_host}:{self.config.ib_port}...")
            await self.ib.connectAsync(
                host=self.config.ib_host,
                port=self.config.ib_port,
                clientId=self.config.ib_client_id
            )
            logging.info("Successfully connected to IBKR.")
            # Register the callback for incoming market data ticks.
            self.ib.pendingTickersEvent += self._on_pending_ticker
        except Exception as e:
            logging.error(f"Failed to connect to IBKR: {e}")
            raise

    async def disconnect(self):
        """Disconnects from the IBKR TWS/Gateway."""
        if self.is_connected():
            logging.info("Disconnecting from IBKR.")
            self.ib.disconnect()

    def _on_pending_ticker(self, tickers: list[Ticker]):
        """
        Callback function for the ib_insync event.
        This is called whenever a new price tick is received from IBKR.
        """
        for ticker in tickers:
            # Place the new ticker data into our mailbox for the processor to handle.
            self.market_data_queue.put_nowait(ticker)

    async def get_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD', **kwargs):
        """Qualifies a contract object."""
        if sec_type.upper() == 'OPTION':
            contract = Option(symbol, kwargs['expiry'], kwargs['strike'], kwargs['right'], exchange, currency)
        else: # Default to Stock
            contract = Stock(symbol, exchange, currency)
        
        qualified_contracts = await self.ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            logging.error(f"Could not qualify contract for {symbol}")
            return None
        return qualified_contracts[0]

    async def subscribe_to_market_data(self, contract):
        """Subscribes to the real-time market data stream for a given contract."""
        logging.info(f"Subscribing to real-time market data for {contract.localSymbol}...")
        self.ib.reqMktData(contract, '', False, False)

    async def unsubscribe_from_market_data(self, contract):
        """Unsubscribes from the real-time market data stream."""
        logging.info(f"Unsubscribing from market data for {contract.localSymbol}...")
        self.ib.cancelMktData(contract)

    async def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        """Fetches historical OHLC data for a contract."""
        logging.info(f"Fetching historical data for {contract.localSymbol}...")
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            return util.df(bars) # Return as a pandas DataFrame
        except Exception as e:
            logging.error(f"Error fetching historical data for {contract.localSymbol}: {e}")
            return None

    async def place_order(self, contract, order: Order):
        """Places an order with IBKR."""
        logging.info(f"Placing order for {contract.localSymbol}: {order.action} {order.totalQuantity} @ {order.orderType}")
        trade = self.ib.placeOrder(contract, order)
        return trade

    async def cancel_order(self, order: Order):
        """Cancels an existing order."""
        if not self.is_connected() or not order:
            return
        logging.info(f"Canceling order ID: {order.orderId}")
        self.ib.cancelOrder(order)