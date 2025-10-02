import asyncio
import logging
from ib_insync import IB, Stock, Option, util, Ticker, Order, Trade
import pandas as pd

class IBInterface:
    """
    Manages the connection and all interactions with the IBKR API.
    This version contains the fixed native trail implementation and order fill callback system.
    """
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.market_data_queue = asyncio.Queue()
        self._order_filled_callback = None

    def is_connected(self):
        return self.ib.isConnected()

    async def connect(self):
        """Establishes and manages the connection to IBKR TWS/Gateway."""
        try:
            if not self.is_connected():
                logging.info(f"Connecting to {self.config.ibkr_host}:{self.config.ibkr_port} with clientId {self.config.ibkr_client_id}...")
                await self.ib.connectAsync(
                    self.config.ibkr_host,
                    self.config.ibkr_port,
                    clientId=self.config.ibkr_client_id
                )
                logging.info("API connection ready")
                self.ib.pendingTickersEvent += self._on_pending_tickers
                self.ib.orderStatusEvent += self._on_order_status
            else:
                logging.info("Already connected to IBKR.")
            
            self.ib.reqMarketDataType(3)
            logging.info("Successfully connected to IBKR.")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to IBKR: {e}", exc_info=True)
            return False

    async def disconnect(self):
        """Disconnects from the IBKR API."""
        if self.is_connected():
            logging.info("Disconnecting from IBKR.")
            self.ib.disconnect()

    def set_order_filled_callback(self, callback):
        """
        Allows the SignalProcessor to register its own fill handler.
        The callback provided MUST be an async function.
        """
        if asyncio.iscoroutinefunction(callback):
            self._order_filled_callback = callback
        else:
            raise TypeError("The provided callback must be a coroutine (an async function).")

    def _on_order_status(self, trade: Trade):
        """
        Internal callback that listens for all order status updates. If an order
        is filled, it triggers the async callback in the SignalProcessor.
        """
        if trade.orderStatus.status == 'Filled':
            if self._order_filled_callback:
                asyncio.create_task(self._order_filled_callback(trade))

    async def get_open_positions(self):
        """Asks the broker for a list of all currently held positions."""
        if not self.is_connected():
            logging.error("Cannot get open positions, not connected to IBKR.")
            return []
        try:
            positions = await self.ib.reqPositionsAsync()
            logging.info(f"Reconciliation: Found {len(positions)} positions held at broker.")
            return positions
        except Exception as e:
            logging.error(f"Error fetching open positions from IBKR: {e}", exc_info=True)
            return []

    async def create_option_contract(self, symbol, expiry, strike, right):
        """Creates a qualified IBKR Option contract object."""
        try:
            contract = Option(symbol, expiry, strike, right, 'SMART', currency='USD')
            details = await self.ib.reqContractDetailsAsync(contract)
            if details:
                return details[0].contract
            else:
                logging.error(f"No contract details found for {symbol} {expiry} {strike}{right}")
                return None
        except Exception as e:
            logging.error(f"Error creating contract: {e}", exc_info=True)
            return None

    async def place_order(self, contract, order_type, quantity, action='BUY'):
        """Places a trade order."""
        try:
            order = Order(action=action, orderType=order_type, totalQuantity=quantity)
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed {action} order for {quantity} of {contract.localSymbol}")
            return order
        except Exception as e:
            logging.error(f"Error placing order for {contract.localSymbol}: {e}", exc_info=True)
            return None
            
    async def attach_native_trail(self, parent_order, trail_percent):
        """
        Attaches a broker-level trailing stop loss to a parent order.
        FIX: Implemented the actual logic instead of placeholder.
        """
        if not self.is_connected():
            logging.error("Cannot attach native trail, not connected to IBKR.")
            return None
        
        try:
            # Find the trade object for this parent order
            parent_trade = None
            for trade in self.ib.trades():
                if trade.order.orderId == parent_order.orderId:
                    parent_trade = trade
                    break
            
            if not parent_trade:
                logging.error(f"Could not find parent trade for order {parent_order.orderId}")
                return None
            
            contract = parent_trade.contract
            quantity = parent_order.totalQuantity
            
            # Create the trailing stop order
            trail_order = Order(
                action='SELL',
                orderType='TRAIL',
                totalQuantity=quantity,
                trailingPercent=trail_percent,
                tif='GTC'
            )
            
            trail_trade = self.ib.placeOrder(contract, trail_order)
            logging.info(f"Attached native trail stop ({trail_percent}%) for {quantity} of {contract.localSymbol}")
            return trail_trade
            
        except Exception as e:
            logging.error(f"Error attaching native trail: {e}", exc_info=True)
            return None

    async def get_live_ticker(self, contract):
        """Requests a single, live market data snapshot for a contract."""
        if not self.is_connected(): return None
        
        ticker = None
        try:
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2) 
            
            if ticker and ticker.last is not None and not pd.isna(ticker.last) and ticker.time:
                logging.debug(f"Received ticker for {contract.localSymbol}: {ticker}")
                return ticker
            else:
                logging.warning(f"Received invalid or empty ticker for {contract.localSymbol}. Data: {ticker}")
                return None
        except Exception as e:
            logging.error(f"Error getting live ticker for {contract.localSymbol}: {e}", exc_info=True)
            return None
        finally:
            if ticker:
                self.ib.cancelMktData(contract)

    async def subscribe_to_market_data(self, contract):
        """Subscribes to a real-time market data stream for a contract."""
        if not self.is_connected(): return False
        try:
            self.ib.reqMktData(contract, '', False, False)
            logging.info(f"Subscribed to market data for {contract.localSymbol}")
            return True
        except Exception as e:
            logging.error(f"Error subscribing to market data for {contract.localSymbol}: {e}", exc_info=True)
            return False

    async def unsubscribe_from_market_data(self, contract):
        """Unsubscribes from a real-time market data stream."""
        if not self.is_connected(): return
        try:
            self.ib.cancelMktData(contract)
            logging.info(f"Unsubscribed from market data for {contract.localSymbol}")
        except Exception as e:
            logging.error(f"Error unsubscribing from market data for {contract.localSymbol}: {e}", exc_info=True)

    def _on_pending_tickers(self, tickers):
        """Internal callback that listens for all real-time data ticks."""
        for ticker in tickers:
            if ticker and ticker.contract and ticker.last is not None and not pd.isna(ticker.last) and ticker.time:
                 self.market_data_queue.put_nowait(ticker)

    async def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        """Fetches historical bar data for a contract."""
        if not self.is_connected(): return None
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
                return util.df(bars).set_index('date')
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching historical data for {contract.localSymbol}: {e}", exc_info=True)
            return None
