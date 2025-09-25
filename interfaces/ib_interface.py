import asyncio
import logging
from ib_insync import IB, Stock, Option, util, Ticker, Order, Trade
import pandas as pd

class IBInterface:
    """
    Manages the connection and all interactions with the IBKR API.
    Handles order placement, data streaming, and portfolio reconciliation.
    """
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.market_data_queue = asyncio.Queue()

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
                self.ib.filledOrderEvent += self._on_order_filled_callback # Forwards to signal processor
            else:
                logging.info("Already connected to IBKR.")
            await self.ib.reqMarketDataTypeAsync(3) # Set to delayed-frozen data
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
        """Allows the SignalProcessor to register its own fill handler."""
        self._on_order_filled_callback = callback
    
    # =================================================================
    # --- NEW CAPABILITY: Portfolio Reconciliation ---
    # =================================================================
    async def get_open_positions(self):
        """
        Asks the broker for a list of all currently held positions.
        This is the core of the state reconciliation logic.
        """
        if not self.is_connected():
            logging.error("Cannot get open positions, not connected to IBKR.")
            return []
        try:
            # reqPositionsAsync will return a live-updating list of positions
            positions = await self.ib.reqPositionsAsync()
            logging.info(f"Reconciliation: Found {len(positions)} positions held at broker.")
            return positions
        except Exception as e:
            logging.error(f"Error fetching open positions from IBKR: {e}", exc_info=True)
            return []

    # =================================================================
    # --- Core Trading Functions ---
    # =================================================================
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
            return trade
        except Exception as e:
            logging.error(f"Error placing order for {contract.localSymbol}: {e}", exc_info=True)
            return None

    # ... (rest of the file is unchanged)

    async def get_live_ticker(self, contract):
        """Requests a single, live market data snapshot for a contract."""
        if not self.is_connected(): return None
        try:
            ticker = await self.ib.reqMktDataAsync(contract, '', False, False)
            await asyncio.sleep(0.5) # Give a moment for data to arrive
            self.ib.cancelMktData(contract)
            
            # Validation Layer
            if ticker and ticker.last is not None and not pd.isna(ticker.last):
                return ticker
            else:
                logging.warning(f"Received invalid or empty ticker for {contract.localSymbol}")
                return None
        except Exception as e:
            logging.error(f"Error getting live ticker for {contract.localSymbol}: {e}", exc_info=True)
            return None

    # =================================================================
    # --- Real-Time Data Streaming ---
    # =================================================================

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
            # Validation Layer
            if ticker and ticker.contract and ticker.last is not None and not pd.isna(ticker.last):
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