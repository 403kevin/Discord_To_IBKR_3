"""
IBKR Interface - Manages all interactions with Interactive Brokers.
FIXED: Now accepts config object like the rest of the codebase.
"""

import asyncio
import logging
from typing import Optional, List
from ib_insync import IB, Trade, Contract, Order, Option, util
import pandas as pd


class IBInterface:
    """Manages all communication with IBKR API."""
    
    def __init__(self, config):
        """
        Initialize with config object to match rest of codebase.
        This is how test_runner.py, data_harvester.py, and others expect it.
        """
        self.config = config
        self.ib = IB()
        self.host = config.ibkr_host
        self.port = config.ibkr_port
        self.client_id = config.ibkr_client_id
        self._order_filled_callback = None
        self.market_data_queue = asyncio.Queue()
        
        # NEW: Track orders being monitored for oversell protection
        self._monitored_close_orders = {}

    def is_connected(self):
        """Returns True if connected to IBKR."""
        return self.ib.isConnected()

    async def connect(self):
        """Connects to the IBKR API."""
        try:
            if self.is_connected():
                logging.info("Already connected to IBKR.")
                return True
            
            logging.info(f"Connecting to IBKR at {self.host}:{self.port} with client ID {self.client_id}...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            
            # Register event handlers
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_exec_details
            self.ib.pendingTickersEvent += self._on_pending_tickers
            
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

    def _on_exec_details(self, trade: Trade, fill):
        """
        NEW: Monitors partial fills on closing orders to prevent overselling.
        Cancels the remaining order once the position hits zero.
        """
        contract = trade.contract
        order = trade.order
        
        # Only monitor SELL orders we're tracking
        if order.action == 'SELL' and contract.conId in self._monitored_close_orders:
            monitor_data = self._monitored_close_orders[contract.conId]
            
            # Update filled quantity
            filled_so_far = trade.orderStatus.filled
            target_qty = monitor_data['target_quantity']
            
            logging.info(f"Close monitor: {filled_so_far}/{target_qty} filled for {contract.localSymbol}")
            
            # If we've filled the target, cancel the remaining order
            if filled_so_far >= target_qty:
                logging.warning(f"🛡️ OVERSELL PROTECTION: Target {target_qty} reached, cancelling remaining order for {contract.localSymbol}")
                asyncio.create_task(self._cancel_monitored_close_order(contract.conId, trade.order.orderId))

    async def _cancel_monitored_close_order(self, conId, order_id):
        """Helper to cancel a monitored close order and clean up tracking."""
        try:
            # Find and cancel the order
            for trade in self.ib.trades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logging.info(f"✅ Cancelled oversell-protected order {order_id}")
                    break
            
            # Clean up tracking
            self._monitored_close_orders.pop(conId, None)
            
        except Exception as e:
            logging.error(f"Error cancelling monitored close order: {e}", exc_info=True)

    def _on_order_status(self, trade: Trade):
        """
        Internal callback that listens for all order status updates.
        If an order is filled, it triggers the async callback in the SignalProcessor.
        """
        if trade.orderStatus.status == 'Filled':
            # Clean up monitor tracking if this was a monitored close
            if trade.contract.conId in self._monitored_close_orders:
                self._monitored_close_orders.pop(trade.contract.conId, None)
                logging.info(f"✅ Close order fully filled for {trade.contract.localSymbol}")
            
            if self._order_filled_callback:
                asyncio.create_task(self._order_filled_callback(trade))
        
        elif trade.orderStatus.status == 'Cancelled':
            # Clean up if cancelled
            if trade.contract.conId in self._monitored_close_orders:
                self._monitored_close_orders.pop(trade.contract.conId, None)

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

    async def cancel_all_orders_for_contract(self, contract):
        """
        Cancels ALL open orders for a specific contract.
        This prevents ghost trailing stops from remaining active after position is closed.
        """
        if not self.is_connected():
            logging.warning("Cannot cancel orders, not connected to IBKR.")
            return
        
        try:
            open_trades = self.ib.openTrades()
            cancelled_count = 0
            
            for trade in open_trades:
                if trade.contract.conId == contract.conId:
                    logging.info(f"Cancelling order {trade.order.orderId} for {contract.localSymbol}")
                    self.ib.cancelOrder(trade.order)
                    cancelled_count += 1
            
            if cancelled_count > 0:
                logging.info(f"Cancelled {cancelled_count} order(s) for {contract.localSymbol}")
                # Give IB a moment to process cancellations
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error cancelling orders for {contract.localSymbol}: {e}", exc_info=True)

    async def place_order(self, contract, order_type, quantity, action='BUY'):
        """
        Places a trade order.
        FIX: If this is a SELL order, cancel all existing orders first AND monitor fills to prevent overselling.
        """
        try:
            # FIX: Cancel all orders before closing position
            if action == 'SELL':
                await self.cancel_all_orders_for_contract(contract)
                
                # Set up monitoring to prevent overselling
                self._monitored_close_orders[contract.conId] = {
                    'target_quantity': quantity,
                    'contract': contract
                }
                logging.info(f"🛡️ OVERSELL PROTECTION: Monitoring close of {quantity} contracts for {contract.localSymbol}")
            
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
            # Find the parent trade from open trades
            parent_trade = None
            for trade in self.ib.trades():
                if trade.order == parent_order:
                    parent_trade = trade
                    break
            
            if not parent_trade:
                logging.error("Parent trade not found for trailing stop")
                return None
            
            contract = parent_trade.contract
            
            # Create trailing stop order
            trail_order = Order()
            trail_order.action = 'SELL'  # Assuming we're protecting a long position
            trail_order.orderType = 'TRAIL'
            trail_order.totalQuantity = parent_order.totalQuantity
            trail_order.trailingPercent = trail_percent
            trail_order.tif = 'GTC'
            trail_order.parentId = parent_order.orderId
            trail_order.transmit = True
            
            # Place the trailing stop
            trail_trade = self.ib.placeOrder(contract, trail_order)
            
            logging.info(f"Attached {trail_percent}% trailing stop to {contract.localSymbol}")
            return trail_order
            
        except Exception as e:
            logging.error(f"Error attaching trailing stop: {e}", exc_info=True)
            return None

    async def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        """
        Fetches historical data for a contract.
        
        Args:
            contract: IBKR contract object
            duration: Time period to fetch (e.g., '1 D', '2 D', '1 W')
            bar_size: Bar size ('5 secs', '1 min', '5 mins', etc.)
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if not self.is_connected():
            logging.error("Cannot fetch historical data, not connected to IBKR.")
            return None
        
        try:
            # Use reqHistoricalDataAsync for async context
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',  # Empty means "now"
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular Trading Hours only
                formatDate=1  # Return as datetime objects
            )
            
            if not bars:
                logging.warning(f"No historical data returned for {contract.localSymbol}")
                return None
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            logging.info(f"Retrieved {len(df)} bars of historical data for {contract.localSymbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {contract.localSymbol}: {e}", exc_info=True)
            return None

    async def get_live_ticker(self, contract):
        """
        Fetches live ticker data for position sizing.
        Returns a ticker object with bid/ask/last prices.
        """
        if not self.is_connected():
            logging.error("Cannot get live ticker, not connected to IBKR.")
            return None
        
        try:
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for ticker to populate (max 5 seconds)
            for _ in range(50):
                await asyncio.sleep(0.1)
                if ticker.ask and ticker.ask > 0:
                    logging.info(f"Got live ticker for {contract.localSymbol}: Ask=${ticker.ask}")
                    return ticker
            
            # If no ask price, try to use last price
            if ticker.last and ticker.last > 0:
                logging.warning(f"No ask price for {contract.localSymbol}, using last=${ticker.last}")
                ticker.ask = ticker.last  # Set ask to last for sizing
                return ticker
            
            logging.warning(f"No live data for {contract.localSymbol}")
            return None
            
        except Exception as e:
            logging.error(f"Error getting live ticker for {contract.localSymbol}: {e}")
            return None
        finally:
            # Cancel market data subscription
            if ticker:
                self.ib.cancelMktData(contract)

    async def stream_market_data_ticks(self, contract):
        """Streams live tick data for a contract and puts each tick into a queue."""
        if not self.is_connected():
            logging.error("Cannot stream market data, not connected to IBKR.")
            return
        
        ticker = self.ib.reqMktData(contract, '', False, False)
        logging.info(f"Subscribed to market data for {contract.localSymbol}")

        async def ticker_callback(ticker_obj):
            """Asynchronous inner function that responds to each new tick event."""
            if ticker_obj.time:
                await self.market_data_queue.put((ticker_obj.contract.conId, ticker_obj.last, ticker_obj.time))
        
        ticker.updateEvent += ticker_callback

    def _on_pending_tickers(self, tickers):
        """
        Internal callback that is automatically invoked by ib_insync whenever
        pending tickers have updated market data. This is for connection heartbeat purposes.
        """
        pass
