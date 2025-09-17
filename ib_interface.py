import logging
import asyncio
from ib_insync import IB, Stock, Option, MarketOrder

logger = logging.getLogger(__name__)

class IBInterface:
    """
    A specialist module responsible for all interactions with the Interactive Brokers API.
    This is the upgraded version with the EOD safety net feature.
    """

    def __init__(self, config):
        """
        Initializes the IB connection object.
        Args:
            config: The main configuration object.
        """
        self.config = config
        self.ib = IB()

    async def connect(self):
        """
        Connects to the IBKR TWS or Gateway.
        """
        try:
            logger.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
            await self.ib.connectAsync(
                self.config.ibkr_host,
                self.config.ibkr_port,
                clientId=self.config.ibkr_client_id
            )
            logger.info("Connection to IBKR successful.")
        except Exception as e:
            logger.critical(f"Failed to connect to IBKR: {e}")
            raise

    async def disconnect(self):
        """
        Disconnects from the IBKR TWS or Gateway.
        """
        if self.ib.isConnected():
            logger.info("Disconnecting from IBKR...")
            self.ib.disconnect()
            logger.info("Disconnected.")

    async def place_order(self, contract, order):
        """
        Places an order with IBKR. Includes a "guard clause" to prevent conflicts.
        """
        open_orders = self.ib.openOrders()
        for open_order in open_orders:
            if open_order.contract.conId == contract.conId:
                logger.warning(
                    f"Order conflict detected for {contract.localSymbol}. An open order already exists. "
                    f"Canceling new order request to prevent rejection."
                )
                return None
        
        logger.info(f"Placing order: {order.action} {order.totalQuantity} {contract.localSymbol} @ {order.orderType}")
        trade = self.ib.placeOrder(contract, order)
        return trade
        
    async def close_all_positions(self):
        """
        The EOD "Kill Switch". Fetches all open positions and closes them with market orders.
        This is a critical, restored safety feature.
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR. Cannot close positions.")
            return

        positions = self.ib.positions()
        if not positions:
            logger.info("EOD Check: No open positions to close.")
            return

        logger.warning(f"EOD CLOSE INITIATED. Closing {len(positions)} open position(s).")
        for position in positions:
            contract = position.contract
            quantity = position.position
            
            # Determine the correct closing action (SELL for long, BUY for short)
            action = 'SELL' if quantity > 0 else 'BUY'
            
            # We must use the absolute quantity for the order
            order_quantity = abs(quantity)

            order = MarketOrder(action, order_quantity)
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"EOD CLOSE: Submitted {action} order for {order_quantity} of {contract.localSymbol}.")
            await asyncio.sleep(0.1) # Small delay between closing orders
        
        logger.warning("EOD CLOSE COMPLETE.")

