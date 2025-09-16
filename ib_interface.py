import logging
import asyncio
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, StopOrder, Order, Trade

logger = logging.getLogger(__name__)

class IBInterface:
    """
    A specialist module responsible for all interactions with the Interactive Brokers API.
    This class uses the ib_insync library in an asynchronous, non-blocking manner.
    It adheres to the "Single Operator" model.
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
            # Use connectAsync for non-blocking connection
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

    async def get_account_summary(self):
        """
        Fetches a summary of account values asynchronously.
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR. Cannot get account summary.")
            return {}
        
        # This is the corrected, modern way to call for the account summary.
        # It subscribes to updates, waits for the data, then unsubscribes.
        self.ib.reqAccountSummary()
        await asyncio.sleep(1) # Brief pause to allow data to arrive
        summary = {acc.tag: acc.value for acc in self.ib.accountSummary()}
        self.ib.cancelAccountSummary()
        return summary

    async def place_order(self, contract, order):
        """
        Places an order with IBKR. Includes a "guard clause" to prevent conflicts.
        """
        # --- SURGICAL FIX: Check for existing open orders to prevent conflicts ---
        open_orders = self.ib.openOrders()
        for open_order in open_orders:
            if open_order.contract.conId == contract.conId:
                logger.warning(
                    f"Order conflict detected for {contract.localSymbol}. An open order already exists. "
                    f"Canceling new order request to prevent rejection."
                )
                # If an order already exists, we do not proceed.
                return None
        # --- END SURGICAL FIX ---
        
        logger.info(f"Placing order: {order.action} {order.totalQuantity} {contract.localSymbol} @ {order.orderType}")
        trade = self.ib.placeOrder(contract, order)
        return trade

