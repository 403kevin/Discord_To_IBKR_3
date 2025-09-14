import logging
from ib_insync import IB, Stock, Option, MarketOrder, StopLimitOrder, LimitOrder, Order, Trade, util
from services.config import Config
# Placeholder for a potential technical analysis module
# from services import technical_analysis as ta
import asyncio
import pandas as pd


class IBInterface:
    """
    Handles all interactions with the Interactive Brokers API via ib_insync.
    This is a specialist class, focused solely on broker communication.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.ib = IB()
        self.account_summary = {}
        self.active_positions = {}
        self.connection_successful = False

    async def connect(self):
        """Establishes connection to IBKR TWS or Gateway."""
        try:
            self.logger.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
            await self.ib.connectAsync(
                self.config.ibkr_host,
                self.config.ibkr_port,
                clientId=self.config.ibkr_client_id
            )
            self.logger.info("Connection to IBKR successful.")
            self.connection_successful = True
            self.ib.reqMarketDataType(3)  # 1=live, 2=frozen, 3=delayed, 4=delayed frozen
            self.ib.accountSummaryEvent += self.on_account_summary
            self.ib.updatePortfolioEvent += self.on_portfolio_update
            self.ib.reqAccountSummary( 'All', 'AccountType,AvailableFunds,ExcessLiquidity,NetLiquidity,TotalCashValue' )

        except Exception as e:
            self.logger.critical(f"Failed to connect to IBKR: {e}")
            self.connection_successful = False

    async def disconnect(self):
        """Disconnects from IBKR."""
        self.logger.info("Disconnecting from IBKR...")
        self.ib.disconnect()
        self.logger.info("Disconnected.")

    def on_account_summary(self, summary):
        """Event handler for account summary updates."""
        self.logger.info(f"Account summary update: {summary}")
        for tag in summary.tags.split(','):
             self.account_summary[tag] = summary.value
    
    def on_portfolio_update(self, item):
        """Event handler for portfolio updates."""
        self.logger.info(f"updatePortfolio: {item}")
        if item.position != 0:
            self.active_positions[item.contract.conId] = item
        else: # Position has been closed
            if item.contract.conId in self.active_positions:
                del self.active_positions[item.contract.conId]

    async def get_contract_details(self, symbol):
        """Fetches contract details for a given stock symbol."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            details = await self.ib.reqContractDetailsAsync(contract)
            if not details:
                self.logger.warning(f"No contract details found for {symbol}")
                return None
            return details[0].contract
        except Exception as e:
            self.logger.error(f"Error fetching contract details for {symbol}: {e}")
            return None

    async def get_option_chain(self, symbol, expiration_date, strike):
        """
        Fetches the specific option contract from the chain.
        Expiration date format: YYYYMMDD
        """
        try:
            # First, qualify the underlying stock contract
            underlying_contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(underlying_contract)

            # Fetch option chains
            chains = await self.ib.reqSecDefOptParamsAsync(
                underlyingSymbol=underlying_contract.symbol,
                futFopExchange='',
                underlyingSecType=underlying_contract.secType,
                underlyingConId=underlying_contract.conId
            )

            # Find the correct exchange from the chains
            chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not chain:
                self.logger.warning(f"No option chain found for {symbol} on SMART exchange.")
                return None
            
            # Filter for the specific expiration and strike
            # Note: This is simplified. A real implementation might need to find the closest match.
            # For now, we assume an exact match is requested.
            
            # Create and qualify the specific option contract
            option_contract = Option(
                symbol,
                expiration_date,
                strike,
                'C',  # Assuming CALL for now, will be dynamic
                'SMART',
                '100',
                'USD'
            )
            
            qualified_contracts = await self.ib.qualifyContractsAsync(option_contract)
            
            if qualified_contracts:
                return qualified_contracts[0]
            else:
                self.logger.warning(f"Could not qualify option contract for {symbol} {expiration_date} C{strike}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching option chain for {symbol}: {e}")
            return None

    def create_order(self, action, quantity, order_type="MKT", time_in_force="DAY", limit_price=None, stop_price=None):
        """Creates an IB order object."""
        if order_type == "MKT":
            return MarketOrder(action, quantity)
        elif order_type == "LMT":
            return LimitOrder(action, quantity, limit_price, tif=time_in_force)
        elif order_type == "STPLMT":
            return StopLimitOrder(action, quantity, stop_price, limit_price, tif=time_in_force)
        else:
            self.logger.error(f"Unsupported order type: {order_type}")
            return None

    def place_order(self, contract, order: Order) -> Trade:
        """Places an order and returns the Trade object."""
        # --- SURGICAL FIX: Check for existing open orders to prevent conflicts ---
        open_orders = self.ib.openOrders()
        for open_order in open_orders:
            if open_order.contract.conId == contract.conId:
                self.logger.warning(
                    f"Order conflict detected for {contract.localSymbol}. An open order already exists. "
                    f"Canceling new order request to prevent rejection."
                )
                # If an order already exists, we do not proceed.
                return None
        # --- END SURGICAL FIX ---
        try:
            self.logger.info(f"Placing order: {order.action} {order.totalQuantity} {contract.localSymbol} @ {order.orderType}")
            trade = self.ib.placeOrder(contract, order)
            return trade
        except Exception as e:
            self.logger.error(f"Error placing order for {contract.localSymbol}: {e}")
            return None
    
    async def get_historical_data(self, contract, duration='1 M', bar_size='1 day'):
        """Fetches historical market data."""
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            if bars:
                df = util.df(bars)
                return df
            else:
                self.logger.warning(f"No historical data returned for {contract.localSymbol}")
                return pd.DataFrame() # Return empty dataframe
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {contract.localSymbol}: {e}")
            return pd.DataFrame()
    
    def get_account_summary(self):
        """Returns the latest account summary data."""
        return self.account_summary

    def get_active_positions(self):
        """Returns the dictionary of active positions."""
        return self.active_positions

    async def get_ticker(self, contract):
        """Gets the live market data ticker for a contract."""
        try:
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2) # Allow time for ticker data to arrive
            return ticker
        except Exception as e:
            self.logger.error(f"Error getting ticker for {contract.localSymbol}: {e}")
            return None
