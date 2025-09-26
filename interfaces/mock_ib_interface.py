import asyncio
import logging
from datetime import datetime
from ib_insync import Contract, Ticker, Order, Trade, OrderStatus, Fill, Execution
import pandas as pd
import os

class MockIBInterface:
    """
    A high-fidelity mock of the IBInterface for offline testing. It simulates
    broker actions and plays back historical tick data to mimic the live market.
    This is our "Flight Simulator."
    """
    def __init__(self, config):
        self.config = config
        self.ib = None # No real connection object
        self.market_data_queue = asyncio.Queue()
        self._order_filled_callback = None
        self._is_connected = False
        self._next_order_id = 1

    def is_connected(self):
        return self._is_connected

    async def connect(self):
        """Simulates a successful connection to the broker."""
        logging.info("MOCK BROKER: Connecting...")
        await asyncio.sleep(0.1) # Simulate network latency
        self._is_connected = True
        logging.info("MOCK BROKER: Successfully connected.")
        return True

    async def disconnect(self):
        """Simulates a successful disconnection."""
        logging.info("MOCK BROKER: Disconnecting...")
        self._is_connected = False
        await asyncio.sleep(0.1)

    def set_order_filled_callback(self, callback):
        """Stores the callback function from the SignalProcessor."""
        self._order_filled_callback = callback

    async def create_option_contract(self, symbol, expiry, strike, right):
        """Creates a mock, unqualified contract object for simulation."""
        logging.info(f"MOCK BROKER: Creating contract for {symbol} {expiry} {strike}{right}")
        # In a mock, we don't need to qualify it. We just create the object.
        return Contract(
            symbol=symbol, secType='OPT', lastTradeDateOrContractMonth=expiry,
            strike=strike, right=right[0].upper(), exchange='SMART', currency='USD',
            localSymbol=f"{symbol} {expiry} {strike}{right[0].upper()}"
        )

    async def get_live_ticker(self, contract):
        """Returns a fake ticker with a plausible ask price for sizing logic."""
        logging.info(f"MOCK BROKER: Fetching live ticker for {contract.localSymbol}")
        await asyncio.sleep(0.2) # Simulate latency
        # Return a static, plausible price.
        return Ticker(contract=contract, time=datetime.now(), ask=1.50, last=1.50)

    async def place_order(self, contract, order_type, quantity, action='BUY'):
        """Simulates placing an order and immediately confirms a fill."""
        logging.info(f"MOCK BROKER: Placing simulated {action} order for {quantity} of {contract.localSymbol}")
        
        # Create a fake Order object
        order = Order(
            orderId=self._next_order_id,
            clientId=self.config.ibkr_client_id,
            action=action,
            totalQuantity=quantity,
            orderType=order_type
        )
        self._next_order_id += 1
        
        # --- The Heart of the Simulation ---
        # Create a fake "Filled" status and a fake fill execution
        # We assume the fill happens at the price we used for sizing.
        fill_price = 1.50 
        order_status = OrderStatus(status='Filled', filled=quantity, remaining=0, avgFillPrice=fill_price)
        
        # Create a fake Trade object to send back to the SignalProcessor
        mock_trade = Trade(
            contract=contract,
            order=order,
            orderStatus=order_status,
            fills=[], # Fills list can be empty as we use orderStatus
            log=[]
        )
        
        # Immediately trigger the fill confirmation callback, just like the real broker would
        if self._order_filled_callback:
            asyncio.create_task(self._order_filled_callback(mock_trade))
            
        return order

    async def subscribe_to_market_data(self, contract):
        """
        Finds the historical data CSV for the contract and starts the
        playback engine. This is our "Time Machine."
        """
        logging.info(f"MOCK BROKER: Attempting to subscribe to market data for {contract.localSymbol}")
        
        # Construct the expected filename from the data harvester
        # Example: SPY_20250926_500C_5min_data.csv (this needs to match harvester output)
        filename = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{int(contract.strike)}{contract.right}_5sec_data.csv"
        filepath = os.path.join('backtester', 'historical_data', filename)

        if not os.path.exists(filepath):
            logging.error(f"MOCK BROKER: Flight recording not found at {filepath}. Cannot simulate market data.")
            return False
        
        # Start the playback engine as a background task
        asyncio.create_task(self._playback_market_data(filepath, contract))
        return True
        
    async def _playback_market_data(self, filepath, contract):
        """
        The "Playback Engine." Reads a CSV of historical tick/bar data and
        feeds it to the market data queue in simulated real-time.
        """
        logging.info(f"MOCK BROKER: Starting market data playback for {contract.localSymbol} from {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['date'])
        
        for i, row in enumerate(df.itertuples(index=False)):
            # Create a fake Ticker object for each row
            mock_ticker = Ticker(
                contract=contract,
                time=row.date,
                last=row.close,
                # You can add open, high, low if needed
            )
            
            # Put the fake tick into the queue for the SignalProcessor to find
            await self.market_data_queue.put(mock_ticker)
            
            # Sleep to simulate the real passage of time between data points
            if i + 1 < len(df):
                current_time = row.date
                next_time = df.iloc[i + 1]['date']
                sleep_duration = (next_time - current_time).total_seconds()
                await asyncio.sleep(max(0, sleep_duration))

        logging.info(f"MOCK BROKER: Market data playback finished for {contract.localSymbol}")

    # --- Other required methods (can be simple placeholders) ---

    async def unsubscribe_from_market_data(self, contract):
        logging.info(f"MOCK BROKER: Unsubscribing from market data for {contract.localSymbol}")
        # In a real mock, you might stop the playback task here
        pass

    async def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        # This can return a small slice of the same data used for playback
        logging.info(f"MOCK BROKER: Fetching initial historical data for {contract.localSymbol}")
        filename = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{int(contract.strike)}{contract.right}_5sec_data.csv"
        filepath = os.path.join('backtester', 'historical_data', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['date']).set_index('date')
            return df.head(10) # Return the first 10 bars as a sample
        return pd.DataFrame()

    async def get_open_positions(self):
        logging.info("MOCK BROKER: Getting open positions (returning empty list).")
        return []