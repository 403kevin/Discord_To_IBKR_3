import asyncio
import logging
from datetime import datetime
from ib_insync import Contract, Ticker, Order, Trade, OrderStatus
import pandas as pd
import os
import sys

# --- GPS FOR THE FORTRESS (PART 1) ---
# This ensures we can import from the project's root directories.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# --- Import Our Custom Tools ---
# FIX: Changed from "from utils import" to "from services.utils import"
from services.utils import get_data_filename  # The "Single Source of Truth"


class MockIBInterface:
    """
    A high-fidelity mock of the IBInterface for offline testing. It simulates
    broker actions and plays back historical tick data to mimic the live market.
    This is our "Flight Simulator."
    """
    def __init__(self, config):
        self.config = config
        self.ib = None
        self.market_data_queue = asyncio.Queue()
        self._order_filled_callback = None
        self._is_connected = False
        self._next_order_id = 1

    def is_connected(self):
        return self._is_connected

    async def connect(self):
        """Simulates a successful connection to the broker."""
        logging.info("MOCK BROKER: Connecting...")
        await asyncio.sleep(0.1)
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
        return Contract(
            symbol=symbol, secType='OPT', lastTradeDateOrContractMonth=expiry,
            strike=strike, right=right[0].upper(), exchange='SMART', currency='USD',
            localSymbol=f"{symbol} {expiry} {int(strike)}{right[0].upper()}"
        )

    async def get_live_ticker(self, contract):
        """Returns a fake ticker with a plausible ask price for sizing logic."""
        logging.info(f"MOCK BROKER: Fetching live ticker for {contract.localSymbol}")
        await asyncio.sleep(0.2)
        return Ticker(contract=contract, time=datetime.now(), ask=1.50, last=1.50)

    async def place_order(self, contract, order_type, quantity, action='BUY'):
        """Simulates placing an order and immediately confirms a fill."""
        logging.info(f"MOCK BROKER: Placing simulated {action} order for {quantity} of {contract.localSymbol}")
        
        order = Order(
            orderId=self._next_order_id,
            action=action,
            totalQuantity=quantity,
            orderType=order_type
        )
        self._next_order_id += 1
        
        fill_price = 1.50 
        order_status = OrderStatus(status='Filled', filled=quantity, remaining=0, avgFillPrice=fill_price)
        
        mock_trade = Trade(
            contract=contract,
            order=order,
            orderStatus=order_status,
            fills=[],
            log=[]
        )
        
        if self._order_filled_callback:
            asyncio.create_task(self._order_filled_callback(mock_trade))
            
        return order

    async def attach_native_trail(self, parent_order, trail_percent):
        """Mock implementation - logs but doesn't actually attach anything."""
        logging.info(f"MOCK BROKER: Would attach {trail_percent}% trail stop (simulated)")
        return None

    async def subscribe_to_market_data(self, contract):
        """
        Finds the historical data CSV for the contract and starts the
        playback engine. This is our "Time Machine."
        """
        logging.info(f"MOCK BROKER: Attempting to subscribe to market data for {contract.localSymbol}")
        
        # --- THE "SINGLE SOURCE OF TRUTH" FIX ---
        # Call the centralized utility function to get the filename.
        filename = get_data_filename(contract)
        
        # The harvester saves data in 'backtester/historical_data/'
        # We need to construct the path relative to the project root.
        filepath = os.path.join(project_root, 'backtester', 'historical_data', filename)

        if not os.path.exists(filepath):
            logging.error(f"MOCK BROKER: Flight recording not found at {filepath}. Cannot simulate market data.")
            return False
        
        asyncio.create_task(self._playback_market_data(filepath, contract))
        return True

    async def unsubscribe_from_market_data(self, contract):
        """Mock implementation - logs unsubscribe."""
        logging.info(f"MOCK BROKER: Unsubscribed from market data for {contract.localSymbol}")
        
    async def _playback_market_data(self, filepath, contract):
        """
        The "Playback Engine." Reads a CSV of historical data and
        feeds it to the market data queue in simulated real-time.
        """
        logging.info(f"MOCK BROKER: Starting market data playback for {contract.localSymbol} from {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['date'])
        
        for i, row in enumerate(df.itertuples(index=False)):
            mock_ticker = Ticker(
                contract=contract, time=row.date,
                last=row.close,
            )
            
            await self.market_data_queue.put(mock_ticker)
            
            if i + 1 < len(df):
                current_time = row.date
                next_time = df.iloc[i + 1]['date']
                sleep_duration = (next_time - current_time).total_seconds()
                await asyncio.sleep(max(0, sleep_duration))

        logging.info(f"MOCK BROKER: Market data playback finished for {contract.localSymbol}")

    async def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        """Returns initial historical data for the contract."""
        logging.info(f"MOCK BROKER: Fetching initial historical data for {contract.localSymbol}")
        
        filename = get_data_filename(contract)
        filepath = os.path.join(project_root, 'backtester', 'historical_data', filename)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['date']).set_index('date')
            return df.head(10)
        return pd.DataFrame()

    async def get_open_positions(self):
        """Mock implementation - returns empty list."""
        logging.info("MOCK BROKER: Getting open positions (returning empty list).")
        return []