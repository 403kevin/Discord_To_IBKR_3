import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# --- SURGICAL ADDITION: The "GPS" ---
# This tells the script how to find the other toolboxes from inside the 'backtester' folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Import our battle-hardened modules ---
from services.config import Config
from bot_engine.signal_processor import SignalProcessor
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: The "Simulator Cockpit" (Mock Interfaces)
# ==============================================================================

class MockIBInterface(IBInterface):
    """
    A simulated version of the IBInterface for backtesting.
    It doesn't connect to a real broker. Instead, it uses historical data.
    """
    def __init__(self, config, historical_data):
        # We don't call super().__init__() because we are replacing the connection logic
        self.config = config
        self.historical_data = historical_data
        self.paper_trades = []
        self.open_positions = {}
        self.current_time = None
        self.ib = self # A little trick so self.ib_interface.ib works in the processor

    async def connect(self):
        logger.info("Mock IBInterface 'connected'.")
        pass

    async def disconnect(self):
        logger.info("Mock IBInterface 'disconnected'.")
        pass

    # Mocking internal ib_insync methods that our processor uses
    async def qualifyContractsAsync(self, contract):
        return [contract] # Assume all contracts in a backtest are valid

    def reqMktData(self, contract, *args, **kwargs):
        # Return a mock ticker object with the historical price
        from ib_insync import Ticker
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            return Ticker(contract=contract, last=0) # Return a dummy ticker if no data
            
        df = self.historical_data[symbol]
        
        try:
            # Find the bar that corresponds to our current simulated time
            price_row = df[df['date'] <= self.current_time].iloc[-1]
            last_price = price_row['close']
        except IndexError:
            last_price = 0 # No data available for this time yet

        return Ticker(contract=contract, last=last_price, close=last_price)

    def cancelMktData(self, contract):
        pass # In a mock, this does nothing

    async def place_order(self, contract, order):
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            logger.error(f"Backtest Error: No historical data found for {symbol}")
            return None
        
        df = self.historical_data[symbol]
        
        try:
            current_price_row = df[df['date'] <= self.current_time].iloc[-1]
            fill_price = current_price_row['close']
        except IndexError:
            logger.error(f"Backtest Error: No price data available for {symbol} at time {self.current_time}")
            return None

        trade_record = {
            "timestamp": self.current_time,
            "contract": contract.localSymbol,
            "action": order.action,
            "quantity": order.totalQuantity,
            "price": fill_price,
            "order_type": order.orderType
        }
        self.paper_trades.append(trade_record)
        logger.info(f"PAPER TRADE EXECUTED: {trade_record}")
        
        from ib_insync import Trade, OrderStatus
        mock_trade = Trade(contract=contract, order=order, orderStatus=OrderStatus(status='Filled', avgFillPrice=fill_price), fills=[], log=[])
        return mock_trade


class MockDiscordInterface(DiscordInterface):
    """
    A simulated version of the DiscordInterface for backtesting.
    """
    def __init__(self, config, signal_log):
        self.config = config
        self.signal_log = signal_log
        self.current_time = None
        self.processed_ids = set() # To ensure we only serve a signal once

    async def initialize(self):
        logger.info("Mock DiscordInterface 'initialized'.")
        pass

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        triggered_signals = []
        for signal in self.signal_log:
            if signal['timestamp'] <= self.current_time and signal['id'] not in self.processed_ids:
                triggered_signals.append(signal)
                self.processed_ids.add(signal['id'])
        return triggered_signals

    async def close(self):
        pass


# ==============================================================================
# SECTION 2: The "Time Machine"
# ==============================================================================

class BacktestEngine:
    """
    The main orchestrator for running a historical simulation.
    """
    def __init__(self, config):
        self.config = config
        self.signal_log = []
        self.historical_data = {}

    def _load_signals(self, signal_file: str):
        """Loads the Battle Plan into memory."""
        logger.info(f"Loading battle plan from {signal_file}...")
        with open(signal_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split('|')
                if len(parts) != 2: continue
                
                timestamp_str, signal_text = parts
                timestamp = pd.to_datetime(timestamp_str.strip())

                self.signal_log.append({
                    "id": line_num, "content": signal_text.strip(),
                    "timestamp": timestamp, "author": "Backtest"
                })
        logger.info(f"Loaded {len(self.signal_log)} historical signals.")

    def _load_price_data(self, data_dir: str):
        """Loads all harvested CSV files into memory."""
        logger.info(f"Loading historical price data from {data_dir}...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                symbol = filename.replace('.csv', '')
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                self.historical_data[symbol] = df
        logger.info(f"Loaded {len(self.historical_data)} historical data files.")

    async def run(self):
        """The main simulation loop."""
        script_dir = os.path.dirname(__file__)
        self._load_signals(os.path.join(script_dir, 'signals_to_test.txt'))
        self._load_price_data(os.path.join(script_dir, 'historical_data'))

        if not self.signal_log or not self.historical_data:
            logger.critical("No signals or historical data found. Aborting backtest.")
            return

        mock_ib = MockIBInterface(self.config, self.historical_data)
        mock_discord = MockDiscordInterface(self.config, self.signal_log)
        class MockTelegram: # Simple mock to prevent crashes
            async def initialize(self): pass
            async def send_message(self, text): pass
            async def close(self): pass
        
        # Use the REAL brain with the FAKE interfaces
        pilot = SignalProcessor(self.config, mock_ib, mock_discord, SentimentAnalyzer(self.config), MockTelegram())

        start_time = self.signal_log[0]['timestamp'] - timedelta(minutes=1)
        end_time = self.signal_log[-1]['timestamp'] + timedelta(hours=4) # Run for 4 hours after last signal
        
        logger.info(f"Starting simulation from {start_time} to {end_time}...")
        
        current_time = start_time
        while current_time <= end_time:
            # Update the simulated world's time
            mock_discord.current_time = current_time
            mock_ib.current_time = current_time
            
            # Simulate the main.py loop
            dummy_profile = self.config.profiles[0] 
            messages = await mock_discord.get_latest_messages(dummy_profile['channel_id'])
            for message in messages:
                await pilot.process_signal(message, dummy_profile)
            
            await pilot.monitor_active_trades()

            current_time += timedelta(minutes=1)
            await asyncio.sleep(0.001)

        logger.info("Simulation complete.")
        # TODO: Add a final results report

async def main():
    """Main entry point for the backtest script."""
    engine = BacktestEngine(Config())
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())

