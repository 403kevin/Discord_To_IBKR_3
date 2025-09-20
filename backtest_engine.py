import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# --- GPS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Imports ---
from services.config import Config
from bot_engine.signal_processor import SignalProcessor
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from backtester.technical_analyzer import TechnicalAnalyzer # <-- New "Flight Computer"

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Mock Interfaces ---
class MockIBInterface(IBInterface):
    """
    A simulated version of the IBInterface for backtesting.
    It uses historical data and tracks paper trades.
    """
    def __init__(self, config, historical_data):
        self.config = config
        self.historical_data = historical_data
        self.paper_trades = []
        self.current_time = None
        self.ib = self

    async def connect(self): pass
    async def disconnect(self): pass
    async def qualifyContractsAsync(self, contract): return [contract]

    def reqMktData(self, contract, *args, **kwargs):
        from ib_insync import Ticker
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            return Ticker(contract=contract, last=0, close=0)
            
        df = self.historical_data[symbol]
        try:
            price_row = df[df['date'] <= self.current_time].iloc[-1]
            last_price = price_row['close']
        except IndexError:
            last_price = 0
        return Ticker(contract=contract, last=last_price, close=last_price)

    def cancelMktData(self, contract): pass

    async def place_order(self, contract, order):
        from ib_insync import Trade, OrderStatus
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            logger.error(f"Backtest Error: No historical data found for {symbol}")
            return None
        
        df = self.historical_data[symbol]
        try:
            price_row = df[df['date'] <= self.current_time].iloc[-1]
            fill_price = price_row['open'] # Assume fill at the open of the next bar
        except IndexError:
            logger.error(f"Backtest Error: No price data for {symbol} at {self.current_time}")
            return None

        trade_record = { "timestamp": self.current_time, "contract": contract.localSymbol, "action": order.action,
                         "quantity": order.totalQuantity, "price": fill_price, "order_type": order.orderType }
        self.paper_trades.append(trade_record)
        logger.info(f"PAPER TRADE EXECUTED: {trade_record}")
        
        mock_trade = Trade(contract=contract, order=order, orderStatus=OrderStatus(status='Filled', avgFillPrice=fill_price), fills=[], log=[])
        return mock_trade


class MockDiscordInterface(DiscordInterface):
    # ... (This class is unchanged from the last correct version) ...
    pass


class BacktestEngine:
    """
    The "Smart Time Machine". This is the Level 2 version that uses the
    new Flight Computer to enable testing of dynamic exit logic.
    """
    def __init__(self, config):
        self.config = config
        self.signal_log = []
        self.historical_data = {}
        self.profile_map = {p['channel_name']: p for p in self.config.profiles}

    def _load_signals(self, signal_file: str):
        # ... (This function is unchanged from the last correct version) ...
        pass

    def _load_price_data(self, data_dir: str):
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
        flight_computer = TechnicalAnalyzer() # <-- Create the flight computer
        
        class MockTelegram:
            async def initialize(self): pass
            async def send_message(self, text): logger.info(f"TELEGRAM (SIM): {text.replace('*', '')}")
            async def close(self): pass
        
        # --- SURGICAL UPGRADE: Pass the Flight Computer to the Pilot ---
        pilot = SignalProcessor(self.config, mock_ib, mock_discord, SentimentAnalyzer(self.config), MockTelegram(), flight_computer)

        signals_by_time = {}
        for signal in self.signal_log:
            time_key = signal['timestamp'].floor('T')
            if time_key not in signals_by_time:
                signals_by_time[time_key] = []
            signals_by_time[time_key].append(signal)

        start_time = self.signal_log[0]['timestamp'].floor('T')
        end_time = self.signal_log[-1]['timestamp'] + timedelta(hours=8)
        
        logger.info(f"Starting simulation from {start_time} to {end_time}...")
        
        current_time = start_time
        while current_time <= end_time:
            mock_ib.current_time = current_time
            
            if current_time in signals_by_time:
                for signal in signals_by_time[current_time]:
                    profile = self.profile_map[signal['channel_name']]
                    logger.info(f"\n--- New Signal at {current_time} from {profile['channel_name']} ---")
                    await pilot.process_signal(signal, profile)
            
            await pilot.monitor_active_trades()

            current_time += timedelta(minutes=1)
            await asyncio.sleep(0.001)

        logger.info("Simulation complete.")

async def main():
    engine = BacktestEngine(Config())
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())