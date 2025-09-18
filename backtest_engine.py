import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Import our battle-hardened modules ---
from services.config import Config
from bot_engine.signal_processor import SignalProcessor
# We need the real interfaces to reference their structure
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface

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
        super().__init__(config)
        self.historical_data = historical_data
        self.paper_trades = []
        self.current_time = None

    async def connect(self):
        logger.info("Mock IBInterface 'connected'.")
        # In a mock, connect does nothing.
        pass

    async def disconnect(self):
        logger.info("Mock IBInterface 'disconnected'.")
        pass

    async def place_order(self, contract, order):
        # In a backtest, placing an order means we find the price at the current time.
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            logger.error(f"Backtest Error: No historical data found for {symbol}")
            return None
        
        df = self.historical_data[symbol]
        
        # Find the bar that corresponds to our current simulated time
        current_price_row = df[df['date'] <= self.current_time].iloc[-1]
        fill_price = current_price_row['close']

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
        
        # We must return a mock "Trade" object that the SignalProcessor expects
        from ib_insync import Trade, OrderStatus
        mock_trade = Trade(contract=contract, order=order, orderStatus=OrderStatus(status='Filled', avgFillPrice=fill_price), fills=[], log=[])
        return mock_trade


class MockDiscordInterface(DiscordInterface):
    """
    A simulated version of the DiscordInterface for backtesting.
    It serves signals from a historical list instead of polling a live channel.
    """
    def __init__(self, config, signal_log):
        super().__init__(config)
        self.signal_log = signal_log
        self.current_time = None

    async def initialize(self):
        logger.info("Mock DiscordInterface 'initialized'.")
        pass

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        # Find all signals that should have "appeared" at or before the current time
        triggered_signals = [
            s for s in self.signal_log 
            if s['timestamp'] <= self.current_time
        ]
        # Return them as if they were just fetched from Discord
        return triggered_signals


# ==============================================================================
# SECTION 2: The "Time Machine"
# ==============================================================================

class BacktestEngine:
    """
    The main orchestrator for running a historical simulation.
    It controls the flow of time and uses mock interfaces to feed data to the bot's brain.
    """
    def __init__(self, config):
        self.config = config
        self.signal_log = []
        self.historical_data = {}

    def _load_signals(self, signal_file: str):
        """Loads the Battle Plan into memory."""
        logger.info(f"Loading battle plan from {signal_file}...")
        with open(signal_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split('|')
                if len(parts) != 2: continue
                
                timestamp_str, signal_text = parts
                timestamp = datetime.fromisoformat(timestamp_str.strip())

                # We store the signal in the format our DiscordInterface produces
                self.signal_log.append({
                    "id": len(self.signal_log), # Unique ID for backtesting
                    "content": signal_text.strip(),
                    "timestamp": timestamp,
                    "author": "Backtest"
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
        # --- 1. Load the historical data ---
        self._load_signals('backtester/signals_to_test.txt')
        self._load_price_data('backtester/historical_data')

        if not self.signal_log or not self.historical_data:
            logger.critical("No signals or historical data found. Aborting backtest.")
            return

        # --- 2. Initialize the Cockpit ---
        mock_ib = MockIBInterface(self.config, self.historical_data)
        mock_discord = MockDiscordInterface(self.config, self.signal_log)
        # We need a mock for sentiment, for now it will just approve everything
        class MockSentiment:
            async def analyze_sentiment(self, ticker): return 1.0
        
        # This is the magic: we use the REAL brain with the FAKE interfaces
        pilot = SignalProcessor(self.config, mock_ib, mock_discord, MockSentiment())

        # --- 3. Run the Simulation ---
        start_time = self.signal_log[0]['timestamp'] - timedelta(minutes=1)
        end_time = self.signal_log[-1]['timestamp'] + timedelta(hours=2) # Run for 2 hours after last signal
        
        logger.info(f"Starting simulation from {start_time} to {end_time}...")
        
        current_time = start_time
        while current_time <= end_time:
            # Update the "current time" in our simulated world
            mock_discord.current_time = current_time
            mock_ib.current_time = current_time
            
            # --- Simulate the main.py loop ---
            # 1. Check for signals (the pilot doesn't know they are historical)
            # We use a dummy profile for now
            dummy_profile = self.config.profiles[0] 
            messages = await mock_discord.get_latest_messages(dummy_profile['channel_id'])
            for message in messages:
                await pilot.process_signal(message, dummy_profile)
            
            # 2. Monitor active trades
            await pilot.monitor_active_trades()

            # Tick the clock forward by one minute
            current_time += timedelta(minutes=1)
            # A tiny sleep to keep the loop from running too fast and locking up
            await asyncio.sleep(0.001)

        logger.info("Simulation complete.")
        self._report_results(mock_ib.paper_trades)

    def _report_results(self, trades):
        """Prints a final summary of the backtest performance."""
        if not trades:
            logger.info("No trades were executed during the backtest.")
            return
            
        logger.info("\n--- BACKTEST RESULTS ---")
        df = pd.DataFrame(trades)
        
        total_pnl = 0
        for i in range(0, len(df), 2):
            if i + 1 < len(df):
                entry = df.iloc[i]
                exit = df.iloc[i+1]
                pnl = (exit['price'] - entry['price']) * entry['quantity'] * 100 # Assuming 100 multiplier
                total_pnl += pnl
                logger.info(f"Trade {entry['contract']}: Entry @ ${entry['price']:.2f}, Exit @ ${exit['price']:.2f}, PnL: ${pnl:.2f}")

        logger.info(f"\nTotal Realized PnL: ${total_pnl:.2f}")
        logger.info("--- END OF REPORT ---")


async def main():
    """Main entry point for the backtest script."""
    engine = BacktestEngine(Config())
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())
