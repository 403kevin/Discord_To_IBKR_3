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
from backtester.technical_analyzer import TechnicalAnalyzer

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: The "Simulator Cockpit" (Mock Interfaces)
# ==============================================================================

class MockIBInterface(IBInterface):
    """
    The "Smart" mock broker. It now understands and records the reason for an exit.
    """

    def __init__(self, config, historical_data):
        self.config = config
        self.historical_data = historical_data
        self.paper_trades = []
        self.current_time = None
        self.ib = self  # Trick so self.ib_interface.ib works in the processor

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def qualifyContractsAsync(self, contract):
        return [contract]  # Assume valid

    def reqMktData(self, contract, *args, **kwargs):
        from ib_insync import Ticker
        # The symbol key in historical_data does not have spaces
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

    def cancelMktData(self, contract):
        pass

    async def place_order(self, contract, order, exit_reason=None):
        from ib_insync import Trade, OrderStatus
        symbol = contract.localSymbol.replace(' ', '_')
        if symbol not in self.historical_data:
            logger.error(f"Backtest Error: No historical data for {symbol}")
            return None

        df = self.historical_data[symbol]
        try:
            # Assume fill at the open price of the bar for the *next* minute
            next_minute = self.current_time + timedelta(minutes=1)
            price_row = df[df['date'] >= next_minute].iloc[0]
            fill_price = price_row['open']
        except IndexError:
            logger.error(f"Backtest Error: No future price data for {symbol} at {self.current_time}")
            return None

        trade_record = {
            "timestamp": self.current_time,
            "contract": contract.localSymbol,
            "action": order.action,
            "quantity": order.totalQuantity,
            "price": fill_price,
            "reason": exit_reason  # Record the reason for the exit
        }
        self.paper_trades.append(trade_record)
        logger.info(f"PAPER TRADE EXECUTED: {trade_record}")

        mock_trade = Trade(contract=contract, order=order,
                           orderStatus=OrderStatus(status='Filled', avgFillPrice=fill_price), fills=[], log=[])
        return mock_trade


class MockDiscordInterface(DiscordInterface):
    # This mock is simple because the engine feeds signals directly
    async def initialize(self): pass

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list: return []

    async def close(self): pass


# ==============================================================================
# SECTION 2: The "Time Machine"
# ==============================================================================

class BacktestEngine:
    """
    The "Smart Time Machine". It understands different channel profiles
    and applies the correct logic to each historical signal.
    """

    def __init__(self, config):
        self.config = config
        self.signal_log = []
        self.historical_data = {}
        self.profile_map = {p['channel_name']: p for p in self.config.profiles}

    def _load_signals(self, signal_file: str):
        """Loads the new, 3-part 'Smart' Battle Plan into memory."""
        logger.info(f"Loading smart battle plan from {signal_file}...")
        with open(signal_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue

                parts = line.split('|')
                if len(parts) != 3:
                    logger.warning(f"Skipping malformed line #{line_num}: Must have 3 parts separated by '|'")
                    continue

                timestamp_str, channel_name, signal_text = parts
                timestamp = pd.to_datetime(timestamp_str.strip())
                channel_name = channel_name.strip()

                if channel_name not in self.profile_map:
                    logger.warning(
                        f"Skipping signal on line #{line_num}: Channel '{channel_name}' not found in config profiles.")
                    continue

                self.signal_log.append({
                    "id": line_num, "content": signal_text.strip(),
                    "timestamp": timestamp, "author": "Backtest",
                    "channel_name": channel_name
                })
        self.signal_log.sort(key=lambda x: x['timestamp'])
        logger.info(f"Loaded {len(self.signal_log)} historical signals.")

    def _load_price_data(self, data_dir: str):
        """Loads all harvested CSV files into memory."""
        logger.info(f"Loading historical price data from {data_dir}...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                symbol_key = filename.replace('.csv', '')
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                self.historical_data[symbol_key] = df
        logger.info(f"Loaded {len(self.historical_data)} historical data files.")

    async def run(self):
        """The main simulation loop."""
        script_dir = os.path.dirname(__file__)
        self._load_signals(os.path.join(script_dir, 'signals_to_test.txt'))
        self._load_price_data(os.path.join(script_dir, 'historical_data'))

        if not self.signal_log:
            logger.critical("No valid signals found in battle plan. Aborting backtest.")
            return
        if not self.historical_data:
            logger.critical("No historical price data found. Run the data_harvester.py first. Aborting backtest.")
            return

        mock_ib = MockIBInterface(self.config, self.historical_data)
        mock_discord = MockDiscordInterface(self.config, [])  # No need to pass signals here
        flight_computer = TechnicalAnalyzer()

        class MockTelegram:
            async def initialize(self): pass

            async def send_message(self, text): logger.info(f"TELEGRAM (SIM): {text.replace('**', '')}")

            async def close(self): pass

        pilot = SignalProcessor(self.config, mock_ib, mock_discord, SentimentAnalyzer(self.config), MockTelegram(),
                                flight_computer)

        signals_by_time = {}
        for signal in self.signal_log:
            time_key = signal['timestamp'].floor('T')
            if time_key not in signals_by_time:
                signals_by_time[time_key] = []
            signals_by_time[time_key].append(signal)

        start_time = self.signal_log[0]['timestamp'].floor('T') - timedelta(minutes=1)
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
        self._report_results(mock_ib.paper_trades)

    def _report_results(self, trades):
        """
        The "Debriefing Room." Generates a detailed final report, including the
        reason for each exit, and saves it to a spreadsheet.
        """
        if not trades:
            logger.info("No trades were executed during the backtest.")
            return

        logger.info("\n--- BACKTEST RESULTS ---")
        df = pd.DataFrame(trades)

        results_path = os.path.join(os.path.dirname(__file__), 'backtest_results.csv')
        df.to_csv(results_path, index=False)
        logger.info(f"Full transaction log saved to spreadsheet: {results_path}")

        total_pnl = 0
        winning_trades = 0
        losing_trades = 0

        for contract_name, group in df.groupby('contract'):
            entries = group[group['action'] == 'BUY'].reset_index(drop=True)
            exits = group[group['action'] == 'SELL'].reset_index(drop=True)

            for i in range(min(len(entries), len(exits))):
                entry = entries.iloc[i]
                exit = exits.iloc[i]

                pnl = (exit['price'] - entry['price']) * entry['quantity'] * 100
                total_pnl += pnl

                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                logger.info(
                    f"Trade {contract_name}: "
                    f"Entry @ ${entry['price']:.2f}, "
                    f"Exit @ ${exit['price']:.2f}, "
                    f"Reason: {exit['reason']}, "
                    f"PnL: ${pnl:.2f}"
                )

        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        logger.info("\n--- PERFORMANCE SUMMARY ---")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total Realized PnL: ${total_pnl:.2f}")
        logger.info("--- END OF REPORT ---")


async def main():
    engine = BacktestEngine(Config())
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())

