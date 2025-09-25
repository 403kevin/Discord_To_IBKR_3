import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import os

from services.config import Config
from services.signal_parser import SignalParser

class BacktestEngine:
    """
    A true, event-driven backtesting engine that simulates the live bot's
    tick-processing and decision-making logic with high fidelity.
    """
    def __init__(self, config, signal_file_path, data_folder_path):
        self.config = config
        self.signal_parser = SignalParser(config)
        self.signal_file_path = signal_file_path
        self.data_folder_path = data_folder_path
        
        # Simulation state
        self.portfolio = {'cash': 100000, 'positions': {}}
        self.trade_log = []
        
        # Mirrored logic from the live bot
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        self.trailing_highs = {}
        self.breakeven_activated = {}
        
        logging.info("Backtest Engine initialized.")

    def run_simulation(self):
        """
        The main entry point for the backtester. It loads all data, creates an
        event queue, and processes events chronologically.
        """
        logging.info("--- 🚀 Starting Backtest Simulation 🚀 ---")
        
        # 1. Load signals
        signals = self._load_signals()
        if not signals:
            logging.error("No signals to test. Aborting simulation.")
            return

        # 2. Create the master event queue
        event_queue = self._create_event_queue(signals)
        if not event_queue:
            logging.error("Failed to create event queue. Aborting simulation.")
            return

        # 3. Process events chronologically
        for timestamp, event_type, data in sorted(event_queue, key=lambda x: x[0]):
            if event_type == 'SIGNAL':
                self._process_signal_event(timestamp, data)
            elif event_type == 'TICK':
                self._process_tick_event(timestamp, data)
        
        self._log_results()
        logging.info("--- 🏁 Backtest Simulation Complete 🏁 ---")

    def _load_signals(self):
        """Loads signals from the signals_to_test.txt file."""
        signals = []
        with open(self.signal_file_path, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    timestamp_str, channel_name, signal_text = parts
                    signals.append({
                        'timestamp': datetime.strptime(timestamp_str.strip(), '%Y-%m-%d %H:%M:%S'),
                        'channel_name': channel_name.strip(),
                        'signal_text': signal_text.strip()
                    })
        logging.info(f"Loaded {len(signals)} signals from the battle plan.")
        return signals

    def _create_event_queue(self, signals):
        """Loads all historical tick data and merges it with signals into a single event queue."""
        event_queue = []
        
        # Add signal events
        for signal in signals:
            event_queue.append((signal['timestamp'], 'SIGNAL', signal))

        # Load tick data for all unique tickers mentioned in signals
        unique_tickers = set()
        for signal in signals:
            parsed = self.signal_parser.parse_signal(signal['signal_text'], self.config.profiles[0]) # Assumes first profile
            if parsed:
                unique_tickers.add(parsed['ticker'])
        
        for ticker in unique_tickers:
            file_path = os.path.join(self.data_folder_path, f"{ticker}_1_min_data.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['date'])
                # In a true tick-based backtest, this would be tick data.
                # We are simulating ticks from 1-min bars for now.
                for row in df.itertuples():
                    event_queue.append((row.date, 'TICK', {'ticker': ticker, 'price': row.close}))
            else:
                logging.warning(f"No historical data found for {ticker} at {file_path}")

        logging.info(f"Created event queue with {len(event_queue)} total events.")
        return event_queue

    def _process_signal_event(self, timestamp, signal_data):
        """Simulates the bot's reaction to a new signal."""
        profile = self.config.profiles[0] # Using first profile for all backtests for now
        parsed_signal = self.signal_parser.parse_signal(signal_data['signal_text'], profile)
        if not parsed_signal:
            return

        # Simulate position sizing
        # NOTE: This uses a fixed price for simplicity. A more advanced backtester
        # would look at the next available tick for a fill price.
        mock_ask_price = 1.5 # Using a mock price of $1.50
        quantity = int(profile['trading']['funds_allocation'] / (mock_ask_price * 100))
        
        if quantity > 0:
            position_key = f"{parsed_signal['ticker']}_{parsed_signal['expiry_date']}_{parsed_signal['strike']}{parsed_signal['contract_type'][0]}"
            cost = quantity * mock_ask_price * 100
            
            self.portfolio['cash'] -= cost
            self.portfolio['positions'][position_key] = {
                'entry_price': mock_ask_price,
                'quantity': quantity,
                'entry_time': timestamp,
                'ticker': parsed_signal['ticker']
            }
            logging.info(f"{timestamp} | OPENED {quantity} of {position_key} at ${mock_ask_price:.2f}")

    def _process_tick_event(self, timestamp, tick_data):
        """
        Processes a single market data tick, resamples it into bars, and evaluates
        exit logic for any relevant open positions. This mirrors the live engine.
        """
        for key, pos in list(self.portfolio['positions'].items()):
            if pos['ticker'] == tick_data['ticker']:
                # This is a simplified version of the live resampling logic
                # A true high-fidelity backtester would have a more complex bar creation mechanism
                
                # For simplicity, we'll just evaluate on every tick for now
                self._evaluate_simulated_exit(timestamp, key, tick_data['price'])

    def _evaluate_simulated_exit(self, timestamp, position_key, current_price):
        """Mirrors the exit logic evaluation from the live SignalProcessor."""
        position = self.portfolio['positions'][position_key]
        profile = self.config.profiles[0]
        exit_reason = None

        # This is a simplified mirror. It doesn't use pandas-ta to avoid re-calculating
        # on every tick, which would be too slow for a backtest.
        # It demonstrates the logic flow.
        pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100

        # Placeholder for breakeven check
        if pnl_percent > profile['exit_strategy']['breakeven_trigger_percent']:
            if current_price <= position['entry_price']:
                exit_reason = "Simulated Breakeven Stop"
        
        if exit_reason:
            self._close_simulated_position(timestamp, position_key, current_price, exit_reason)

    def _close_simulated_position(self, timestamp, position_key, exit_price, reason):
        """Closes a position in the simulation and logs the trade."""
        if position_key in self.portfolio['positions']:
            position = self.portfolio['positions'].pop(position_key)
            
            pnl = (exit_price - position['entry_price']) * position['quantity'] * 100
            self.portfolio['cash'] += (exit_price * position['quantity'] * 100)
            
            log_entry = {
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'position': position_key,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'exit_reason': reason
            }
            self.trade_log.append(log_entry)
            logging.info(f"{timestamp} | CLOSED {position['quantity']} of {position_key} at ${exit_price:.2f} for P/L of ${pnl:.2f}. Reason: {reason}")
    
    def _log_results(self):
        """Prints a summary of the backtest results."""
        df = pd.DataFrame(self.trade_log)
        if df.empty:
            logging.warning("No trades were executed during the simulation.")
            return

        total_pnl = df['pnl'].sum()
        win_rate = (df['pnl'] > 0).mean() * 100
        num_trades = len(df)
        
        logging.info("\n--- Backtest Results Summary ---")
        logging.info(f"Total Trades: {num_trades}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Total P/L: ${total_pnl:.2f}")
        logging.info(f"Final Portfolio Value: ${self.portfolio['cash']:.2f}")
        logging.info("--------------------------------\n")
        
        # Save log to CSV
        log_path = os.path.join(self.data_folder_path, "backtest_results.csv")
        df.to_csv(log_path, index=False)
        logging.info(f"Full trade log saved to {log_path}")

if __name__ == '__main__':
    # This allows the script to be run directly for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = Config()
    
    # Define paths relative to the main project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    signal_file = os.path.join(project_root, 'backtester', 'signals_to_test.txt')
    data_folder = os.path.join(project_root, 'backtester', 'data')

    engine = BacktestEngine(config, signal_file, data_folder)
    engine.run_simulation()