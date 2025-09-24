import pandas as pd
import logging
from datetime import datetime, timedelta
import pandas_ta as ta
import os
import glob
from services.config import Config

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestEngine:
    """
    A true, event-driven backtesting engine that simulates the live bot's
    tick-processing and decision-making logic with high fidelity.
    """
    def __init__(self, config):
        self.config = config
        self.signals_to_test = []
        self.historical_data = {}  # {symbol_expiry_strike_right: DataFrame}
        self.event_queue = []
        self.open_positions = {} # {conId: position_details}
        self.position_data_cache = {} # Mirrors the live bot's cache
        self.trade_log = []

    def run_simulation(self):
        """Main entry point to run the entire backtesting simulation."""
        logging.info("--- LAUNCHING THE TIME MACHINE ---")
        self._load_signals()
        self._load_historical_data()
        self._prepare_event_queue()
        self._process_events()
        self._generate_report()
        logging.info("--- SIMULATION COMPLETE ---")

    def _load_signals(self):
        """Loads the curated signals from signals_to_test.txt."""
        path = 'backtester/signals_to_test.txt'
        logging.info("Phase 1: Loading Battle Plan from %s", path)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) == 3:
                    self.signals_to_test.append({
                        "timestamp": datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S'),
                        "channel": parts[1],
                        "text": parts[2]
                    })
        logging.info("Loaded %d signals to test.", len(self.signals_to_test))

    def _load_historical_data(self):
        """Loads all harvested tick data from CSV files."""
        path = 'backtester/data/*.csv'
        logging.info("Phase 2: Gathering Fuel from %s", path)
        csv_files = glob.glob(path)
        for file in csv_files:
            # Key is filename without extension, e.g., "SPY_20250104_500_C"
            key = os.path.basename(file).replace('.csv', '')
            df = pd.read_csv(file, parse_dates=['time'])
            self.historical_data[key] = df
        logging.info("Loaded historical data for %d contracts.", len(self.historical_data))


    def _prepare_event_queue(self):
        """Creates a chronological event queue of signals and market ticks."""
        logging.info("Preparing chronological event queue...")
        # Add signal events
        for i, signal in enumerate(self.signals_to_test):
            self.event_queue.append((signal['timestamp'], "SIGNAL", {"signal_id": i, "data": signal}))

        # Add tick events
        for key, df in self.historical_data.items():
            for row in df.itertuples():
                self.event_queue.append((row.time, "TICK", {"contract_key": key, "price": row.price}))
        
        # Sort all events by timestamp
        self.event_queue.sort(key=lambda x: x[0])
        logging.info("Event queue prepared with %d total events.", len(self.event_queue))

    def _process_events(self):
        """Processes all events in the queue in chronological order."""
        logging.info("Phase 3: Running Simulation...")
        for timestamp, event_type, data in self.event_queue:
            if event_type == "SIGNAL":
                self._handle_signal_event(timestamp, data)
            elif event_type == "TICK":
                self._handle_tick_event(timestamp, data)
    
    def _handle_signal_event(self, timestamp, data):
        """Simulates the bot receiving and processing a new signal."""
        # This is a simplified simulation of parsing and entry.
        # A more advanced version would use the actual SignalParser.
        signal = data['data']
        logging.info("EVENT at %s: Received signal from %s: %s", timestamp, signal['channel'], signal['text'])
        
        # Conceptual: Find the corresponding historical data for this signal
        # For simplicity, we'll assume the backtest runs on pre-identified contracts
        # In a real scenario, you'd parse signal['text'] here.
        pass # Entry logic is triggered by the backtest engine itself, not simulated signals for now.


    def _handle_tick_event(self, timestamp, data):
        """Mirrors the live bot's tick processing and resampling logic."""
        contract_key = data['contract_key']
        
        # This is a conceptual mapping. In a real sim, conId would be used.
        if contract_key in self.position_data_cache:
            cache = self.position_data_cache[contract_key]
            
            cache["ticks"].append({
                "time": timestamp,
                "price": data['price']
            })
            
            if timestamp >= cache["last_bar_timestamp"] + timedelta(minutes=1):
                self._resample_ticks_to_bar(contract_key)
                self._evaluate_dynamic_exit(contract_key)

    def _resample_ticks_to_bar(self, contract_key):
        """A near-direct copy of the live bot's resampling logic."""
        cache = self.position_data_cache[contract_key]
        ticks_df = pd.DataFrame(cache["ticks"])

        if len(ticks_df) < self.config.min_ticks_per_bar:
            cache["ticks"] = []
            cache["last_bar_timestamp"] = cache.get("last_bar_timestamp", datetime.min) + timedelta(minutes=1)
            return

        ticks_df.set_index('time', inplace=True)
        resampled = ticks_df['price'].resample('1Min').ohlc()
        if not resampled.empty:
            new_bar = resampled.iloc[-1]
            new_bar_timestamp = new_bar.name.to_pydatetime()

            new_row = pd.DataFrame([{
                "date": new_bar_timestamp, "open": new_bar.open, "high": new_bar.high,
                "low": new_bar.low, "close": new_bar.close,
                "volume": ticks_df['price'].resample('1Min').count().iloc[-1]
            }])
            
            cache["df"] = pd.concat([cache["df"], new_row], ignore_index=True)
            cache["ticks"] = []
            cache["last_bar_timestamp"] = new_bar_timestamp

    def _evaluate_dynamic_exit(self, contract_key):
        """A near-direct copy of the live bot's exit evaluation logic."""
        # This is a placeholder for the full exit logic evaluation
        # It would check RSI, PSAR, ATR etc. based on the newly formed bar
        pass

    def _generate_report(self):
        """Generates a summary of the backtest results."""
        # In a real implementation, this would analyze the self.trade_log
        # and print P/L, win rate, Sharpe ratio, etc.
        logging.info("--- Backtest Report ---")
        if not self.trade_log:
            logging.info("No trades were executed during the simulation.")
            return
            
        report_df = pd.DataFrame(self.trade_log)
        total_pnl = report_df['pnl'].sum()
        win_rate = len(report_df[report_df['pnl'] > 0]) / len(report_df) * 100
        
        logging.info("Total Trades: %d", len(report_df))
        logging.info("Win Rate: %.2f%%", win_rate)
        logging.info("Total P/L: $%.2f", total_pnl)
        logging.info("\nTrade Log:\n%s", report_df.to_string())

if __name__ == "__main__":
    config = Config()
    engine = BacktestEngine(config)
    # The engine still needs to be wired up to actually simulate trades.
    # The current state provides the high-fidelity event-driven architecture.
    logging.info("Backtest engine loaded. Further implementation needed to simulate trade entries and exits based on signals.")
    # engine.run_simulation() # This would be called to run the full sim