import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import os
import sys

# --- GPS FOR THE FORTRESS ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser
from utils import get_data_filename

class BacktestEngine:
    """
    A true, event-driven backtesting engine that simulates the live bot's
    tick-processing and decision-making logic with high fidelity.
    """
    def __init__(self, signal_file_path, data_folder_path):
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        self.signal_file_path = signal_file_path
        self.data_folder_path = data_folder_path
        
        # Simulation state
        self.portfolio = {'cash': 100000, 'positions': {}}
        self.trade_log = []
        
        # Mirrored logic from the live bot
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        self.trailing_highs_and_lows = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}
        
        logging.info("Backtest Engine initialized.")

    def run_simulation(self):
        """Main entry point: loads data, creates an event queue, and processes events."""
        logging.info("--- 🚀 Starting Backtest Simulation 🚀 ---")
        
        signals = self._load_signals()
        if not signals: return

        event_queue = self._create_event_queue(signals)
        if not event_queue: return

        for timestamp, event_type, data in sorted(event_queue, key=lambda x: x[0]):
            if event_type == 'SIGNAL':
                self._process_signal_event(timestamp, data)
            elif event_type == 'TICK':
                self._process_tick_event(timestamp, data)
        
        self._log_results()
        logging.info("--- 🏁 Backtest Simulation Complete 🏁 ---")

    def _load_signals(self):
        """Loads and parses signals from the signals_to_test.txt file."""
        signals = []
        with open(self.signal_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    timestamp_str, channel_name, signal_text = parts
                    parsed_signal = self.signal_parser.parse_signal(signal_text, self.config.profiles[0])
                    if parsed_signal:
                        parsed_signal['timestamp'] = datetime.strptime(timestamp_str.strip(), '%Y-%m-%d %H:%M:%S')
                        signals.append(parsed_signal)
        logging.info(f"Loaded {len(signals)} valid signals.")
        return signals

    def _create_event_queue(self, signals):
        """Loads all historical tick data and merges it with signals into a single event queue."""
        event_queue = []
        
        for signal in signals:
            event_queue.append((signal['timestamp'], 'SIGNAL', signal))

        unique_contracts = {
            f"{s['ticker']}_{s['expiry_date']}_{int(s['strike'])}{s['contract_type'][0]}" for s in signals
        }
        
        for contract_str in unique_contracts:
            # Reconstruct a mock contract to generate the filename
            parts = contract_str.split('_')
            mock_contract = type('MockContract', (), {'symbol': parts[0], 'lastTradeDateOrContractMonth': parts[1], 'strike': int(parts[2][:-1]), 'right': parts[2][-1]})
            filename = get_data_filename(mock_contract)
            file_path = os.path.join(self.data_folder_path, filename)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['date'])
                for row in df.itertuples():
                    event_queue.append((row.date, 'TICK', {'contract_str': contract_str, 'price': row.close}))
            else:
                logging.warning(f"No historical data found for {contract_str} at {file_path}")

        logging.info(f"Created event queue with {len(event_queue)} total events.")
        return event_queue

    def _process_signal_event(self, timestamp, signal_data):
        """Simulates the bot's reaction to a new signal."""
        profile = self.config.profiles[0]
        
        mock_ask_price = 1.5 # Using a mock price for sizing
        quantity = int(profile['trading']['funds_allocation'] / (mock_ask_price * 100))
        
        if quantity > 0:
            position_key = f"{signal_data['ticker']}_{signal_data['expiry_date']}_{int(signal_data['strike'])}{signal_data['contract_type'][0]}"
            cost = quantity * mock_ask_price * 100
            
            self.portfolio['cash'] -= cost
            self.portfolio['positions'][position_key] = {
                'entry_price': mock_ask_price, 'quantity': quantity,
                'entry_time': timestamp, 'is_call': signal_data['contract_type'] == 'CALL'
            }
            self.trailing_highs_and_lows[position_key] = {'high': mock_ask_price, 'low': mock_ask_price}
            self.breakeven_activated[position_key] = False
            logging.info(f"{timestamp} | OPENED {quantity} of {position_key} at ${mock_ask_price:.2f}")

    def _process_tick_event(self, timestamp, tick_data):
        """Processes a single market data tick for any relevant open positions."""
        position_key = tick_data['contract_str']
        if position_key in self.portfolio['positions']:
            self._resample_simulated_ticks_to_bar(timestamp, position_key, tick_data['price'])

    def _resample_simulated_ticks_to_bar(self, timestamp, position_key, price):
        """Mirrors the live bot's resampling logic."""
        if position_key not in self.tick_buffer:
            self.tick_buffer[position_key] = []
            self.last_bar_timestamp[position_key] = timestamp.replace(second=0, microsecond=0)

        self.tick_buffer[position_key].append(price)

        if timestamp >= self.last_bar_timestamp[position_key] + timedelta(minutes=1):
            profile = self.config.profiles[0]
            min_ticks = profile['exit_strategy']['min_ticks_per_bar']

            if len(self.tick_buffer[position_key]) >= min_ticks:
                prices = self.tick_buffer[position_key]
                new_bar = {'open': prices[0], 'high': max(prices), 'low': min(prices), 'close': prices[-1]}
                new_bar_df = pd.DataFrame([new_bar], index=[self.last_bar_timestamp[position_key]])
                
                if position_key in self.position_data_cache:
                    self.position_data_cache[position_key] = pd.concat([self.position_data_cache[position_key], new_bar_df])
                else:
                    self.position_data_cache[position_key] = new_bar_df
                
                self._evaluate_simulated_exit(timestamp, position_key)

            self.tick_buffer[position_key] = []
            self.last_bar_timestamp[position_key] = timestamp.replace(second=0, microsecond=0)

    def _evaluate_simulated_exit(self, timestamp, position_key):
        """
        THE AWAKENED PILOT: A near-perfect mirror of the live bot's dynamic exit logic.
        """
        position = self.portfolio['positions'][position_key]
        profile = self.config.profiles[0]
        data = self.position_data_cache.get(position_key)
        if data is None or data.empty or len(data) < 2: return

        is_call = position['is_call']
        exit_reason = None
        current_price = data['close'].iloc[-1]
        
        high_low = self.trailing_highs_and_lows[position_key]
        high_low['high'] = max(high_low['high'], current_price)
        high_low['low'] = min(high_low['low'], current_price)

        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_reason: break

            if exit_type == "breakeven" and not self.breakeven_activated.get(position_key):
                pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
                trigger_pct = profile['exit_strategy']['breakeven_trigger_percent']
                if (is_call and pnl_percent >= trigger_pct) or (not is_call and pnl_percent <= -trigger_pct):
                    self.breakeven_activated[position_key] = True
            
            if self.breakeven_activated.get(position_key):
                if (is_call and current_price <= position['entry_price']) or (not is_call and current_price >= position['entry_price']):
                    exit_reason = "Breakeven Stop Hit"

            if exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                settings = profile['exit_strategy']['trail_settings']
                data.ta.atr(length=settings['atr_period'], append=True)
                last_atr = data.get(f'ATRr_{settings["atr_period"]}', pd.Series([0])).iloc[-1]
                if pd.isna(last_atr): continue
                
                if is_call:
                    stop_price = current_price - (last_atr * settings['atr_multiplier'])
                    self.atr_stop_prices[position_key] = max(self.atr_stop_prices.get(position_key, 0), stop_price)
                    if current_price < self.atr_stop_prices[position_key]:
                        exit_reason = f"ATR Trail ({current_price:.2f} < {self.atr_stop_prices[position_key]:.2f})"
                else:
                    stop_price = current_price + (last_atr * settings['atr_multiplier'])
                    self.atr_stop_prices[position_key] = min(self.atr_stop_prices.get(position_key, float('inf')), stop_price)
                    if current_price > self.atr_stop_prices[position_key]:
                        exit_reason = f"ATR Trail ({current_price:.2f} > {self.atr_stop_prices[position_key]:.2f})"

            elif exit_type == "pullback_stop" and profile['exit_strategy']['trail_method'] == 'pullback_percent':
                pullback_pct = profile['exit_strategy']['trail_settings']['pullback_percent']
                if is_call:
                    stop_price = high_low['high'] * (1 - (pullback_pct / 100))
                    if current_price < stop_price: exit_reason = f"Pullback Stop ({pullback_pct}%)"
                else:
                    stop_price = high_low['low'] * (1 + (pullback_pct / 100))
                    if current_price > stop_price: exit_reason = f"Pullback Stop ({pullback_pct}%)"

        if exit_reason:
            self._close_simulated_position(timestamp, position_key, current_price, exit_reason)

    def _close_simulated_position(self, timestamp, position_key, exit_price, reason):
        """Closes a position in the simulation and logs the trade."""
        if position_key in self.portfolio['positions']:
            position = self.portfolio['positions'].pop(position_key)
            pnl = (exit_price - position['entry_price']) * position['quantity'] * 100
            self.portfolio['cash'] += (exit_price * position['quantity'] * 100)
            
            self.trade_log.append({
                'entry_time': position['entry_time'], 'exit_time': timestamp,
                'position': position_key, 'entry_price': position['entry_price'],
                'exit_price': exit_price, 'quantity': position['quantity'],
                'pnl': pnl, 'exit_reason': reason
            })
            logging.info(f"{timestamp} | CLOSED {position['quantity']} of {position_key} at ${exit_price:.2f} for P/L ${pnl:.2f}. Reason: {reason}")
    
    def _log_results(self):
        """Prints a summary of the backtest results."""
        df = pd.DataFrame(self.trade_log)
        if df.empty:
            logging.warning("No trades were executed during the simulation.")
            return

        total_pnl = df['pnl'].sum()
        win_rate = (df['pnl'] > 0).mean() * 100 if not df.empty else 0
        
        logging.info("\n--- Backtest Results Summary ---")
        logging.info(f"Total Trades: {len(df)}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Total P/L: ${total_pnl:.2f}")
        logging.info(f"Final Portfolio Value: ${self.portfolio['cash']:.2f}")
        logging.info("--------------------------------\n")
        
        log_path = os.path.join(self.data_folder_path, "backtest_results.csv")
        df.to_csv(log_path, index=False)
        logging.info(f"Full trade log saved to {log_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'data')

    engine = BacktestEngine(signal_file, data_folder)
    engine.run_simulation()