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

# FIX: Changed from "from utils import" to "from services.utils import"
from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename

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
        """
        Loads and parses signals from the signals_to_test.txt file.
        Supports both simple and timestamped formats.
        """
        signals = []
        if not os.path.exists(self.signal_file_path):
            logging.error(f"FATAL: signals_to_test.txt not found at '{self.signal_file_path}'")
            return []

        # Use first profile as default for parsing
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }

        with open(self.signal_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if line has timestamp format
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        timestamp_str = parts[0].strip()
                        channel = parts[1].strip()
                        signal_text = parts[2].strip()
                        
                        # Parse the signal
                        parsed_signal = self.signal_parser.parse_signal(signal_text, default_profile)
                        if parsed_signal:
                            # Add timestamp to parsed signal
                            parsed_signal['signal_timestamp'] = datetime.strptime(
                                timestamp_str, '%Y-%m-%d %H:%M:%S'
                            )
                            parsed_signal['channel'] = channel
                            signals.append(parsed_signal)
                            logging.info(f"Loaded timestamped signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]} at {timestamp_str}")
                    else:
                        logging.warning(f"Malformed timestamped line #{line_num}: '{line}'")
                else:
                    # Simple format without timestamp
                    parsed_signal = self.signal_parser.parse_signal(line, default_profile)
                    if parsed_signal:
                        # Assign a default timestamp (market open)
                        parsed_signal['signal_timestamp'] = datetime.now().replace(hour=9, minute=30, second=0)
                        parsed_signal['channel'] = 'default'
                        signals.append(parsed_signal)
                        logging.info(f"Loaded simple signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]}")
        
        logging.info(f"Successfully loaded {len(signals)} signals for backtesting")
        return signals

    def _create_event_queue(self, signals):
        """Creates a chronological event queue from signals and market data."""
        event_queue = []
        
        for signal in signals:
            # Add signal event
            event_queue.append((signal['signal_timestamp'], 'SIGNAL', signal))
            
            # Load corresponding historical data
            data_file = get_data_filename(signal, self.data_folder_path)
            if not os.path.exists(data_file):
                logging.warning(f"No data file found for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} at {data_file}")
                continue
            
            try:
                df = pd.read_csv(data_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add each tick as a TICK event
                for _, row in df.iterrows():
                    position_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
                    tick_data = {
                        'position_key': position_key,
                        'signal': signal,
                        'price': row['close'],
                        'volume': row.get('volume', 0),
                        'high': row.get('high', row['close']),
                        'low': row.get('low', row['close'])
                    }
                    event_queue.append((row['timestamp'], 'TICK', tick_data))
                
                logging.info(f"Added {len(df)} tick events for {position_key}")
            except Exception as e:
                logging.error(f"Error loading data for {signal}: {e}")
        
        logging.info(f"Created event queue with {len(event_queue)} total events")
        return event_queue

    def _process_signal_event(self, timestamp, signal):
        """Processes a signal event (opens a position)."""
        position_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
        
        if position_key in self.portfolio['positions']:
            logging.info(f"{timestamp} | Position already exists for {position_key}")
            return
        
        # Simulate opening a position
        entry_price = signal.get('entry_price', signal['strike'] * 0.02)  # Default 2% of strike
        quantity = 4  # Default quantity
        
        self.portfolio['positions'][position_key] = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'quantity': quantity,
            'signal': signal,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        # Deduct cost from cash
        cost = entry_price * quantity * 100
        self.portfolio['cash'] -= cost
        
        # Initialize position tracking
        self.position_data_cache[position_key] = pd.DataFrame()
        self.breakeven_activated[position_key] = False
        
        logging.info(f"{timestamp} | OPENED {quantity} of {position_key} at ${entry_price:.2f}")

    def _process_tick_event(self, timestamp, tick_data):
        """Processes a market tick event."""
        position_key = tick_data['position_key']
        
        if position_key not in self.portfolio['positions']:
            return  # No position to update
        
        position = self.portfolio['positions'][position_key]
        current_price = tick_data['price']
        
        # Update high/low tracking
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Add to data cache for technical indicators
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'close': current_price,
            'high': tick_data['high'],
            'low': tick_data['low'],
            'volume': tick_data['volume']
        }])
        
        self.position_data_cache[position_key] = pd.concat([
            self.position_data_cache[position_key], 
            new_row
        ], ignore_index=True)
        
        # Evaluate exit conditions
        self._evaluate_simulated_exit(timestamp, position_key, current_price)

    def _evaluate_simulated_exit(self, timestamp, position_key, current_price):
        """Evaluates exit conditions for a position."""
        if position_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][position_key]
        signal = position['signal']
        profile = self.config.profiles[0]  # Use first profile for backtesting
        
        entry_price = position['entry_price']
        is_call = signal['contract_type'] == 'CALL'
        
        # Get cached data
        data = self.position_data_cache[position_key]
        if len(data) < 2:
            return  # Need at least 2 rows for calculations
        
        exit_reason = None
        
        # Check each exit type in priority order
        for exit_type in ["breakeven", "atr_trail", "pullback_trail", "rsi_hook", "psar_flip"]:
            if exit_reason:
                break
            
            if exit_type == "breakeven":
                trigger_pct = profile['exit_strategy']['breakeven_trigger_percent']
                if not self.breakeven_activated.get(position_key):
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    if pnl_percent >= trigger_pct * 100:
                        self.breakeven_activated[position_key] = True
                        logging.info(f"Breakeven activated for {position_key} at {pnl_percent:.2f}% gain")
                
                if self.breakeven_activated.get(position_key):
                    if (is_call and current_price <= entry_price) or (not is_call and current_price >= entry_price):
                        exit_reason = f"Breakeven Stop (entry: ${entry_price:.2f})"
            
            elif exit_type == "atr_trail" and profile['exit_strategy']['trail_method'] == 'atr':
                settings = profile['exit_strategy']['trail_settings']
                data.ta.atr(length=settings['atr_period'], append=True)
                atr_col = f'ATRr_{settings["atr_period"]}'
                
                if atr_col in data.columns and not pd.isna(data[atr_col].iloc[-1]):
                    atr_value = data[atr_col].iloc[-1]
                    if is_call:
                        stop_price = position['highest_price'] - (atr_value * settings['atr_multiplier'])
                        if current_price <= stop_price:
                            exit_reason = f"ATR Trail Stop (ATR: {atr_value:.2f}, multiplier: {settings['atr_multiplier']})"
                    else:
                        stop_price = position['lowest_price'] + (atr_value * settings['atr_multiplier'])
                        if current_price >= stop_price:
                            exit_reason = f"ATR Trail Stop (ATR: {atr_value:.2f}, multiplier: {settings['atr_multiplier']})"
            
            elif exit_type == "pullback_trail" and profile['exit_strategy']['trail_method'] == 'pullback_percent':
                pullback_pct = profile['exit_strategy']['trail_settings']['pullback_percent']
                if is_call:
                    stop_price = position['highest_price'] * (1 - pullback_pct)
                    if current_price <= stop_price:
                        exit_reason = f"Pullback Trail ({pullback_pct*100:.0f}% from high ${position['highest_price']:.2f})"
                else:
                    stop_price = position['lowest_price'] * (1 + pullback_pct)
                    if current_price >= stop_price:
                        exit_reason = f"Pullback Trail ({pullback_pct*100:.0f}% from low ${position['lowest_price']:.2f})"
            
            elif exit_type == "rsi_hook" and profile['exit_strategy']['momentum_exits']['rsi_hook_enabled']:
                settings = profile['exit_strategy']['momentum_exits']['rsi_settings']
                data.ta.rsi(length=settings['period'], append=True)
                rsi_col = f'RSI_{settings["period"]}'
                
                if rsi_col not in data.columns: continue
                last_rsi = data[rsi_col].iloc[-1]
                if pd.isna(last_rsi) or len(data) < 2: continue

                prev_rsi = data[f'RSI_{settings["period"]}'].iloc[-2]
                if is_call and prev_rsi > settings['overbought_level'] and last_rsi <= settings['overbought_level']:
                    exit_reason = f"RSI Hook from Overbought ({prev_rsi:.2f} -> {last_rsi:.2f})"
                elif not is_call and prev_rsi < settings.get('oversold_level', 30) and last_rsi >= settings.get('oversold_level', 30):
                    exit_reason = f"RSI Hook from Oversold ({prev_rsi:.2f} -> {last_rsi:.2f})"
            
            elif exit_type == "psar_flip" and profile['exit_strategy']['momentum_exits']['psar_enabled']:
                settings = profile['exit_strategy']['momentum_exits']['psar_settings']
                data.ta.psar(initial=settings['start'], increment=settings['increment'], maximum=settings['max'], append=True)
                psar_long_col = f'PSARl_{settings["start"]}_{settings["max"]}'
                psar_short_col = f'PSARs_{settings["start"]}_{settings["max"]}'
                
                if psar_long_col in data.columns and not pd.isna(data[psar_long_col].iloc[-1]):
                    if is_call and current_price < data[psar_long_col].iloc[-1]:
                        exit_reason = "PSAR Flip"
                if psar_short_col in data.columns and not pd.isna(data[psar_short_col].iloc[-1]):
                    if not is_call and current_price > data[psar_short_col].iloc[-1]:
                        exit_reason = "PSAR Flip"

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
    data_folder = os.path.join(script_dir, 'historical_data')

    engine = BacktestEngine(signal_file, data_folder)
    engine.run_simulation()
