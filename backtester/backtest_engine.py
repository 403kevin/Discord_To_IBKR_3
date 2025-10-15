import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from ib_insync import Contract
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
    FIXED: Added null checks for technical indicators.
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
            
            # FIX: Create a proper Contract object to pass to get_data_filename()
            contract = Contract(
                symbol=signal['ticker'],
                lastTradeDateOrContractMonth=signal['expiry_date'],
                strike=signal['strike'],
                right=signal['contract_type'][0].upper()
            )
            
            # Now get the filename using the contract object
            data_filename = get_data_filename(contract)
            data_file = os.path.join(self.data_folder_path, data_filename)
            
            if not os.path.exists(data_file):
                logging.warning(f"No data file found for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} at {data_file}")
                continue
            
            try:
                df = pd.read_csv(data_file)
                # FIX: Strip timezone info to make timestamps naive for comparison
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
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
        
        # Assume entry at a default price (this would be refined in a real backtest)
        entry_price = 1.50
        quantity = int(1000 / entry_price)  # Simple sizing
        
        self.portfolio['positions'][position_key] = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'quantity': quantity,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'bars': [],
            'breakeven_activated': False
        }
        
        self.portfolio['cash'] -= (entry_price * quantity * 100)
        
        logging.info(f"[{timestamp}] OPENED {position_key} | Qty: {quantity} | Entry: ${entry_price:.2f}")

    def _process_tick_event(self, timestamp, tick_data):
        """Processes a market tick event and evaluates exit conditions."""
        position_key = tick_data['position_key']
        
        if position_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][position_key]
        current_price = tick_data['price']
        
        # Update highest/lowest for trailing logic
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Build bar data for technical indicators (simulated 1-min bars)
        if position_key not in self.tick_buffer:
            self.tick_buffer[position_key] = []
        
        self.tick_buffer[position_key].append({
            'timestamp': timestamp,
            'close': current_price,
            'high': tick_data['high'],
            'low': tick_data['low'],
            'volume': tick_data['volume']
        })
        
        # Every 60 seconds, aggregate a bar
        if position_key not in self.last_bar_timestamp or \
           (timestamp - self.last_bar_timestamp[position_key]).total_seconds() >= 60:
            self._aggregate_bar(position_key, timestamp)
            self.last_bar_timestamp[position_key] = timestamp
        
        # Evaluate exit conditions
        exit_reason = self._evaluate_exit_conditions(position, current_price, timestamp)
        
        if exit_reason:
            self._close_position(position_key, current_price, timestamp, exit_reason)

    def _aggregate_bar(self, position_key, timestamp):
        """Aggregates tick buffer into a 1-minute bar."""
        if position_key not in self.tick_buffer or not self.tick_buffer[position_key]:
            return
        
        ticks = self.tick_buffer[position_key]
        bar = {
            'timestamp': timestamp,
            'open': ticks[0]['close'],
            'high': max(t['high'] for t in ticks),
            'low': min(t['low'] for t in ticks),
            'close': ticks[-1]['close'],
            'volume': sum(t['volume'] for t in ticks)
        }
        
        position = self.portfolio['positions'].get(position_key)
        if position:
            position['bars'].append(bar)
        
        # Clear buffer
        self.tick_buffer[position_key] = []

    def _evaluate_exit_conditions(self, position, current_price, timestamp):
        """Evaluates all exit conditions in priority order with NULL CHECKS."""
        entry_price = position['entry_price']
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # Get config settings
        profile = self.config.profiles[0] if self.config.profiles else {}
        exit_strategy = profile.get('exit_strategy', {})
        
        # 1. Breakeven trigger
        breakeven_trigger = exit_strategy.get('breakeven_trigger_percent', 10)
        if not position['breakeven_activated'] and pnl_percent >= breakeven_trigger:
            position['breakeven_activated'] = True
            logging.info(f"[{timestamp}] BREAKEVEN activated for {position['signal']['ticker']}")
        
        if position['breakeven_activated'] and current_price <= entry_price:
            return "Breakeven stop hit"
        
        # 2. RSI Hook (check before ATR to prioritize momentum)
        momentum_exits = exit_strategy.get('momentum_exits', {})
        if momentum_exits.get('rsi_hook_enabled', False) and len(position['bars']) >= 14:
            rsi_exit = self._check_rsi_hook(position, current_price)
            if rsi_exit:
                return rsi_exit
        
        # 3. PSAR Flip
        if momentum_exits.get('psar_enabled', False) and len(position['bars']) >= 5:
            psar_exit = self._check_psar_flip(position, current_price)
            if psar_exit:
                return psar_exit
        
        # 4. ATR trailing stop
        trail_method = exit_strategy.get('trail_method', 'pullback_percent')
        if trail_method == 'atr' and len(position['bars']) >= 14:
            atr_exit = self._check_atr_trail(position, current_price)
            if atr_exit:
                return atr_exit
        
        # 5. Pullback percent trailing stop (fallback)
        pullback_exit = self._check_pullback_stop(position, current_price)
        if pullback_exit:
            return pullback_exit
        
        return None

    def _check_rsi_hook(self, position, current_price):
        """Check RSI hook exit with NULL protection."""
        if len(position['bars']) < 14:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            rsi_settings = self.config.profiles[0]['exit_strategy']['momentum_exits'].get('rsi_settings', {})
            rsi_period = rsi_settings.get('period', 14)
            overbought = rsi_settings.get('overbought_level', 70)
            oversold = rsi_settings.get('oversold_level', 30)
            
            # ✅ NULL CHECK - Calculate RSI
            rsi_series = ta.rsi(df['close'], length=rsi_period)
            
            if rsi_series is None or len(rsi_series) == 0:
                return None
            
            # Get last two RSI values
            if len(rsi_series) < 2:
                return None
                
            current_rsi = rsi_series.iloc[-1]
            previous_rsi = rsi_series.iloc[-2]
            
            # ✅ NULL CHECK - Verify RSI values are valid
            if pd.isna(current_rsi) or pd.isna(previous_rsi):
                return None
            
            # Check for hook from overbought
            if previous_rsi > overbought and current_rsi < overbought:
                return f"RSI Hook from overbought ({current_rsi:.1f})"
            
            # Check for hook from oversold
            if previous_rsi < oversold and current_rsi > oversold:
                return f"RSI Hook from oversold ({current_rsi:.1f})"
        
        except Exception as e:
            logging.warning(f"Error in RSI calculation: {e}")
            return None
        
        return None

    def _check_psar_flip(self, position, current_price):
        """Check PSAR flip exit with NULL protection."""
        if len(position['bars']) < 5:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            psar_settings = self.config.profiles[0]['exit_strategy']['momentum_exits'].get('psar_settings', {})
            
            # ✅ NULL CHECK - Calculate PSAR
            psar_series = ta.psar(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                af0=psar_settings.get('start', 0.02),
                af=psar_settings.get('increment', 0.02),
                max_af=psar_settings.get('max', 0.2)
            )
            
            if psar_series is None or len(psar_series) == 0:
                return None
            
            # Get PSARl and PSARs columns
            if isinstance(psar_series, pd.DataFrame):
                psar_long = psar_series.get('PSARl_0.02_0.2')
                psar_short = psar_series.get('PSARs_0.02_0.2')
                
                if psar_long is None or psar_short is None:
                    return None
                
                # ✅ NULL CHECK - Get last values
                if len(psar_long) == 0 or len(psar_short) == 0:
                    return None
                
                last_psar_long = psar_long.iloc[-1]
                last_psar_short = psar_short.iloc[-1]
                
                # ✅ NULL CHECK - Verify values are valid
                if pd.isna(last_psar_long) and pd.isna(last_psar_short):
                    return None
                
                # If PSARs has a value (bearish), exit
                if not pd.isna(last_psar_short):
                    return f"PSAR Flip (bearish)"
        
        except Exception as e:
            logging.warning(f"Error in PSAR calculation: {e}")
            return None
        
        return None

    def _check_atr_trail(self, position, current_price):
        """Check ATR trailing stop with NULL protection."""
        if len(position['bars']) < 14:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            trail_settings = self.config.profiles[0]['exit_strategy'].get('trail_settings', {})
            atr_period = trail_settings.get('atr_period', 14)
            atr_multiplier = trail_settings.get('atr_multiplier', 1.5)
            
            # ✅ NULL CHECK - Calculate ATR
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
            
            if atr_series is None or len(atr_series) == 0:
                return None
            
            # ✅ NULL CHECK - Get last ATR value
            if len(atr_series) < 1:
                return None
            
            atr = atr_series.iloc[-1]
            
            # ✅ NULL CHECK - Verify ATR is valid
            if pd.isna(atr) or atr <= 0:
                return None
            
            atr_stop = position['highest_price'] - (atr * atr_multiplier)
            
            if current_price <= atr_stop:
                return f"ATR trail stop (ATR: {atr:.2f})"
        
        except Exception as e:
            logging.warning(f"Error in ATR calculation: {e}")
            return None
        
        return None

    def _check_pullback_stop(self, position, current_price):
        """Check pullback percentage stop."""
        trail_settings = self.config.profiles[0]['exit_strategy'].get('trail_settings', {})
        pullback_percent = trail_settings.get('pullback_percent', 10)
        
        pullback_stop = position['highest_price'] * (1 - pullback_percent / 100)
        
        if current_price <= pullback_stop:
            return f"Pullback stop ({pullback_percent}% from high)"
        
        return None

    def _close_position(self, position_key, exit_price, timestamp, reason):
        """Closes a position and logs the trade."""
        position = self.portfolio['positions'][position_key]
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        pnl = (exit_price - entry_price) * quantity * 100
        
        self.portfolio['cash'] += (exit_price * quantity * 100)
        
        trade_record = {
            'ticker': position['signal']['ticker'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'reason': reason
        }
        
        self.trade_log.append(trade_record)
        
        logging.info(f"[{timestamp}] CLOSED {position_key} | Exit: ${exit_price:.2f} | P&L: ${pnl:.2f} | Reason: {reason}")
        
        del self.portfolio['positions'][position_key]

    def _log_results(self):
        """Logs final backtest results."""
        logging.info("\n" + "="*80)
        logging.info("BACKTEST RESULTS SUMMARY")
        logging.info("="*80)
        
        total_trades = len(self.trade_log)
        winning_trades = sum(1 for t in self.trade_log if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trade_log)
        
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Winning Trades: {winning_trades}")
        logging.info(f"Win Rate: {(winning_trades/total_trades*100):.1f}%" if total_trades > 0 else "N/A")
        logging.info(f"Total P&L: ${total_pnl:.2f}")
        logging.info(f"Final Portfolio Value: ${self.portfolio['cash']:.2f}")
        logging.info("="*80)
        
        # Save detailed results
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            output_file = os.path.join(self.data_folder_path, '../backtest_results.csv')
            df.to_csv(output_file, index=False)
            logging.info(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'historical_data')
    
    engine = BacktestEngine(signal_file, data_folder)
    engine.run_simulation()
