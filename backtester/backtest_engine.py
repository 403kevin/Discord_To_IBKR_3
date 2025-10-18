import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from ib_insync import Contract
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename, get_data_filename_databento

class BacktestEngine:
    def __init__(self, signal_file_path, data_folder_path):
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        self.signal_file_path = signal_file_path
        self.data_folder_path = data_folder_path
        
        self.portfolio = {'cash': 100000, 'positions': {}}
        self.trade_log = []
        
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        self.trailing_highs_and_lows = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}
        
        logging.info("Backtest Engine initialized.")

    def run_simulation(self):
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
        signals = []
        if not os.path.exists(self.signal_file_path):
            logging.error(f"FATAL: signals_to_test.txt not found at '{self.signal_file_path}'")
            return []

        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }

        with open(self.signal_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        timestamp_str = parts[0].strip()
                        channel = parts[1].strip()
                        signal_text = parts[2].strip()
                        
                        parsed_signal = self.signal_parser.parse_signal(signal_text, default_profile)
                        if parsed_signal:
                            parsed_signal['signal_timestamp'] = datetime.strptime(
                                timestamp_str, '%Y-%m-%d %H:%M:%S'
                            )
                            parsed_signal['channel'] = channel
                            signals.append(parsed_signal)
                            logging.info(f"Loaded timestamped signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]} at {timestamp_str}")
                    else:
                        logging.warning(f"Malformed timestamped line #{line_num}: '{line}'")
                else:
                    parsed_signal = self.signal_parser.parse_signal(line, default_profile)
                    if parsed_signal:
                        parsed_signal['signal_timestamp'] = datetime.now().replace(hour=9, minute=30, second=0)
                        parsed_signal['channel'] = 'default'
                        signals.append(parsed_signal)
                        logging.info(f"Loaded simple signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]}")
        
        logging.info(f"Successfully loaded {len(signals)} signals for backtesting")
        return signals

    def _create_event_queue(self, signals):
        event_queue = []
        
        for signal in signals:
            event_queue.append((signal['signal_timestamp'], 'SIGNAL', signal))
            
            contract = Contract(
                symbol=signal['ticker'],
                lastTradeDateOrContractMonth=signal['expiry_date'],
                strike=signal['strike'],
                right=signal['contract_type'][0].upper()
            )
            
            # Use Databento filename format
            expiry_date = signal['expiry_date'].replace('-', '')  # Convert YYYY-MM-DD to YYYYMMDD
            data_filename = get_data_filename_databento(
                signal['ticker'],
                expiry_date,
                signal['strike'],
                signal['contract_type'][0].upper()
            )
            data_file = os.path.join(self.data_folder_path, data_filename)
            
            if not os.path.exists(data_file):
                logging.warning(f"No data file found for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} at {data_file}")
                continue
            
            try:
                df = pd.read_csv(data_file)
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
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
        """
        FIXED VERSION: Gets real entry price from historical data instead of hardcoding $1.50
        """
        position_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
        
        # NEW: Get actual entry price from historical data
        entry_price = self._get_entry_price_from_data(signal, timestamp)
        
        if entry_price is None:
            logging.warning(f"Could not find entry price for {position_key} at {timestamp}, skipping")
            return
        
        # Use realistic position sizing (10% of portfolio per trade)
        position_size = self.portfolio['cash'] * 0.10
        quantity = int(position_size / (entry_price * 100))  # Divide by 100 for contract multiplier
        
        if quantity == 0:
            logging.warning(f"Position size too small for {position_key}, skipping")
            return
        
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

    def _get_entry_price_from_data(self, signal, signal_timestamp):
        """
        NEW METHOD: Gets the actual entry price from historical data
        Returns the first tick price AFTER the signal timestamp
        """
        # Use Databento filename format
        expiry_date = signal['expiry_date'].replace('-', '')  # Convert YYYY-MM-DD to YYYYMMDD
        data_filename = get_data_filename_databento(
            signal['ticker'],
            expiry_date,
            signal['strike'],
            signal['contract_type'][0].upper()
        )
        data_file = os.path.join(self.data_folder_path, data_filename)
        
        if not os.path.exists(data_file):
            logging.warning(f"No data file for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]}")
            return None
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
            # Find first tick AFTER signal timestamp
            future_ticks = df[df['timestamp'] >= signal_timestamp]
            
            if future_ticks.empty:
                logging.warning(f"No price data after signal timestamp for {signal['ticker']}")
                return None
            
            entry_price = future_ticks.iloc[0]['close']
            logging.debug(f"Found real entry price ${entry_price:.2f} for {signal['ticker']} at {future_ticks.iloc[0]['timestamp']}")
            return entry_price
            
        except Exception as e:
            logging.error(f"Error reading data file for entry price: {e}")
            return None

    def _process_tick_event(self, timestamp, tick_data):
        position_key = tick_data['position_key']
        
        if position_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][position_key]
        current_price = tick_data['price']
        
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        if position_key not in self.tick_buffer:
            self.tick_buffer[position_key] = []
        
        self.tick_buffer[position_key].append({
            'timestamp': timestamp,
            'close': current_price,
            'high': tick_data['high'],
            'low': tick_data['low'],
            'volume': tick_data['volume']
        })
        
        if position_key not in self.last_bar_timestamp or \
           (timestamp - self.last_bar_timestamp[position_key]).total_seconds() >= 60:
            self._aggregate_bar(position_key, timestamp)
            self.last_bar_timestamp[position_key] = timestamp
        
        exit_reason = self._evaluate_exit_conditions(position, current_price, timestamp)
        
        if exit_reason:
            self._close_position(position_key, current_price, timestamp, exit_reason)

    def _aggregate_bar(self, position_key, timestamp):
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
        
        self.tick_buffer[position_key] = []

    def _evaluate_exit_conditions(self, position, current_price, timestamp):
        entry_price = position['entry_price']
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        profile = self.config.profiles[0] if self.config.profiles else {}
        exit_strategy = profile.get('exit_strategy', {})
        
        # 1. Breakeven
        breakeven_trigger = exit_strategy.get('breakeven_trigger_percent', 10)
        if not position['breakeven_activated'] and pnl_percent >= breakeven_trigger:
            position['breakeven_activated'] = True
            logging.info(f"[{timestamp}] BREAKEVEN activated for {position['signal']['ticker']}")
        
        if position['breakeven_activated'] and current_price <= entry_price:
            return "Breakeven stop hit"
        
        # 2. Native Trail
        native_trail_percent = exit_strategy.get('native_trail_percent')
        if native_trail_percent:
            native_exit = self._check_native_trail(position, current_price, native_trail_percent)
            if native_exit:
                return native_exit
        
        # 3. Trail Method
        trail_method = exit_strategy.get('trail_method', 'atr')
        if trail_method == 'atr':
            atr_exit = self._check_atr_trail(position, current_price)
            if atr_exit:
                return atr_exit
        elif trail_method == 'pullback_percent':
            pullback_exit = self._check_pullback_stop(position, current_price)
            if pullback_exit:
                return pullback_exit
        
        # 4. Momentum Exits
        momentum_exits = exit_strategy.get('momentum_exits', {})
        
        if momentum_exits.get('rsi_hook_enabled', False):
            rsi_exit = self._check_rsi_hook(position, current_price)
            if rsi_exit:
                return rsi_exit
        
        if momentum_exits.get('psar_enabled', False):
            psar_exit = self._check_psar_flip(position, current_price)
            if psar_exit:
                return psar_exit
        
        return None

    def _check_native_trail(self, position, current_price, trail_percent):
        """Native trailing stop at fixed percentage from high"""
        trail_stop = position['highest_price'] * (1 - trail_percent / 100)
        
        if current_price <= trail_stop:
            return f"Native trail stop ({trail_percent}% from high)"
        
        return None

    def _check_rsi_hook(self, position, current_price):
        if len(position['bars']) < 14:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            momentum_exits = self.config.profiles[0]['exit_strategy'].get('momentum_exits', {})
            rsi_settings = momentum_exits.get('rsi_settings', {})
            rsi_period = rsi_settings.get('period', 14)
            
            rsi_series = ta.rsi(df['close'], length=rsi_period)
            
            if rsi_series is None or len(rsi_series) == 0:
                return None
            
            current_rsi = rsi_series.iloc[-1]
            
            if pd.isna(current_rsi):
                return None
            
            overbought = rsi_settings.get('overbought_level', 70)
            
            if current_rsi >= overbought:
                peak_price = position['highest_price']
                if current_price < peak_price * 0.98:
                    return f"RSI hook (RSI: {current_rsi:.1f})"
        
        except Exception as e:
            logging.warning(f"Error in RSI calculation: {e}")
            return None
        
        return None

    def _check_psar_flip(self, position, current_price):
        if len(position['bars']) < 5:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            momentum_exits = self.config.profiles[0]['exit_strategy'].get('momentum_exits', {})
            psar_settings = momentum_exits.get('psar_settings', {})
            
            psar_series = ta.psar(
                df['high'],
                df['low'],
                df['close'],
                af=psar_settings.get('start', 0.02),
                max_af=psar_settings.get('max', 0.2)
            )
            
            if psar_series is None or 'PSARl_0.02_0.2' not in psar_series.columns:
                return None
            
            psar = psar_series['PSARl_0.02_0.2'].iloc[-1]
            
            if pd.isna(psar):
                return None
            
            if current_price < psar:
                return f"PSAR flip (PSAR: ${psar:.2f})"
        
        except Exception as e:
            logging.warning(f"Error in PSAR calculation: {e}")
            return None
        
        return None

    def _check_atr_trail(self, position, current_price):
        if len(position['bars']) < 14:
            return None
        
        try:
            df = pd.DataFrame(position['bars'])
            trail_settings = self.config.profiles[0]['exit_strategy'].get('trail_settings', {})
            atr_period = trail_settings.get('atr_period', 14)
            atr_multiplier = trail_settings.get('atr_multiplier', 1.5)
            
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
            
            if atr_series is None or len(atr_series) == 0:
                return None
            
            if len(atr_series) < 1:
                return None
            
            atr = atr_series.iloc[-1]
            
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
        """
        CRITICAL FIX: This method now correctly handles pullback percentages.
        
        Config stores: pullback_percent = 0.10 (meaning 10% as a decimal)
        This method now treats it as a decimal (0.10 = 10%)
        
        OLD BUG: pullback_stop = high * (1 - pullback_percent / 100)
                 Result: 0.10 / 100 = 0.001 = 0.1% instead of 10%
        
        NEW FIX: pullback_stop = high * (1 - pullback_percent)
                 Result: 1 - 0.10 = 0.90 = 90% of high = correct 10% pullback
        """
        trail_settings = self.config.profiles[0]['exit_strategy'].get('trail_settings', {})
        pullback_percent = trail_settings.get('pullback_percent', 0.10)  # Already a decimal (0.10 = 10%)
        
        # FIXED: Removed the "/ 100" that was causing the bug
        pullback_stop = position['highest_price'] * (1 - pullback_percent)
        
        if current_price <= pullback_stop:
            # Convert back to percentage for display
            display_percent = pullback_percent * 100
            return f"Pullback stop ({display_percent}% from high)"
        
        return None

    def _close_position(self, position_key, exit_price, timestamp, reason):
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
