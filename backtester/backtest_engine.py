#!/usr/bin/env python3
"""
backtest_engine.py - FIXED VERSION WITH DEBUG LOGGING
This version includes fixes for the event queue issue and comprehensive debugging
"""

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
        
        logging.info("🔍 DEBUG: Backtest Engine initialized.")
        logging.info(f"🔍 DEBUG: Signal file: {signal_file_path}")
        logging.info(f"🔍 DEBUG: Data folder: {data_folder_path}")

    def run_simulation(self):
        """FIXED VERSION with proper event queue handling and debug logging"""
        logging.info("--- 🚀 Starting Backtest Simulation 🚀 ---")
        logging.info("🔍 DEBUG: run_simulation() called")
        
        # Load signals
        logging.info("🔍 DEBUG: About to load signals...")
        signals = self._load_signals()
        
        if not signals:
            logging.error("❌ No signals loaded - check signals_to_test.txt")
            return
        
        logging.info(f"🔍 DEBUG: Loaded {len(signals)} signals successfully")
        
        # Create event queue
        logging.info("🔍 DEBUG: About to create event queue...")
        event_queue = self._create_event_queue(signals)
        
        # FIXED: Proper event queue validation
        if event_queue is None:
            logging.error("❌ Failed to create event queue (returned None)")
            return
        
        if len(event_queue) == 0:
            logging.warning("⚠️ Event queue is empty - no events to process")
            logging.warning("🔍 DEBUG: This usually means no historical data files were found")
            return
        
        logging.info(f"🔍 DEBUG: Event queue created with {len(event_queue)} events")
        
        # Process events
        logging.info("🔍 DEBUG: Starting event processing loop...")
        events_processed = 0
        
        for timestamp, event_type, data in sorted(event_queue, key=lambda x: x[0]):
            events_processed += 1
            
            if events_processed % 1000 == 0:
                logging.info(f"🔍 DEBUG: Processed {events_processed}/{len(event_queue)} events...")
            
            if event_type == 'SIGNAL':
                logging.debug(f"🔍 DEBUG: Processing SIGNAL event at {timestamp}")
                self._process_signal_event(timestamp, data)
            elif event_type == 'TICK':
                # Don't log every tick (too verbose), but count them
                self._process_tick_event(timestamp, data)
        
        logging.info(f"🔍 DEBUG: Finished processing all {events_processed} events")
        
        # Log results
        self._log_results()
        logging.info("--- 🏁 Backtest Simulation Complete 🏁 ---")

    def _load_signals(self):
        """Load signals with debug logging"""
        signals = []
        
        logging.info(f"🔍 DEBUG: Looking for signals file at: {self.signal_file_path}")
        
        if not os.path.exists(self.signal_file_path):
            logging.error(f"❌ FATAL: signals_to_test.txt not found at '{self.signal_file_path}'")
            return []
        
        logging.info("🔍 DEBUG: Signal file exists, opening...")
        
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        with open(self.signal_file_path, 'r') as f:
            lines = f.readlines()
            logging.info(f"🔍 DEBUG: Read {len(lines)} lines from signal file")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                logging.debug(f"🔍 DEBUG: Processing line {line_num}: {line}")
                
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
                            logging.info(f"✅ Loaded timestamped signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]} at {timestamp_str}")
                        else:
                            logging.warning(f"⚠️ Could not parse signal on line {line_num}: {signal_text}")
                    else:
                        logging.warning(f"⚠️ Malformed timestamped line #{line_num}: '{line}'")
                else:
                    # Simple format
                    parsed_signal = self.signal_parser.parse_signal(line, default_profile)
                    if parsed_signal:
                        parsed_signal['signal_timestamp'] = datetime.now().replace(hour=9, minute=30, second=0)
                        parsed_signal['channel'] = 'default'
                        signals.append(parsed_signal)
                        logging.info(f"✅ Loaded simple signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]}")
                    else:
                        logging.warning(f"⚠️ Could not parse signal on line {line_num}: {line}")
        
        logging.info(f"🔍 DEBUG: Successfully loaded {len(signals)} signals for backtesting")
        return signals

    def _create_event_queue(self, signals):
        """Create event queue with debug logging"""
        event_queue = []
        
        logging.info(f"🔍 DEBUG: Creating event queue for {len(signals)} signals")
        
        for signal_num, signal in enumerate(signals, 1):
            logging.info(f"🔍 DEBUG: Processing signal {signal_num}/{len(signals)}: {signal['ticker']} {signal['strike']}{signal['contract_type'][0]}")
            
            # Add signal event
            event_queue.append((signal['signal_timestamp'], 'SIGNAL', signal))
            
            # Create contract for data lookup
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
            
            logging.info(f"🔍 DEBUG: Looking for data file: {data_file}")
            
            if not os.path.exists(data_file):
                logging.warning(f"⚠️ No data file found for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} at {data_file}")
                continue
            
            try:
                df = pd.read_csv(data_file)
                logging.info(f"🔍 DEBUG: Loaded {len(df)} rows from {data_filename}")
                
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                # Add tick events
                tick_count = 0
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
                    tick_count += 1
                
                logging.info(f"✅ Added {tick_count} tick events for {position_key}")
                
            except Exception as e:
                logging.error(f"❌ Error loading data for {signal}: {e}")
        
        logging.info(f"🔍 DEBUG: Created event queue with {len(event_queue)} total events")
        return event_queue

    def _process_signal_event(self, timestamp, signal):
        """Process signal event with debug logging"""
        position_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
        
        logging.debug(f"🔍 DEBUG: Processing signal event for {position_key} at {timestamp}")
        
        # Get actual entry price from historical data
        entry_price = self._get_entry_price_from_data(signal, timestamp)
        
        if entry_price is None:
            logging.warning(f"⚠️ Could not find entry price for {position_key} at {timestamp}, skipping")
            return
        
        # Use realistic position sizing (10% of portfolio per trade)
        position_size = self.portfolio['cash'] * 0.10
        quantity = int(position_size / (entry_price * 100))  # Divide by 100 for contract multiplier
        
        if quantity == 0:
            logging.warning(f"⚠️ Position size too small for {position_key}, skipping")
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
        
        logging.info(f"📈 [{timestamp}] OPENED {position_key} | Qty: {quantity} | Entry: ${entry_price:.2f}")

    def _get_entry_price_from_data(self, signal, signal_timestamp):
        """Get actual entry price from historical data"""
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
            logging.debug(f"🔍 DEBUG: Found real entry price ${entry_price:.2f} for {signal['ticker']} at {future_ticks.iloc[0]['timestamp']}")
            return entry_price
            
        except Exception as e:
            logging.error(f"Error reading data file for entry price: {e}")
            return None

    def _process_tick_event(self, timestamp, tick_data):
        """Process tick events"""
        position_key = tick_data['position_key']
        
        if position_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][position_key]
        current_price = tick_data['price']
        
        # Update tracking
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Buffer ticks for bar aggregation
        if position_key not in self.tick_buffer:
            self.tick_buffer[position_key] = []
        
        self.tick_buffer[position_key].append({
            'timestamp': timestamp,
            'close': current_price,
            'high': tick_data['high'],
            'low': tick_data['low'],
            'volume': tick_data['volume']
        })
        
        # Aggregate bars every 60 seconds
        if position_key not in self.last_bar_timestamp or \
           (timestamp - self.last_bar_timestamp[position_key]).total_seconds() >= 60:
            self._aggregate_bar(position_key, timestamp)
            self.last_bar_timestamp[position_key] = timestamp
        
        # Evaluate exit conditions
        exit_reason = self._evaluate_exit_conditions(position, current_price, timestamp)
        
        if exit_reason:
            self._close_position(position_key, current_price, timestamp, exit_reason)

    def _aggregate_bar(self, position_key, timestamp):
        """Aggregate ticks into bars"""
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
        """Evaluate all exit conditions"""
        entry_price = position['entry_price']
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        profile = self.config.profiles[0] if self.config.profiles else {}
        exit_strategy = profile.get('exit_strategy', {})
        
        # 1. Breakeven logic
        breakeven_trigger = exit_strategy.get('breakeven_trigger_percent', 10)
        if not position['breakeven_activated'] and pnl_percent >= breakeven_trigger:
            position['breakeven_activated'] = True
            logging.info(f"🔔 [{timestamp}] BREAKEVEN activated for {position['signal']['ticker']}")
        
        if position['breakeven_activated'] and current_price <= entry_price:
            return "Breakeven stop hit"
        
        # 2. Native trailing stop
        native_trail_percent = exit_strategy.get('native_trail_percent')
        if native_trail_percent:
            native_exit = self._check_native_trail(position, current_price, native_trail_percent)
            if native_exit:
                return native_exit
        
        # 3. Dynamic exits (ATR, pullback, PSAR, RSI)
        trail_method = exit_strategy.get('trail_method', 'pullback_percent')
        
        if trail_method == 'pullback_percent':
            pullback_percent = exit_strategy.get('trail_settings', {}).get('pullback_percent', 0.10)
            pullback_price = position['highest_price'] * (1 - pullback_percent)
            if current_price <= pullback_price:
                return f"Pullback stop hit ({pullback_percent*100:.0f}%)"
        
        elif trail_method == 'atr':
            atr_exit = self._check_atr_exit(position, current_price, timestamp)
            if atr_exit:
                return atr_exit
        
        # 4. PSAR exit
        if exit_strategy.get('momentum_exits', {}).get('psar_enabled', False):
            psar_exit = self._check_psar_exit(position, current_price)
            if psar_exit:
                return psar_exit
        
        # 5. RSI hook exit
        if exit_strategy.get('momentum_exits', {}).get('rsi_hook_enabled', False):
            rsi_exit = self._check_rsi_exit(position, current_price)
            if rsi_exit:
                return rsi_exit
        
        return None

    def _check_native_trail(self, position, current_price, trail_percent):
        """Check native trailing stop"""
        trail_price = position['highest_price'] * (1 - trail_percent / 100)
        if current_price <= trail_price:
            return f"Native trail stop hit ({trail_percent}%)"
        return None

    def _check_atr_exit(self, position, current_price, timestamp):
        """Check ATR-based exit"""
        if len(position['bars']) < 14:  # Need enough bars for ATR
            return None
        
        # Calculate ATR
        df = pd.DataFrame(position['bars'])
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        if atr is not None and not atr.empty:
            current_atr = atr.iloc[-1]
            profile = self.config.profiles[0] if self.config.profiles else {}
            atr_multiplier = profile.get('exit_strategy', {}).get('trail_settings', {}).get('atr_multiplier', 1.5)
            
            atr_stop = position['highest_price'] - (current_atr * atr_multiplier)
            
            if current_price <= atr_stop:
                return f"ATR stop hit ({atr_multiplier}x)"
        
        return None

    def _check_psar_exit(self, position, current_price):
        """Check Parabolic SAR exit"""
        if len(position['bars']) < 2:
            return None
        
        df = pd.DataFrame(position['bars'])
        psar = ta.psar(df['high'], df['low'])
        
        if psar is not None and not psar.empty:
            current_psar = psar['PSARl_0.02_0.2'].iloc[-1]  # Long PSAR
            if current_psar and current_price <= current_psar:
                return "PSAR stop hit"
        
        return None

    def _check_rsi_exit(self, position, current_price):
        """Check RSI hook exit"""
        if len(position['bars']) < 14:
            return None
        
        df = pd.DataFrame(position['bars'])
        rsi = ta.rsi(df['close'], length=14)
        
        if rsi is not None and not rsi.empty:
            current_rsi = rsi.iloc[-1]
            if current_rsi > 70:  # Overbought
                return "RSI overbought exit"
        
        return None

    def _close_position(self, position_key, exit_price, timestamp, reason):
        """Close a position and record the trade"""
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
        
        logging.info(f"📉 [{timestamp}] CLOSED {position_key} | Exit: ${exit_price:.2f} | P&L: ${pnl:.2f} | Reason: {reason}")
        
        del self.portfolio['positions'][position_key]

    def _log_results(self):
        """Log final results"""
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
            logging.info(f"📊 Detailed results saved to {output_file}")
        
        logging.info("🔍 DEBUG: _log_results() completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("🔍 DEBUG: Script started")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'historical_data')
    
    logging.info(f"🔍 DEBUG: Creating BacktestEngine...")
    engine = BacktestEngine(signal_file, data_folder)
    
    logging.info(f"🔍 DEBUG: Calling run_simulation()...")
    engine.run_simulation()
    
    logging.info("🔍 DEBUG: Script completed")
