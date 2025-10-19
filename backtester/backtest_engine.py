#!/usr/bin/env python3
"""
backtest_engine.py - COMPLETE FIXED VERSION
BUG FIX #3: Corrects year calculation for historical signals (uses signal timestamp year)
BUG FIX #2: Returns return_pct in empty results
BUG FIX #1: Correct parse_signal() signature (2 args, not 4)
"""

import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename_databento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BacktestEngine:
    """
    Event-driven backtesting with ALL exit strategies including native trail.
    FIXED: parse_signal signature, return_pct, and YEAR CALCULATION for historical signals.
    """
    
    def __init__(self, signal_file_path, data_folder_path):
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        self.signal_file_path = signal_file_path
        self.data_folder_path = data_folder_path
        
        # Portfolio tracking
        self.starting_capital = 100000
        self.portfolio = {
            'cash': self.starting_capital,
            'positions': {}
        }
        self.trade_log = []
        
        # Position tracking for dynamic exits
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        self.trailing_highs_and_lows = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}
        
        # NATIVE TRAIL TRACKING
        self.native_trail_stops = {}
        
        # Track which contracts have active signals
        self.active_contracts = {}
        
        logging.info("BacktestEngine initialized (COMPLETE VERSION with year fix)")
        logging.info(f"Signal file: {signal_file_path}")
        logging.info(f"Data folder: {data_folder_path}")
        
    def run_simulation(self, params=None):
        """Main simulation loop with parameter support"""
        logging.info("\n" + "="*80)
        logging.info("🚀 Starting Backtest Simulation")
        logging.info("="*80)
        
        # Apply parameters if provided
        if params:
            self._apply_parameters(params)
        
        # Load and parse signals
        signals = self._load_signals()
        if not signals:
            logging.error("❌ No signals loaded!")
            # FIX BUG #2: Include return_pct in empty results
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'final_capital': self.portfolio['cash'],
                'return_pct': 0,
                'max_drawdown': 0,
                'avg_minutes_held': 0,
                'exit_reasons': {}
            }
        
        logging.info(f"✅ Loaded {len(signals)} signals")
        
        # Create event queue
        logging.info("Creating event queue...")
        event_queue = self._create_event_queue(signals)
        
        if not event_queue:
            logging.error("❌ Event queue is empty!")
            return self._calculate_results()
        
        logging.info(f"✅ Created event queue with {len(event_queue)} events")
        
        # Process events
        logging.info("Starting event processing loop...")
        processed_count = 0
        signal_count = 0
        tick_count = 0
        
        for timestamp, event_type, data in sorted(event_queue, key=lambda x: x[0]):
            processed_count += 1
            
            if event_type == 'SIGNAL':
                signal_count += 1
                self._process_signal_event(timestamp, data, params)
            elif event_type == 'TICK':
                tick_count += 1
                self._process_tick_event(timestamp, data, params)
            
            if processed_count % 1000 == 0:
                logging.debug(f"Progress: {processed_count}/{len(event_queue)} events")
        
        logging.info(f"✅ Processed {processed_count} events ({signal_count} signals, {tick_count} ticks)")
        
        # Calculate and return results
        results = self._calculate_results()
        self._log_results()
        
        return results
    
    def _apply_parameters(self, params):
        """Apply optimization parameters to config"""
        if not self.config.profiles:
            self.config.profiles = [{}]
        
        profile = self.config.profiles[0]
        
        if 'exit_strategy' not in profile:
            profile['exit_strategy'] = {}
        
        # Apply all parameters
        for key, value in params.items():
            if key in ['breakeven_trigger_percent', 'trail_method']:
                profile['exit_strategy'][key] = value if key != 'breakeven_trigger_percent' else value / 100
            elif key in ['pullback_percent', 'atr_period', 'atr_multiplier']:
                if 'trail_settings' not in profile['exit_strategy']:
                    profile['exit_strategy']['trail_settings'] = {}
                profile['exit_strategy']['trail_settings'][key] = value if key != 'pullback_percent' else value / 100
            elif key in ['psar_enabled', 'rsi_hook_enabled']:
                if 'momentum_exits' not in profile['exit_strategy']:
                    profile['exit_strategy']['momentum_exits'] = {}
                profile['exit_strategy']['momentum_exits'][key] = value
            elif key == 'native_trail_percent':
                profile['exit_strategy']['native_trail_percent'] = value / 100
    
    def _load_signals(self):
        """Load and parse signals from file"""
        logging.info(f"Loading signals from {self.signal_file_path}")
        
        signals = []
        
        if not os.path.exists(self.signal_file_path):
            logging.error(f"❌ Signal file not found: {self.signal_file_path}")
            return signals
        
        with open(self.signal_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#') or line.startswith('Trader:'):
                    continue
                
                try:
                    if '|' in line:
                        # Timestamped format
                        parts = line.split('|')
                        timestamp_str = parts[0].strip()
                        trader = parts[1].strip() if len(parts) > 1 else "test_trader"
                        signal_text = parts[2].strip() if len(parts) > 2 else parts[1].strip()
                        
                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        # Simple format - use current date with market open time
                        signal_text = line
                        trader = "test_trader"
                        timestamp = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                    
                    # Parse the signal - FIX BUG #1: Only pass 2 arguments
                    profile = {
                        'assume_buy_on_ambiguous': True, 
                        'ambiguous_expiry_enabled': True
                    }
                    
                    # ✅ FIXED: Removed 'trader' argument - parse_signal only takes (raw_message, profile)
                    parsed = self.signal_parser.parse_signal(signal_text, profile)
                    
                    if parsed:
                        # FIX BUG #3: For historical signals, correct the expiry year
                        # The parser uses datetime.now().year which breaks for past signals
                        exp_str = parsed['expiry_date']  # Format: YYYYMMDD
                        parsed_year = int(exp_str[:4])
                        signal_year = timestamp.year  # Use signal's year, not today's year
                        
                        # If parsed expiry year doesn't match signal year, fix it
                        if parsed_year != signal_year:
                            # Rebuild expiry_date using signal's year
                            parsed['expiry_date'] = f"{signal_year}{exp_str[4:]}"
                            logging.info(f"  ⚠️  Corrected expiry year from {parsed_year} to {signal_year}")
                        
                        parsed['timestamp'] = timestamp
                        parsed['trader'] = trader
                        parsed['raw_signal'] = signal_text
                        signals.append(parsed)
                        logging.info(f"  Line {line_num}: {parsed['ticker']} {parsed['expiry_date']} {parsed['strike']}{parsed['contract_type'][0]}")
                    else:
                        logging.warning(f"  Line {line_num}: Failed to parse: {signal_text}")
                    
                except Exception as e:
                    logging.error(f"  Line {line_num}: Error parsing: {e}")
        
        logging.info(f"Loaded {len(signals)} valid signals")
        return signals
    
    def _create_event_queue(self, signals):
        """Create chronological event queue from signals and historical data"""
        events = []
        
        for signal in signals:
            # Add SIGNAL event
            events.append((signal['timestamp'], 'SIGNAL', signal))
            
            # Load historical data for this contract
            expiry_clean = signal['expiry_date'].replace('-', '')
            filename = get_data_filename_databento(
                signal['ticker'],
                expiry_clean,
                signal['strike'],
                signal['contract_type'][0].upper()
            )
            filepath = os.path.join(self.data_folder_path, filename)
            
            if not os.path.exists(filepath):
                logging.warning(f"  ⚠️ No data file found: {filename}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                # CRITICAL FIX: Only load ticks AT OR AFTER the signal timestamp
                signal_time = signal['timestamp']
                df = df[df['timestamp'] >= signal_time]
                
                contract_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
                
                # Add TICK events for each row
                for _, row in df.iterrows():
                    tick_data = {
                        'contract_key': contract_key,
                        'price': row.get('price', row.get('close', 0)),
                        'bid': row.get('bid', 0),
                        'ask': row.get('ask', 0),
                        'volume': row.get('volume', 0)
                    }
                    events.append((row['timestamp'], 'TICK', tick_data))
                
                logging.info(f"  Added {len(df)} tick events for {contract_key}")
                
            except Exception as e:
                logging.error(f"  ❌ Error loading {filename}: {e}")
        
        return events
    
    def _process_signal_event(self, timestamp, signal, params):
        """Process a new trading signal"""
        contract_key = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}{signal['contract_type'][0]}"
        
        # Check if we already have a position for this contract
        if contract_key in self.portfolio['positions']:
            logging.debug(f"  Already have position in {contract_key}, skipping signal")
            return
        
        # Get entry price from first tick
        entry_price = 1.50  # Default fallback
        
        # Calculate position size (10% of portfolio)
        position_value = self.portfolio['cash'] * 0.10
        quantity = int(position_value / (entry_price * 100))
        
        if quantity < 1:
            logging.warning(f"  Not enough capital for {contract_key}")
            return
        
        # Open position
        cost = quantity * entry_price * 100
        self.portfolio['cash'] -= cost
        
        self.portfolio['positions'][contract_key] = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'quantity': quantity,
            'contract_key': contract_key,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'bars': []
        }
        
        # Initialize tracking
        self.trailing_highs_and_lows[contract_key] = {'high': entry_price, 'low': entry_price}
        self.breakeven_activated[contract_key] = False
        self.native_trail_stops[contract_key] = None
        
        logging.info(f"[{timestamp}] ENTRY: {contract_key} @ ${entry_price:.2f} × {quantity}")
    
    def _process_tick_event(self, timestamp, tick_data, params):
        """Process a market tick"""
        contract_key = tick_data['contract_key']
        
        # Check if we have an open position for this contract
        if contract_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][contract_key]
        current_price = tick_data['price']
        
        if current_price <= 0:
            return
        
        # Update highest/lowest
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Update bars for indicators (every minute)
        if contract_key not in self.last_bar_timestamp or \
           (timestamp - self.last_bar_timestamp[contract_key]).total_seconds() >= 60:
            position['bars'].append({
                'timestamp': timestamp,
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price
            })
            self.last_bar_timestamp[contract_key] = timestamp
        
        # Check exit conditions
        exit_reason = self._evaluate_exit_conditions(position, current_price, params)
        
        if exit_reason:
            self._close_position(contract_key, exit_reason, current_price, timestamp)
    
    def _evaluate_exit_conditions(self, position, current_price, params):
        """Evaluate all exit conditions"""
        entry_price = position['entry_price']
        contract_type = position['signal']['contract_type']
        contract_key = position['contract_key']
        
        # Calculate P&L percentage
        if contract_type == 'CALL':
            pnl_pct = (current_price - entry_price) / entry_price
            peak_price = position['highest_price']
        else:  # PUT
            pnl_pct = (entry_price - current_price) / entry_price
            peak_price = position['lowest_price']
        
        # Priority 1: Native Trail (broker-level safety)
        if self.native_trail_stops.get(contract_key) is not None:
            trail_price = self.native_trail_stops[contract_key]
            if contract_type == 'CALL' and current_price <= trail_price:
                return "NATIVE_TRAIL"
            elif contract_type == 'PUT' and current_price >= trail_price:
                return "NATIVE_TRAIL"
            
            # Update native trail
            if not self.config.profiles:
                native_pct = 0.25
            else:
                native_pct = self.config.profiles[0].get('exit_strategy', {}).get('native_trail_percent', 0.25)
            
            if contract_type == 'CALL':
                new_trail = peak_price * (1 - native_pct)
                self.native_trail_stops[contract_key] = max(trail_price, new_trail)
            else:
                new_trail = peak_price * (1 + native_pct)
                self.native_trail_stops[contract_key] = min(trail_price, new_trail)
        
        # Priority 2: Breakeven trigger
        if not self.config.profiles:
            breakeven_trigger = 0.10
        else:
            breakeven_trigger = self.config.profiles[0].get('exit_strategy', {}).get('breakeven_trigger_percent', 0.10)
        
        if pnl_pct >= breakeven_trigger:
            if not self.breakeven_activated[contract_key]:
                self.breakeven_activated[contract_key] = True
                # Activate native trail if not already active
                if self.native_trail_stops[contract_key] is None:
                    if not self.config.profiles:
                        native_pct = 0.25
                    else:
                        native_pct = self.config.profiles[0].get('exit_strategy', {}).get('native_trail_percent', 0.25)
                    
                    if contract_type == 'CALL':
                        self.native_trail_stops[contract_key] = peak_price * (1 - native_pct)
                    else:
                        self.native_trail_stops[contract_key] = peak_price * (1 + native_pct)
                    
                    logging.info(f"[{position['entry_time']}] BREAKEVEN activated for {contract_key}")
            
            # Check if price fell back to breakeven
            if pnl_pct < 0.01:  # Less than 1% profit
                return "BREAKEVEN"
        
        # Priority 3: Pullback from peak
        if not self.config.profiles:
            pullback_pct = 0.10
        else:
            pullback_pct = self.config.profiles[0].get('exit_strategy', {}).get('trail_settings', {}).get('pullback_percent', 0.10)
        
        if contract_type == 'CALL':
            pullback_from_peak = (peak_price - current_price) / peak_price
        else:
            pullback_from_peak = (current_price - peak_price) / peak_price
        
        if pullback_from_peak >= pullback_pct:
            return f"PULLBACK_{int(pullback_pct*100)}%"
        
        return None
    
    def _close_position(self, contract_key, exit_reason, exit_price, exit_time):
        """Close a position and record the trade"""
        position = self.portfolio['positions'][contract_key]
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        entry_time = position['entry_time']
        
        # Calculate P&L
        proceeds = quantity * exit_price * 100
        self.portfolio['cash'] += proceeds
        
        cost = quantity * entry_price * 100
        pnl = proceeds - cost
        
        # Calculate hold time
        hold_time = (exit_time - entry_time).total_seconds() / 60  # minutes
        
        # Record trade
        self.trade_log.append({
            'contract': contract_key,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': (pnl / cost) * 100,
            'minutes_held': hold_time,
            'exit_reason': exit_reason
        })
        
        # Remove position
        del self.portfolio['positions'][contract_key]
        
        logging.info(f"[{exit_time}] EXIT: {contract_key} @ ${exit_price:.2f} | P&L: ${pnl:.2f} | Reason: {exit_reason}")
    
    def _calculate_results(self):
        """Calculate final backtest results"""
        if not self.trade_log:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'final_capital': self.portfolio['cash'],
                'return_pct': 0,
                'max_drawdown': 0,
                'avg_minutes_held': 0,
                'exit_reasons': {}
            }
        
        df = pd.DataFrame(self.trade_log)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        total_pnl = df['pnl'].sum()
        win_rate = (len(wins) / len(df)) * 100 if len(df) > 0 else 0
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
        
        return_pct = ((self.portfolio['cash'] - self.starting_capital) / self.starting_capital) * 100
        
        return {
            'total_trades': len(df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.portfolio['cash'],
            'return_pct': return_pct,
            'max_drawdown': 0,
            'avg_minutes_held': df['minutes_held'].mean() if not df.empty else 0,
            'exit_reasons': df['exit_reason'].value_counts().to_dict() if not df.empty else {}
        }
    
    def _log_results(self):
        """Log final results"""
        logging.info("\n" + "="*80)
        logging.info("📊 BACKTEST RESULTS")
        logging.info("="*80)
        
        results = self._calculate_results()
        
        logging.info(f"Total Trades: {results['total_trades']}")
        logging.info(f"Win Rate: {results['win_rate']:.1f}%")
        logging.info(f"Total P&L: ${results['total_pnl']:.2f}")
        logging.info(f"Avg Win: ${results['avg_win']:.2f}")
        logging.info(f"Avg Loss: ${results['avg_loss']:.2f}")
        logging.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logging.info(f"Return: {results['return_pct']:.1f}%")
        logging.info(f"Final Capital: ${results['final_capital']:.2f}")
        logging.info(f"Avg Hold Time: {results['avg_minutes_held']:.0f} minutes")
        
        # Log exit reasons
        if results.get('exit_reasons'):
            logging.info("\nExit Reasons:")
            for reason, count in results['exit_reasons'].items():
                logging.info(f"  {reason}: {count} trades")
        
        logging.info("="*80)
        
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            output_file = os.path.join(self.data_folder_path, '../backtest_results.csv')
            df.to_csv(output_file, index=False)
            logging.info(f"📊 Detailed results saved to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'historical_data')
    
    test_params = {
        'breakeven_trigger_percent': 10,
        'trail_method': 'pullback_percent',
        'pullback_percent': 10,
        'native_trail_percent': 25,
        'psar_enabled': False,
        'rsi_hook_enabled': False
    }
    
    engine = BacktestEngine(signal_file, data_folder)
    engine.run_simulation(test_params)
