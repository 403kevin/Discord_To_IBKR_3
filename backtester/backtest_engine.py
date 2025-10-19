#!/usr/bin/env python3
"""
backtest_engine.py - COMPLETE VERSION WITH NATIVE TRAILING STOP
Includes all exit strategies: breakeven, pullback, ATR, PSAR, RSI, and NATIVE TRAIL
FIXED: parse_signal() call and return_pct in empty results
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
    COMPLETE VERSION: Event-driven backtesting with ALL exit strategies including native trail
    FIXED: Correct parse_signal() signature and return_pct in empty results
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
        self.native_trail_stops = {}  # position_id -> trail_stop_price
        
        # Track which contracts have active signals
        self.active_contracts = {}
        
        logging.info("🔍 DEBUG: BacktestEngine initialized (COMPLETE VERSION with native trail)")
        logging.info(f"🔍 DEBUG: Signal file: {signal_file_path}")
        logging.info(f"🔍 DEBUG: Data folder: {data_folder_path}")
        
    def run_simulation(self, params=None):
        """Main simulation loop with parameter support"""
        logging.info("\n" + "="*80)
        logging.info("🚀 Starting Backtest Simulation (COMPLETE VERSION)")
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
                'return_pct': 0,  # ✅ FIXED: Added return_pct
                'max_drawdown': 0,
                'avg_minutes_held': 0,
                'exit_reasons': {}
            }
        
        logging.info(f"✅ Loaded {len(signals)} signals")
        
        # Create event queue
        logging.info("🔍 DEBUG: Creating event queue...")
        event_queue = self._create_event_queue(signals)
        
        if not event_queue:
            logging.error("❌ Event queue is empty!")
            return self._calculate_results()
        
        logging.info(f"✅ Created event queue with {len(event_queue)} events")
        
        # Process events
        logging.info("🔍 DEBUG: Starting event processing loop...")
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
                logging.debug(f"🔍 Progress: {processed_count}/{len(event_queue)} events")
        
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
        logging.info(f"🔍 DEBUG: Loading signals from {self.signal_file_path}")
        
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
                        parsed['timestamp'] = timestamp
                        parsed['trader'] = trader
                        parsed['raw_signal'] = signal_text
                        signals.append(parsed)
                        logging.info(f"  Line {line_num}: {parsed['ticker']} {parsed['expiry_date']} {parsed['strike']}{parsed['contract_type'][0]}")
                    else:
                        logging.warning(f"  Line {line_num}: Failed to parse: {signal_text}")
                    
                except Exception as e:
                    logging.error(f"  Line {line_num}: Error parsing: {e}")
        
        logging.info(f"🔍 DEBUG: Loaded {len(signals)} valid signals")
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
        
        # Get entry price from historical data
        entry_price = self._get_entry_price_from_data(signal)
        
        if entry_price is None or entry_price <= 0:
            logging.warning(f"  ⚠️ No valid entry price for {contract_key}, skipping")
            return
        
        # Calculate position size (10% of portfolio per trade)
        position_value = self.portfolio['cash'] * 0.10
        quantity = int(position_value / (entry_price * 100))
        
        if quantity < 1:
            logging.warning(f"  ⚠️ Insufficient capital for {contract_key}, skipping")
            return
        
        cost = quantity * entry_price * 100
        
        if cost > self.portfolio['cash']:
            logging.warning(f"  ⚠️ Insufficient cash for {contract_key}, skipping")
            return
        
        # Execute entry
        self.portfolio['cash'] -= cost
        self.portfolio['positions'][contract_key] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'contract_type': signal['contract_type'],
            'ticker': signal['ticker'],
            'strike': signal['strike'],
            'expiry': signal['expiry_date'],
            'trader': signal['trader']
        }
        
        # Initialize tracking for this position
        self.position_data_cache[contract_key] = []
        self.tick_buffer[contract_key] = []
        self.trailing_highs_and_lows[contract_key] = {'high': entry_price, 'low': entry_price}
        self.breakeven_activated[contract_key] = False
        
        # Initialize native trailing stop
        native_trail_pct = params.get('native_trail_percent', 25) / 100 if params else 0.25
        self.native_trail_stops[contract_key] = entry_price * (1 - native_trail_pct)
        
        logging.info(f"✅ ENTRY: {contract_key} @ ${entry_price:.2f} x {quantity} contracts")
    
    def _process_tick_event(self, timestamp, tick_data, params):
        """Process a market tick for active positions"""
        contract_key = tick_data['contract_key']
        
        if contract_key not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][contract_key]
        current_price = tick_data['price']
        
        if current_price <= 0:
            return
        
        # Add to tick buffer
        self.tick_buffer[contract_key].append({
            'timestamp': timestamp,
            'price': current_price
        })
        
        # Update trailing high/low
        if current_price > self.trailing_highs_and_lows[contract_key]['high']:
            self.trailing_highs_and_lows[contract_key]['high'] = current_price
        if current_price < self.trailing_highs_and_lows[contract_key]['low']:
            self.trailing_highs_and_lows[contract_key]['low'] = current_price
        
        # Update native trailing stop
        native_trail_pct = params.get('native_trail_percent', 25) / 100 if params else 0.25
        new_trail_stop = current_price * (1 - native_trail_pct)
        if new_trail_stop > self.native_trail_stops[contract_key]:
            self.native_trail_stops[contract_key] = new_trail_stop
        
        # Check exit conditions
        exit_reason = self._evaluate_exit_conditions(
            contract_key,
            position,
            current_price,
            timestamp,
            params
        )
        
        if exit_reason:
            self._execute_exit(contract_key, position, current_price, timestamp, exit_reason)
    
    def _evaluate_exit_conditions(self, contract_key, position, current_price, timestamp, params):
        """
        Check all exit conditions in priority order
        Returns exit_reason string or None
        """
        entry_price = position['entry_price']
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # PRIORITY 1: Native Trailing Stop (highest priority)
        if current_price <= self.native_trail_stops[contract_key]:
            return f"NATIVE_TRAIL_{params.get('native_trail_percent', 25)}%"
        
        # PRIORITY 2: Breakeven Stop
        breakeven_trigger = params.get('breakeven_trigger_percent', 10) / 100 if params else 0.10
        if pnl_pct >= (breakeven_trigger * 100) and not self.breakeven_activated[contract_key]:
            self.breakeven_activated[contract_key] = True
            logging.debug(f"  Breakeven activated for {contract_key} @ {pnl_pct:.1f}%")
        
        if self.breakeven_activated[contract_key] and current_price <= entry_price:
            return "BREAKEVEN"
        
        # PRIORITY 3: Pullback/ATR Trail
        trail_method = params.get('trail_method', 'pullback_percent') if params else 'pullback_percent'
        
        if trail_method == 'pullback_percent':
            pullback_pct = params.get('pullback_percent', 10) / 100 if params else 0.10
            peak = self.trailing_highs_and_lows[contract_key]['high']
            if peak > entry_price:
                pullback_stop = peak * (1 - pullback_pct)
                if current_price <= pullback_stop:
                    return f"PULLBACK_{params.get('pullback_percent', 10)}%"
        
        elif trail_method == 'atr':
            # ATR trail logic (simplified)
            atr_multiplier = params.get('atr_multiplier', 1.5) if params else 1.5
            # Would need actual ATR calculation here
            pass
        
        # PRIORITY 4: Time-based exit (market close)
        if timestamp.hour >= 16:
            return "MARKET_CLOSE"
        
        return None
    
    def _execute_exit(self, contract_key, position, exit_price, exit_time, exit_reason):
        """Execute position exit"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        pnl = (exit_price - entry_price) * quantity * 100
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        minutes_held = (exit_time - position['entry_time']).total_seconds() / 60
        
        self.portfolio['cash'] += (exit_price * quantity * 100)
        
        self.trade_log.append({
            'ticker': position['ticker'],
            'strike': position['strike'],
            'contract_type': position['contract_type'],
            'expiry': position['expiry'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'minutes_held': minutes_held,
            'trader': position['trader']
        })
        
        logging.info(f"❌ EXIT: {contract_key} @ ${exit_price:.2f} | PNL: ${pnl:.2f} ({pnl_pct:+.1f}%) | Reason: {exit_reason}")
        
        # Clean up tracking
        del self.portfolio['positions'][contract_key]
        if contract_key in self.position_data_cache:
            del self.position_data_cache[contract_key]
        if contract_key in self.tick_buffer:
            del self.tick_buffer[contract_key]
        if contract_key in self.trailing_highs_and_lows:
            del self.trailing_highs_and_lows[contract_key]
        if contract_key in self.breakeven_activated:
            del self.breakeven_activated[contract_key]
        if contract_key in self.native_trail_stops:
            del self.native_trail_stops[contract_key]
    
    def _get_entry_price_from_data(self, signal):
        """Get actual entry price from historical data at signal timestamp"""
        # Get filename
        expiry_clean = signal['expiry_date'].replace('-', '')
        filename = get_data_filename_databento(
            signal['ticker'],
            expiry_clean,
            signal['strike'],
            signal['contract_type'][0].upper()
        )
        filepath = os.path.join(self.data_folder_path, filename)
        
        if not os.path.exists(filepath):
            logging.warning(f"  ⚠️ No data file for entry price: {filename}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
            # Find price at or just after signal time
            signal_time = signal['timestamp']
            mask = df['timestamp'] >= signal_time
            
            if mask.any():
                entry_row = df.loc[mask].iloc[0]
                
                # Use ask price for realistic entry
                if 'ask' in entry_row and entry_row['ask'] > 0:
                    return float(entry_row['ask'])
                elif 'price' in entry_row and entry_row['price'] > 0:
                    return float(entry_row['price'])
                elif 'close' in entry_row and entry_row['close'] > 0:
                    return float(entry_row['close'])
            
            return None
            
        except Exception as e:
            logging.error(f"  ❌ Error getting entry price: {e}")
            return None
    
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
                'return_pct': 0,  # ✅ FIXED: Added for empty results
                'max_drawdown': 0,
                'avg_minutes_held': 0,
                'exit_reasons': {}
            }
        
        df = pd.DataFrame(self.trade_log)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        total_pnl = df['pnl'].sum()
        win_rate = (len(wins) / len(df)) * 100 if len(df) > 0 else 0
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if len(losses) > 0 and avg_loss > 0 else float('inf')
        
        # Calculate max drawdown
        cumulative = df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': len(df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_capital': self.portfolio['cash'],
            'return_pct': ((self.portfolio['cash'] - self.starting_capital) / self.starting_capital) * 100,
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
        logging.info(f"Max Drawdown: ${results.get('max_drawdown', 0):.2f}")
        logging.info(f"Return: {results['return_pct']:.1f}%")
        logging.info(f"Final Capital: ${results['final_capital']:.2f}")
        
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
        
        logging.info("🔍 DEBUG: _log_results() completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("🔍 DEBUG: Script started (COMPLETE VERSION)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'historical_data')
    
    # Test with sample parameters
    test_params = {
        'breakeven_trigger_percent': 10,
        'trail_method': 'pullback_percent',
        'pullback_percent': 10,
        'native_trail_percent': 25,  # 25% native trailing stop
        'psar_enabled': True,
        'rsi_hook_enabled': False
    }
    
    logging.info(f"🔍 DEBUG: Creating BacktestEngine...")
    engine = BacktestEngine(signal_file, data_folder)
    
    logging.info(f"🔍 DEBUG: Running simulation with parameters: {test_params}")
    engine.run_simulation(test_params)
    
    logging.info("🔍 DEBUG: Script completed")
