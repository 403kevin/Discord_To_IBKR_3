#!/usr/bin/env python3
"""
backtest_engine.py - FIXED VERSION 
Fixes the phantom trade bug by only processing ticks that occur AFTER signal timestamps
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
    FIXED: Event-driven backtesting engine that properly handles signal timing
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
        
        # FIXED: Track which contracts have active signals
        self.active_contracts = {}  # contract_key -> signal_timestamp
        
        logging.info("🔍 DEBUG: BacktestEngine initialized (FIXED VERSION)")
        logging.info(f"🔍 DEBUG: Signal file: {signal_file_path}")
        logging.info(f"🔍 DEBUG: Data folder: {data_folder_path}")
        
    def run_simulation(self, params=None):
        """Main simulation loop with parameter support"""
        logging.info("\n" + "="*80)
        logging.info("🔍 DEBUG: run_simulation() called (FIXED VERSION)")
        logging.info("="*80)
        
        # Apply parameters if provided
        if params:
            self._apply_parameters(params)
        
        # Load and parse signals
        signals = self._load_signals()
        if not signals:
            logging.error("❌ No signals loaded!")
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'final_capital': self.portfolio['cash']
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
                self._process_signal_event(timestamp, data)
            elif event_type == 'TICK':
                tick_count += 1
                self._process_tick_event(timestamp, data)
            
            if processed_count % 1000 == 0:
                logging.debug(f"🔍 Progress: {processed_count}/{len(event_queue)} events ({signal_count} signals, {tick_count} ticks)")
        
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
                    
                    # Parse the signal
                    profile = {
                        'assume_buy_on_ambiguous': True, 
                        'ambiguous_expiry_enabled': True
                    }
                    
                    parsed = self.signal_parser.parse_signal(signal_text, trader, profile)
                    
                    if parsed:
                        parsed['timestamp'] = timestamp
                        parsed['trader'] = trader
                        parsed['raw_signal'] = signal_text
                        signals.append(parsed)
                        logging.info(f"  Line {line_num}: {parsed['ticker']} {parsed['strike']}{parsed['contract_type'][0]} - {parsed['expiry_date']} @ {timestamp}")
                    else:
                        logging.warning(f"  Line {line_num}: Could not parse: {signal_text}")
                        
                except Exception as e:
                    logging.error(f"  Line {line_num}: Error parsing: {e}")
        
        logging.info(f"🔍 DEBUG: Loaded {len(signals)} valid signals")
        return signals
    
    def _create_event_queue(self, signals):
        """FIXED: Create event queue with proper signal-tick association"""
        logging.info("🔍 DEBUG: _create_event_queue() called (FIXED VERSION)")
        event_queue = []
        
        # Find earliest signal time for filtering
        earliest_signal_time = min(s['timestamp'] for s in signals) if signals else None
        logging.info(f"🔍 DEBUG: Earliest signal time: {earliest_signal_time}")
        
        # Build mapping of contracts to their signal times
        contract_signal_times = {}
        for signal in signals:
            contract_key = (
                signal['ticker'],
                signal['expiry_date'],
                signal['strike'],
                signal['contract_type'][0]
            )
            # Store the signal time for this specific contract
            if contract_key not in contract_signal_times:
                contract_signal_times[contract_key] = []
            contract_signal_times[contract_key].append(signal['timestamp'])
            
            # Add signal event
            event_queue.append((signal['timestamp'], 'SIGNAL', signal))
        
        logging.info(f"🔍 DEBUG: Added {len(signals)} signal events")
        logging.info(f"🔍 DEBUG: Contract->Signal mapping: {len(contract_signal_times)} unique contracts")
        
        # Load tick data ONLY for contracts that have signals
        for (ticker, expiry, strike, ctype), signal_times in contract_signal_times.items():
            # Build filename
            expiry_clean = expiry.replace('-', '')
            filename = get_data_filename_databento(ticker, expiry_clean, strike, ctype.upper())
            filepath = os.path.join(self.data_folder_path, filename)
            
            logging.debug(f"🔍 DEBUG: Looking for data file: {filepath}")
            
            if not os.path.exists(filepath):
                logging.warning(f"  ⚠️ Data file not found: {filename}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                
                if 'timestamp' not in df.columns:
                    logging.error(f"  ❌ No timestamp column in {filename}")
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                # FIXED: Only add ticks that occur AFTER the signal time for THIS contract
                earliest_signal_for_contract = min(signal_times)
                df_filtered = df[df['timestamp'] >= earliest_signal_for_contract]
                
                # Add filtered tick events
                ticks_added = 0
                for _, row in df_filtered.iterrows():
                    event_data = {
                        'ticker': ticker,
                        'expiry': expiry,
                        'strike': strike,
                        'contract_type': ctype,
                        'price': row.get('price', row.get('close', 0)),
                        'bid': row.get('bid', row.get('price', 0)),
                        'ask': row.get('ask', row.get('price', 0)),
                        'volume': row.get('volume', 0)
                    }
                    event_queue.append((row['timestamp'], 'TICK', event_data))
                    ticks_added += 1
                
                logging.info(f"  ✅ Added {ticks_added} tick events (out of {len(df)} total) for {ticker}_{expiry_clean}_{strike}{ctype}")
                logging.info(f"     Filtered to ticks after {earliest_signal_for_contract}")
                
            except Exception as e:
                logging.error(f"  ❌ Error loading {filename}: {e}")
        
        logging.info(f"🔍 DEBUG: Total events in queue: {len(event_queue)}")
        
        # Sanity check
        signal_events = sum(1 for _, t, _ in event_queue if t == 'SIGNAL')
        tick_events = sum(1 for _, t, _ in event_queue if t == 'TICK')
        logging.info(f"🔍 DEBUG: Event breakdown: {signal_events} signals, {tick_events} ticks")
        
        if tick_events == 0:
            logging.warning("⚠️ WARNING: No tick data loaded! Check your historical data files.")
        
        return event_queue
    
    def _process_signal_event(self, timestamp, signal):
        """Process a signal event (entry)"""
        logging.debug(f"🔍 DEBUG: Processing signal at {timestamp}")
        
        # Create position ID
        position_id = f"{signal['ticker']}_{signal['strike']}{signal['contract_type'][0]}_{timestamp.strftime('%H%M%S')}"
        
        # Get entry price from data
        entry_price = self._get_entry_price_from_data(signal)
        
        if entry_price is None or entry_price <= 0:
            logging.warning(f"  ⚠️ Skipping {position_id}: No valid entry price")
            return
        
        # Calculate position size (10% of portfolio)
        position_size = int(self.portfolio['cash'] * 0.10 / (entry_price * 100))
        if position_size <= 0:
            position_size = 1
        
        # Record position
        self.portfolio['positions'][position_id] = {
            'signal': signal,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'quantity': position_size,
            'status': 'OPEN',
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        # Initialize tracking
        self.trailing_highs_and_lows[position_id] = {
            'high': entry_price,
            'low': entry_price
        }
        self.breakeven_activated[position_id] = False
        
        # Mark this contract as active
        contract_key = (
            signal['ticker'],
            signal['expiry_date'], 
            signal['strike'],
            signal['contract_type'][0]
        )
        self.active_contracts[contract_key] = timestamp
        
        logging.info(f"  📈 ENTRY: {position_id} @ ${entry_price:.2f} x {position_size}")
    
    def _process_tick_event(self, timestamp, tick_data):
        """Process a tick event (check for exits)"""
        # Only process ticks for contracts we have positions in
        contract_key = (
            tick_data['ticker'],
            tick_data['expiry'],
            tick_data['strike'],
            tick_data['contract_type']
        )
        
        # Skip if this contract doesn't have an active signal
        if contract_key not in self.active_contracts:
            return
        
        # Find matching open positions
        for position_id, position in list(self.portfolio['positions'].items()):
            if position['status'] != 'OPEN':
                continue
            
            signal = position['signal']
            
            # Check if this tick matches the position
            if (tick_data['ticker'] == signal['ticker'] and
                tick_data['strike'] == signal['strike'] and
                tick_data['contract_type'] == signal['contract_type'][0] and
                tick_data['expiry'] == signal['expiry_date']):
                
                current_price = tick_data['price']
                
                # Update high/low tracking
                if position_id in self.trailing_highs_and_lows:
                    if signal['contract_type'] == 'CALL':
                        self.trailing_highs_and_lows[position_id]['high'] = max(
                            self.trailing_highs_and_lows[position_id]['high'], 
                            current_price
                        )
                    else:  # PUT
                        self.trailing_highs_and_lows[position_id]['low'] = min(
                            self.trailing_highs_and_lows[position_id]['low'], 
                            current_price
                        )
                
                # Check exit conditions
                exit_price, exit_reason = self._check_exit_conditions(
                    position_id, position, current_price, timestamp
                )
                
                if exit_price:
                    self._close_position(position_id, exit_price, exit_reason, timestamp)
                    
                    # Remove from active contracts if no more positions
                    has_other_positions = any(
                        p['signal']['ticker'] == signal['ticker'] and
                        p['signal']['strike'] == signal['strike'] and
                        p['signal']['contract_type'] == signal['contract_type'] and
                        p['signal']['expiry_date'] == signal['expiry_date'] and
                        p['status'] == 'OPEN'
                        for pid, p in self.portfolio['positions'].items()
                        if pid != position_id
                    )
                    if not has_other_positions:
                        if contract_key in self.active_contracts:
                            del self.active_contracts[contract_key]
    
    def _check_exit_conditions(self, position_id, position, current_price, timestamp):
        """Check if position should exit"""
        signal = position['signal']
        entry_price = position['entry_price']
        
        # Get exit strategy from config
        profile = self.config.profiles[0] if self.config.profiles else {}
        exit_strategy = profile.get('exit_strategy', {})
        
        # Calculate P&L percentage
        if signal['contract_type'] == 'CALL':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # PUT
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check breakeven activation
        breakeven_trigger = exit_strategy.get('breakeven_trigger_percent', 0.10) * 100
        if pnl_pct >= breakeven_trigger and not self.breakeven_activated.get(position_id, False):
            self.breakeven_activated[position_id] = True
            logging.debug(f"  🎯 Breakeven activated for {position_id} at {pnl_pct:.1f}%")
        
        # Check breakeven stop
        if self.breakeven_activated.get(position_id, False):
            if pnl_pct <= 0:
                return current_price, "BREAKEVEN_STOP"
        
        # Check trailing stop based on method
        trail_method = exit_strategy.get('trail_method', 'pullback_percent')
        
        if trail_method == 'pullback_percent':
            pullback_pct = exit_strategy.get('trail_settings', {}).get('pullback_percent', 0.10) * 100
            
            if signal['contract_type'] == 'CALL':
                highest = self.trailing_highs_and_lows[position_id]['high']
                pullback_from_high = ((highest - current_price) / highest) * 100
                
                if pullback_from_high >= pullback_pct and pnl_pct > 0:
                    return current_price, f"PULLBACK_{pullback_pct:.0f}%"
            else:  # PUT
                lowest = self.trailing_highs_and_lows[position_id]['low']
                pullback_from_low = ((current_price - lowest) / lowest) * 100
                
                if pullback_from_low >= pullback_pct and pnl_pct > 0:
                    return current_price, f"PULLBACK_{pullback_pct:.0f}%"
        
        # Check time-based exit (close before market close)
        market_close = timestamp.replace(hour=15, minute=50, second=0)
        if timestamp >= market_close:
            return current_price, "EOD_CLOSE"
        
        return None, None
    
    def _close_position(self, position_id, exit_price, exit_reason, timestamp):
        """Close a position and record the trade"""
        position = self.portfolio['positions'][position_id]
        signal = position['signal']
        
        # Calculate P&L
        if signal['contract_type'] == 'CALL':
            pnl = (exit_price - position['entry_price']) * position['quantity'] * 100
        else:  # PUT
            pnl = (position['entry_price'] - exit_price) * position['quantity'] * 100
        
        pnl_pct = (pnl / (position['entry_price'] * position['quantity'] * 100)) * 100
        
        # Record trade
        trade = {
            'position_id': position_id,
            'ticker': signal['ticker'],
            'strike': signal['strike'],
            'contract_type': signal['contract_type'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'minutes_held': (timestamp - position['entry_time']).total_seconds() / 60
        }
        
        self.trade_log.append(trade)
        self.portfolio['cash'] += pnl
        position['status'] = 'CLOSED'
        
        logging.info(f"  📉 EXIT: {position_id} @ ${exit_price:.2f} | "
                    f"P&L: ${pnl:.0f} ({pnl_pct:.1f}%) | Reason: {exit_reason}")
    
    def _get_entry_price_from_data(self, signal):
        """Get entry price from historical data at signal time"""
        # Build filename
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
                'final_capital': self.portfolio['cash']
            }
        
        df = pd.DataFrame(self.trade_log)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        total_pnl = df['pnl'].sum()
        win_rate = (len(wins) / len(df)) * 100 if len(df) > 0 else 0
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if len(losses) > 0 and avg_loss > 0 else float('inf')
        
        return {
            'total_trades': len(df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.portfolio['cash'],
            'return_pct': ((self.portfolio['cash'] - self.starting_capital) / self.starting_capital) * 100,
            'avg_minutes_held': df['minutes_held'].mean() if not df.empty else 0
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
        logging.info("="*80)
        
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            output_file = os.path.join(self.data_folder_path, '../backtest_results.csv')
            df.to_csv(output_file, index=False)
            logging.info(f"📊 Detailed results saved to {output_file}")
        
        logging.info("🔍 DEBUG: _log_results() completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("🔍 DEBUG: Script started (FIXED VERSION)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_file = os.path.join(script_dir, 'signals_to_test.txt')
    data_folder = os.path.join(script_dir, 'historical_data')
    
    logging.info(f"🔍 DEBUG: Creating BacktestEngine...")
    engine = BacktestEngine(signal_file, data_folder)
    
    logging.info(f"🔍 DEBUG: Calling run_simulation()...")
    engine.run_simulation()
    
    logging.info("🔍 DEBUG: Script completed")
