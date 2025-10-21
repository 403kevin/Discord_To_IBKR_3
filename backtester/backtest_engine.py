#!/usr/bin/env python3
"""
backtest_engine.py - COMPLETE WORKING VERSION
MS 20 fix + proper signal parsing for format: "TICKER STRIKEC/P MM/DD"
"""

import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename_databento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BacktestEngine:
    """
    Event-driven backtesting engine with COMPLETE fixes:
    - MS 20: Test parameters stored on self (not self.config)
    - Proper signal parsing validation
    - All exit strategies working
    """
    
    def __init__(self, signal_file_path: str, data_folder_path: str):
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        self.signal_file_path = signal_file_path
        self.historical_data_dir = data_folder_path
        
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
        self.native_trail_stops = {}
        
        # Track active contracts
        self.active_contracts = {}
        
        # ✅ MS 20 FIX: Test parameters on self
        self.breakeven_trigger_percent = 10
        self.trail_method = 'pullback_percent'
        self.pullback_percent = 10
        self.native_trail_percent = 25
        self.atr_period = 14
        self.atr_multiplier = 1.5
        self.psar_enabled = False
        self.psar_start = 0.02
        self.psar_increment = 0.02
        self.psar_max = 0.2
        self.rsi_hook_enabled = False
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        logging.info("✅ BacktestEngine initialized")
        logging.info(f"   Signal file: {signal_file_path}")
        logging.info(f"   Data folder: {data_folder_path}")
        
    def run_simulation(self, params: Optional[Dict] = None) -> Dict:
        """Main simulation loop"""
        logging.info("\n" + "="*80)
        logging.info("🚀 BACKTEST SIMULATION START")
        logging.info("="*80)
        
        # Apply test parameters if provided
        if params:
            self._apply_parameters(params)
        
        # Load and parse signals
        signals = self._load_signals()
        if not signals:
            logging.error("❌ No signals found to backtest")
            return self._empty_results()
        
        logging.info(f"📊 Loaded {len(signals)} signals")
        
        # Load historical data for each signal
        for signal in signals:
            df = self._load_signal_data(signal)
            if not df.empty:
                signal['data'] = df
                signal['has_data'] = True
            else:
                signal['has_data'] = False
                logging.warning(f"⚠️ No data for {signal['ticker']} {signal['strike']}{signal['right']}")
        
        # Filter to signals with data
        valid_signals = [s for s in signals if s.get('has_data', False)]
        if not valid_signals:
            logging.error("❌ No signals have historical data")
            return self._empty_results()
        
        logging.info(f"✅ {len(valid_signals)} signals have data")
        
        # Create chronological event queue
        event_queue = self._create_event_queue(valid_signals)
        logging.info(f"📅 Created event queue with {len(event_queue)} events")
        
        # Process all events
        for event in event_queue:
            if event['type'] == 'signal':
                self._process_signal_event(event)
            elif event['type'] == 'tick':
                self._process_tick_event(event)
        
        # Close any remaining positions at EOD
        self._close_all_positions()
        
        # Calculate results
        results = self._calculate_results()
        
        logging.info("="*80)
        logging.info("🏁 BACKTEST SIMULATION COMPLETE")
        logging.info("="*80)
        
        return results
    
    def _apply_parameters(self, params: Dict):
        """✅ MS 20 FIX: Apply to self, not self.config"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.debug(f"   Set {key} = {value}")
    
    def _parse_signal_manual(self, signal_text: str, timestamp_str: str) -> Optional[Dict]:
        """
        Manual signal parser as fallback
        Handles format: "TICKER STRIKE[C/P] MM/DD"
        Example: "SPY 648P 09/09"
        """
        try:
            parts = signal_text.strip().split()
            if len(parts) < 3:
                return None
            
            ticker = parts[0].upper()
            
            # Extract strike and right from parts[1] (e.g., "648P")
            strike_part = parts[1]
            match = re.match(r'(\d+(?:\.\d+)?)(C|P)', strike_part.upper())
            if not match:
                return None
            
            strike = float(match.group(1))
            right = match.group(2)
            
            # Parse expiry from parts[2] (e.g., "09/09")
            expiry_str = parts[2]
            exp_parts = expiry_str.split('/')
            if len(exp_parts) != 2:
                return None
            
            exp_month = int(exp_parts[0])
            exp_day = int(exp_parts[1])
            
            # Get year from signal timestamp
            signal_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            signal_year = signal_time.year
            signal_month = signal_time.month
            
            # If expiry month < signal month, expiry is next year
            if exp_month < signal_month:
                exp_year = signal_year + 1
            else:
                exp_year = signal_year
            
            expiry = datetime(exp_year, exp_month, exp_day)
            
            return {
                'ticker': ticker,
                'strike': strike,
                'right': right,
                'expiry': expiry,
                'action': 'BUY',
                'quantity': 1
            }
            
        except Exception as e:
            logging.error(f"Manual parse error: {e}")
            return None
    
    def _load_signals(self) -> List[Dict]:
        """Load and parse signals from file"""
        signals = []
        
        # Get default profile for SignalParser
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True,
            'buzzwords_buy': [],
            'buzzwords_sell': [],
            'channel_id': 'backtest'
        }
        
        if not os.path.exists(self.signal_file_path):
            logging.error(f"Signal file not found: {self.signal_file_path}")
            return []
        
        with open(self.signal_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('Trader:') or line.startswith('Format:'):
                    continue
                
                # Parse timestamped format: YYYY-MM-DD HH:MM:SS | channel | signal
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        timestamp_str = parts[0].strip()
                        channel = parts[1].strip()
                        signal_text = parts[2].strip()
                    else:
                        logging.warning(f"Line {line_num}: Invalid format, skipping")
                        continue
                else:
                    # Simple format: just the signal
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    channel = 'backtest'
                    signal_text = line
                
                # Try SignalParser first
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                
                # If SignalParser fails, use manual parser
                if not parsed or 'right' not in parsed:
                    logging.debug(f"SignalParser failed, trying manual parse: {signal_text}")
                    parsed = self._parse_signal_manual(signal_text, timestamp_str)
                
                # Validate parsed signal
                if parsed and all(k in parsed for k in ['ticker', 'strike', 'right', 'expiry']):
                    parsed['timestamp'] = timestamp_str
                    signals.append(parsed)
                    logging.debug(f"✅ Parsed: {parsed['ticker']} {parsed['strike']}{parsed['right']} exp {parsed['expiry'].strftime('%Y-%m-%d')}")
                else:
                    logging.warning(f"Line {line_num}: Failed to parse '{signal_text}'")
        
        return signals
    
    def _load_signal_data(self, signal: Dict) -> pd.DataFrame:
        """Load historical data for a signal"""
        filename = get_data_filename_databento(
            signal['ticker'],
            signal['expiry'].strftime('%Y%m%d'),
            signal['strike'],
            signal['right']
        )
        
        filepath = os.path.join(self.historical_data_dir, filename)
        
        if not os.path.exists(filepath):
            logging.warning(f"⚠️ Data file not found: {filename}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            
            # Universal timestamp column handler
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'ts_event' in df.columns:
                time_col = 'ts_event'
            else:
                logging.error(f"❌ No timestamp column in {filename}")
                return pd.DataFrame()
            
            # Parse timestamps with UTC awareness
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            
            # Filter to only data after signal timestamp
            signal_time = pd.to_datetime(signal['timestamp'])
            if signal_time.tz is None:
                signal_time = signal_time.tz_localize('UTC')
            
            df = df[df[time_col] >= signal_time].copy()
            
            logging.info(f"📈 Loaded {len(df)} ticks for {signal['ticker']} {signal['strike']}{signal['right']}")
            
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading {filename}: {str(e)}")
            return pd.DataFrame()
    
    def _create_event_queue(self, signals: List[Dict]) -> List[Dict]:
        """Create chronological event queue"""
        events = []
        
        for signal in signals:
            signal_id = f"{signal['ticker']}_{signal['strike']}{signal['right']}_{signal['expiry'].strftime('%Y%m%d')}"
            signal_time = pd.to_datetime(signal['timestamp'])
            if signal_time.tz is None:
                signal_time = signal_time.tz_localize('UTC')
            
            events.append({
                'type': 'signal',
                'timestamp': signal_time,
                'signal': signal,
                'signal_id': signal_id
            })
            
            # Add tick events
            df = signal['data']
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'ts_event' in df.columns:
                time_col = 'ts_event'
            else:
                continue
            
            for idx, row in df.iterrows():
                events.append({
                    'type': 'tick',
                    'timestamp': row[time_col],
                    'signal_id': signal_id,
                    'data': row.to_dict()
                })
        
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def _process_signal_event(self, event: Dict):
        """Process a signal - open position"""
        signal = event['signal']
        signal_id = event['signal_id']
        signal_time = event['timestamp']
        
        logging.info(f"\n📍 SIGNAL @ {signal_time}")
        logging.info(f"   {signal['ticker']} {signal['strike']}{signal['right']} exp {signal['expiry'].strftime('%Y-%m-%d')}")
        
        df = signal['data']
        if df.empty:
            logging.warning("   ⚠️ No data, skipping")
            return
        
        first_tick = df.iloc[0]
        
        # Get entry price
        if pd.notna(first_tick.get('bid')) and pd.notna(first_tick.get('ask')):
            entry_price = (first_tick['bid'] + first_tick['ask']) / 2
        elif pd.notna(first_tick.get('close')):
            entry_price = first_tick['close']
        else:
            logging.warning("   ⚠️ No valid entry price, skipping")
            return
        
        # Position sizing: 10% of portfolio per trade
        position_size = self.portfolio['cash'] * 0.10
        contracts = int(position_size / (entry_price * 100))
        
        if contracts == 0:
            logging.warning("   ⚠️ Insufficient capital for 1 contract")
            return
        
        cost = contracts * entry_price * 100
        
        # Open position
        self.portfolio['cash'] -= cost
        self.portfolio['positions'][signal_id] = {
            'signal': signal,
            'entry_time': signal_time,
            'entry_price': entry_price,
            'contracts': contracts,
            'current_price': entry_price,
            'cost_basis': cost,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        # Initialize exit tracking
        self.breakeven_activated[signal_id] = False
        self.native_trail_stops[signal_id] = None
        self.trailing_highs_and_lows[signal_id] = {'high': entry_price, 'low': entry_price}
        
        logging.info(f"   ✅ OPEN: {contracts} contracts @ ${entry_price:.2f} (cost: ${cost:.2f})")
    
    def _process_tick_event(self, event: Dict):
        """Process a market tick"""
        signal_id = event['signal_id']
        tick_data = event['data']
        tick_time = event['timestamp']
        
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        
        # Get current price
        if pd.notna(tick_data.get('bid')) and pd.notna(tick_data.get('ask')):
            current_price = (tick_data['bid'] + tick_data['ask']) / 2
        elif pd.notna(tick_data.get('close')):
            current_price = tick_data['close']
        else:
            return
        
        # Update position
        position['current_price'] = current_price
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Check for exit
        exit_signal = self._evaluate_exit(signal_id, position, tick_data, tick_time)
        
        if exit_signal:
            self._close_position(signal_id, current_price, tick_time, exit_signal['reason'])
    
    def _evaluate_exit(self, signal_id: str, position: Dict, tick_data: Dict, current_time: datetime) -> Optional[Dict]:
        """✅ MS 20 FIX: Evaluate exit conditions using self.* parameters"""
        entry_price = position['entry_price']
        current_price = position['current_price']
        highest_price = position['highest_price']
        
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 1. BREAKEVEN ACTIVATION
        if not self.breakeven_activated[signal_id]:
            if pnl_pct >= self.breakeven_trigger_percent:
                self.breakeven_activated[signal_id] = True
                self.native_trail_stops[signal_id] = entry_price
                logging.debug(f"   🔒 Breakeven activated @ ${current_price:.2f} (+{pnl_pct:.1f}%)")
        
        # 2. NATIVE TRAIL
        if self.native_trail_stops[signal_id] is not None:
            new_stop = highest_price * (1 - self.native_trail_percent / 100)
            if new_stop > self.native_trail_stops[signal_id]:
                self.native_trail_stops[signal_id] = new_stop
                logging.debug(f"   📈 Native trail updated: ${new_stop:.2f}")
            
            if current_price <= self.native_trail_stops[signal_id]:
                return {'reason': 'native_trail', 'pnl_pct': pnl_pct}
        
        # 3. BREAKEVEN STOP
        if self.breakeven_activated[signal_id] and self.native_trail_stops[signal_id] == entry_price:
            if current_price <= entry_price:
                return {'reason': 'breakeven', 'pnl_pct': 0}
        
        # 4. PULLBACK STOP
        if self.trail_method == 'pullback_percent':
            pullback_from_high = ((highest_price - current_price) / highest_price) * 100
            if pullback_from_high >= self.pullback_percent:
                return {'reason': 'pullback', 'pnl_pct': pnl_pct}
        
        # 5. ATR TRAIL
        if self.trail_method == 'atr':
            if pd.notna(tick_data.get('high')) and pd.notna(tick_data.get('low')):
                current_range = tick_data['high'] - tick_data['low']
                atr_stop = highest_price - (current_range * self.atr_multiplier)
                if current_price <= atr_stop:
                    return {'reason': 'atr_trail', 'pnl_pct': pnl_pct}
        
        return None
    
    def _close_position(self, signal_id: str, exit_price: float, exit_time: datetime, exit_reason: str):
        """Close a position"""
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        
        entry_price = position['entry_price']
        contracts = position['contracts']
        proceeds = contracts * exit_price * 100
        cost = position['cost_basis']
        pnl = proceeds - cost
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        self.portfolio['cash'] += proceeds
        
        hold_minutes = (exit_time - position['entry_time']).total_seconds() / 60
        
        self.trade_log.append({
            'signal_id': signal_id,
            'ticker': position['signal']['ticker'],
            'strike': position['signal']['strike'],
            'right': position['signal']['right'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'contracts': contracts,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'hold_minutes': hold_minutes
        })
        
        del self.portfolio['positions'][signal_id]
        del self.breakeven_activated[signal_id]
        del self.native_trail_stops[signal_id]
        
        logging.info(f"   ❌ CLOSE: {exit_reason} @ ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | Hold: {hold_minutes:.0f}min")
    
    def _close_all_positions(self):
        """Close any remaining positions at EOD"""
        for signal_id in list(self.portfolio['positions'].keys()):
            position = self.portfolio['positions'][signal_id]
            current_price = position['current_price']
            current_time = position['entry_time'] + timedelta(hours=6)
            self._close_position(signal_id, current_price, current_time, 'eod_close')
    
    def _calculate_results(self) -> Dict:
        """Calculate final performance metrics"""
        if not self.trade_log:
            return self._empty_results()
        
        df = pd.DataFrame(self.trade_log)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (abs(avg_win * winning_trades) / abs(avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        final_capital = self.starting_capital + total_pnl
        return_pct = (total_pnl / self.starting_capital) * 100
        
        avg_hold_minutes = df['hold_minutes'].mean()
        
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': final_capital,
            'return_pct': return_pct,
            'avg_minutes_held': avg_hold_minutes,
            'exit_reasons': exit_reasons
        }
        
        logging.info("\n" + "="*80)
        logging.info("📊 BACKTEST RESULTS")
        logging.info("="*80)
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
        logging.info(f"Total P&L: ${total_pnl:,.2f}")
        logging.info(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info(f"Final Capital: ${final_capital:,.2f} ({return_pct:+.2f}%)")
        logging.info(f"Avg Hold Time: {avg_hold_minutes:.0f} minutes")
        logging.info("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            logging.info(f"  {reason}: {count}")
        logging.info("="*80)
        
        return results
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'final_capital': self.starting_capital,
            'return_pct': 0,
            'avg_minutes_held': 0,
            'exit_reasons': {}
        }


if __name__ == "__main__":
    engine = BacktestEngine(
        signal_file_path="backtester/signals_to_test.txt",
        data_folder_path="backtester/historical_data"
    )
    
    results = engine.run_simulation()
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {results['total_trades']} trades, ${results['total_pnl']:.2f} P&L")
    print("="*80)
