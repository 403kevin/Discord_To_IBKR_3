#!/usr/bin/env python3
"""
backtest_engine.py - COMPLETE WORKING VERSION (MS 21 REPAIR)

THE ACTUAL FIX (from screenshot):
- SignalParser returns: contract_type, expiry_date
- BacktestEngine expects: right, expiry
- Solution: Map the keys after parsing
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
    """Event-driven backtesting engine with key mapping fix"""
    
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
        
        # Position tracking
        self.position_data_cache = {}
        self.tick_buffer = {}
        self.last_bar_timestamp = {}
        self.trailing_highs_and_lows = {}
        self.atr_stop_prices = {}
        self.breakeven_activated = {}
        self.native_trail_stops = {}
        self.active_contracts = {}
        
        # Test parameters (stored on self, not self.config)
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
    
    def run_simulation(self, params: Optional[Dict] = None) -> Dict:
        """Main simulation loop"""
        logging.info("\n" + "="*80)
        logging.info("🚀 BACKTEST SIMULATION START")
        logging.info("="*80)
        
        if params:
            self._apply_parameters(params)
        
        signals = self._load_signals()
        if not signals:
            logging.error("❌ No signals found")
            return self._empty_results()
        
        logging.info(f"📊 Loaded {len(signals)} signals")
        
        # Load historical data
        for signal in signals:
            df = self._load_signal_data(signal)
            if not df.empty:
                signal['data'] = df
                signal['has_data'] = True
            else:
                signal['has_data'] = False
        
        valid_signals = [s for s in signals if s.get('has_data', False)]
        if not valid_signals:
            logging.error("❌ No signals have historical data")
            return self._empty_results()
        
        logging.info(f"✅ {len(valid_signals)} signals have data")
        
        # Create event queue and process
        event_queue = self._create_event_queue(valid_signals)
        logging.info(f"📅 Created {len(event_queue)} events")
        
        for event in event_queue:
            if event['type'] == 'signal':
                self._process_signal_event(event)
            elif event['type'] == 'tick':
                self._process_tick_event(event)
        
        self._close_all_positions()
        results = self._calculate_results()
        
        logging.info("="*80)
        logging.info("🏁 SIMULATION COMPLETE")
        logging.info("="*80)
        
        return results
    
    def _apply_parameters(self, params: Dict):
        """Apply test parameters to self"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _load_signals(self) -> List[Dict]:
        """Load and parse signals with KEY MAPPING FIX"""
        signals = []
        
        # Default profile for parsing
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
                
                if not line or line.startswith('#') or line.startswith('Trader:') or line.startswith('Format:'):
                    continue
                
                # Parse timestamped format: YYYY-MM-DD HH:MM:SS | channel | signal
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        timestamp_str = parts[0].strip()
                        signal_text = parts[2].strip()
                    else:
                        continue
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    signal_text = line
                
                # Parse with SignalParser
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                
                if parsed:
                    # ✅ THE FIX - MAP KEYS
                    parsed['right'] = 'C' if parsed['contract_type'] == 'CALL' else 'P'
                    parsed['expiry'] = datetime.strptime(parsed['expiry_date'], '%Y%m%d')
                    parsed['timestamp'] = timestamp_str
                    
                    signals.append(parsed)
                    logging.debug(f"✅ Line {line_num}: {parsed['ticker']} {parsed['strike']}{parsed['right']}")
                else:
                    logging.warning(f"❌ Line {line_num}: Failed to parse '{signal_text}'")
        
        return signals
    
    def _load_signal_data(self, signal: Dict) -> pd.DataFrame:
        """Load historical CSV data"""
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
            
            # Handle timestamp column
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'ts_event' in df.columns:
                time_col = 'ts_event'
            else:
                logging.error(f"❌ No timestamp column in {filename}")
                return pd.DataFrame()
            
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            
            signal_time = pd.to_datetime(signal['timestamp'])
            if signal_time.tz is None:
                signal_time = signal_time.tz_localize('UTC')
            
            df = df[df[time_col] >= signal_time].copy()
            
            logging.info(f"📈 Loaded {len(df)} ticks for {signal['ticker']}")
            
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def _create_event_queue(self, signals: List[Dict]) -> List[Dict]:
        """Create chronological event queue"""
        events = []
        
        for signal in signals:
            signal_id = f"{signal['ticker']}_{signal['strike']}{signal['right']}"
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
            df = signal.get('data')
            if df is not None and not df.empty:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
                
                for idx, row in df.iterrows():
                    events.append({
                        'type': 'tick',
                        'timestamp': row[time_col],
                        'signal_id': signal_id,
                        'data': row
                    })
        
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def _process_signal_event(self, event: Dict):
        """Process signal entry"""
        signal = event['signal']
        signal_id = event['signal_id']
        
        df = signal.get('data')
        if df is None or df.empty:
            return
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
        first_tick = df.iloc[0]
        
        entry_price = first_tick.get('mid', (first_tick.get('bid', 0) + first_tick.get('ask', 0)) / 2)
        
        if entry_price <= 0:
            return
        
        position_cost = entry_price * 100
        
        if self.portfolio['cash'] < position_cost:
            return
        
        self.portfolio['cash'] -= position_cost
        self.portfolio['positions'][signal_id] = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': first_tick[time_col],
            'quantity': 1,
            'highest_price': entry_price
        }
        
        self.active_contracts[signal_id] = signal
        
        logging.info(f"📊 ENTRY: {signal['ticker']} {signal['strike']}{signal['right']} @ ${entry_price:.2f}")
    
    def _process_tick_event(self, event: Dict):
        """Process tick and check exits"""
        signal_id = event['signal_id']
        
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        tick_data = event['data']
        
        current_price = tick_data.get('mid', (tick_data.get('bid', 0) + tick_data.get('ask', 0)) / 2)
        
        if current_price <= 0:
            return
        
        position['highest_price'] = max(position['highest_price'], current_price)
        
        exit_reason = self._evaluate_exit_conditions(position, current_price)
        
        if exit_reason:
            self._close_position(signal_id, exit_reason, current_price)
    
    def _evaluate_exit_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        """Evaluate all exit conditions"""
        entry_price = position['entry_price']
        highest_price = position['highest_price']
        signal = position['signal']
        
        # Native trailing stop
        if highest_price > entry_price:
            native_trail_stop = highest_price * (1 - self.native_trail_percent / 100)
            if current_price <= native_trail_stop:
                return f"native_trail_{self.native_trail_percent}pct"
        
        # Breakeven stop
        gain_pct = ((current_price - entry_price) / entry_price) * 100
        if gain_pct >= self.breakeven_trigger_percent:
            if current_price <= entry_price:
                return "breakeven_stop"
        
        # Pullback stop
        if self.trail_method == 'pullback_percent':
            if highest_price > entry_price:
                pullback_stop = highest_price * (1 - self.pullback_percent / 100)
                if current_price <= pullback_stop:
                    return f"pullback_{self.pullback_percent}pct"
        
        return None
    
    def _close_position(self, signal_id: str, exit_reason: str, exit_price: float):
        """Close position and log trade"""
        position = self.portfolio['positions'].pop(signal_id)
        
        entry_price = position['entry_price']
        pnl = (exit_price - entry_price) * 100
        
        self.portfolio['cash'] += exit_price * 100
        
        if signal_id in self.active_contracts:
            del self.active_contracts[signal_id]
        
        trade = {
            'ticker': position['signal']['ticker'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason
        }
        
        self.trade_log.append(trade)
        
        logging.info(f"📤 EXIT: {position['signal']['ticker']} @ ${exit_price:.2f} | {exit_reason} | P&L: ${pnl:.2f}")
    
    def _close_all_positions(self):
        """Close remaining positions at end of day"""
        for signal_id in list(self.portfolio['positions'].keys()):
            position = self.portfolio['positions'][signal_id]
            
            df = position['signal'].get('data')
            if df is not None and not df.empty:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
                last_tick = df.iloc[-1]
                exit_price = last_tick.get('mid', last_tick.get('close', 0))
                
                if exit_price > 0:
                    self._close_position(signal_id, 'eod_close', exit_price)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""
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
        
        profit_factor = (avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss > 0 else 0
        
        final_capital = self.starting_capital + total_pnl
        return_pct = (total_pnl / self.starting_capital) * 100
        
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
            'avg_minutes_held': 0,
            'exit_reasons': exit_reasons
        }
        
        logging.info("\n" + "="*80)
        logging.info("📊 BACKTEST RESULTS")
        logging.info("="*80)
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.1f}%")
        logging.info(f"Total P&L: ${total_pnl:,.2f}")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info(f"Final Capital: ${final_capital:,.2f} ({return_pct:+.2f}%)")
        logging.info("="*80)
        
        return results
    
    def _empty_results(self) -> Dict:
        """Empty results structure"""
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
    
    print(f"\n{'='*80}")
    print(f"FINAL: {results['total_trades']} trades | ${results['total_pnl']:.2f} P&L")
    print(f"{'='*80}")
