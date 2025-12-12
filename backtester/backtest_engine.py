#!/usr/bin/env python3
"""
backtest_engine.py - TIMEZONE FIXED VERSION v3 + OUTLIER METRICS
=================================================================
CRITICAL FIX: Signal timestamps are now properly converted from LOCAL TIME to UTC

THE BUG:
- Signal timestamps in files are in USER'S LOCAL TIME (Mountain Time)
- Previous code treated them as UTC, causing entries at wrong times
- Example: "07:35:00" MT should be 13:35 UTC, but was being treated as 07:35 UTC

THE FIX:
- Added SIGNAL_TIMEZONE setting (default: America/Denver for Mountain Time)
- Signal timestamps are now properly localized then converted to UTC
- Data timestamps (from Databento) are already in UTC - no change needed

NEW IN v3:
- Added outlier-resistant metrics (median_pnl, trimmed_pnl, consistency_score)
- Outlier detection flag when single trade > 50% of total P&L
- pnl_without_best metric for robustness testing

Exit logic maintained:
- Correct exit priority order
- Working ATR trailing stop logic  
- Native trail as last resort only
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import pytz

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.signal_parser import SignalParser
from services.config import Config
from services.utils import get_data_filename_databento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# TIMEZONE CONFIGURATION
# =============================================================================
# Set this to YOUR timezone - the timezone your signal timestamps are in
# Common options:
#   'America/Denver'     - Mountain Time (MT)
#   'America/Chicago'    - Central Time (CT)  
#   'America/New_York'   - Eastern Time (ET)
#   'America/Los_Angeles' - Pacific Time (PT)
#   'UTC'                - If your signals are already in UTC
# =============================================================================
SIGNAL_TIMEZONE = 'America/Denver'  # <-- CHANGE THIS IF NEEDED


class BacktestEngine:
    """
    Event-driven backtest engine for day trading options
    NOW WITH PROPER LOCAL TIME -> UTC CONVERSION
    AND OUTLIER-RESISTANT METRICS
    """
    
    def __init__(self, signal_file_path: str, data_folder_path: str = "backtester/historical_data"):
        self.signal_file_path = Path(signal_file_path)
        self.data_folder_path = Path(data_folder_path)
        self.starting_capital = 100000
        
        # Timezone for signal timestamps
        self.signal_tz = pytz.timezone(SIGNAL_TIMEZONE)
        
        # Exit parameters (set via run_simulation)
        self.breakeven_trigger_percent = 10
        self.trail_method = 'pullback_percent'
        self.pullback_percent = 10
        self.atr_period = 14
        self.atr_multiplier = 1.5
        self.native_trail_percent = 25
        
        # PSAR parameters
        self.psar_enabled = False
        self.psar_start = 0.02
        self.psar_increment = 0.02
        self.psar_max = 0.2
        
        # RSI parameters
        self.rsi_hook_enabled = False
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # State tracking
        self.portfolio = {'cash': self.starting_capital, 'positions': {}}
        self.active_contracts = {}
        self.trade_log = []
        
        # ATR calculation storage
        self.atr_values = {}   # {signal_id: current_atr}
        self.atr_stops = {}    # {signal_id: current_stop_price}
    
    def run_simulation(self, params: Dict = None) -> Dict:
        """Run backtest with given parameters"""
        
        # Apply parameters if provided
        if params:
            self.breakeven_trigger_percent = params.get('breakeven_trigger_percent', 10)
            self.trail_method = params.get('trail_method', 'pullback_percent')
            self.pullback_percent = params.get('pullback_percent', 10)
            self.atr_period = params.get('atr_period', 14)
            self.atr_multiplier = params.get('atr_multiplier', 1.5)
            self.native_trail_percent = params.get('native_trail_percent', 25)
            
            # PSAR parameters
            self.psar_enabled = params.get('psar_enabled', False)
            self.psar_start = params.get('psar_start', 0.02)
            self.psar_increment = params.get('psar_increment', 0.02)
            self.psar_max = params.get('psar_max', 0.2)
            
            # RSI parameters
            self.rsi_hook_enabled = params.get('rsi_hook_enabled', False)
            self.rsi_period = params.get('rsi_period', 14)
            self.rsi_overbought = params.get('rsi_overbought', 70)
            self.rsi_oversold = params.get('rsi_oversold', 30)
        
        # Reset state
        self.portfolio = {'cash': self.starting_capital, 'positions': {}}
        self.active_contracts = {}
        self.trade_log = []
        self.atr_values = {}
        self.atr_stops = {}
        
        # Load signals
        signals = self._load_signals()
        if not signals:
            logging.warning("No valid signals found")
            return self._empty_results()
        
        # Load historical data for each signal
        for signal in signals:
            self._load_signal_data(signal)
        
        # Build event queue
        events = self._build_event_queue(signals)
        
        if not events:
            logging.warning("No events in queue")
            return self._empty_results()
        
        # Process events
        for event in events:
            if event['type'] == 'signal':
                self._process_signal_event(event)
            elif event['type'] == 'tick':
                self._process_tick_event(event)
        
        # Close remaining positions
        self._close_all_positions()
        
        # Calculate results
        return self._calculate_results()
    
    def _load_signals(self) -> List[Dict]:
        """Load and parse signals from file"""
        config = Config()
        parser = SignalParser(config)
        
        default_profile = config.profiles[0] if config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True,
            'buzzwords_buy': [],
            'buzzwords_sell': [],
            'channel_id': 'backtest'
        }
        
        signals = []
        with open(self.signal_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:'):
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    signal_text = parts[2].strip() if len(parts) > 2 else parts[1].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    signal_text = line
                
                parsed = parser.parse_signal(signal_text, default_profile)
                if parsed:
                    parsed['timestamp'] = timestamp_str
                    parsed['signal_id'] = f"{parsed['ticker']}_{parsed['strike']}_{parsed['contract_type']}_{timestamp_str}"
                    signals.append(parsed)
        
        logging.info(f"Loaded {len(signals)} signals")
        return signals
    
    def _load_signal_data(self, signal: Dict):
        """Load historical data for a signal"""
        filename = get_data_filename_databento(
            signal['ticker'],
            signal['expiry_date'],
            signal['strike'],
            signal['contract_type'][0] if len(signal['contract_type']) > 1 else signal['contract_type']
        )
        
        filepath = self.data_folder_path / filename
        
        if not filepath.exists():
            logging.warning(f"Data file not found: {filepath}")
            signal['data'] = None
            return
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp column
        time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        
        # Calculate mid price if not present
        if 'mid' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
            elif 'close' in df.columns:
                df['mid'] = df['close']
        
        signal['data'] = df
    
    def _build_event_queue(self, signals: List[Dict]) -> List[Dict]:
        """Build chronological event queue"""
        events = []
        
        for signal in signals:
            if signal.get('data') is None:
                continue
            
            signal_id = signal['signal_id']
            
            # Parse signal timestamp and convert to UTC
            # Signal timestamps are in LOCAL TIME (Mountain Time)
            timestamp_str = signal['timestamp']
            
            try:
                # Parse the naive datetime
                naive_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                # Localize to signal timezone (Mountain Time)
                local_dt = self.signal_tz.localize(naive_dt)
                
                # Convert to UTC
                signal_time = local_dt.astimezone(pytz.UTC)
                
            except Exception as e:
                logging.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
                continue
            
            # Add signal event
            events.append({
                'timestamp': signal_time,
                'type': 'signal',
                'signal': signal
            })
            
            # Add tick events (data is already in UTC)
            df = signal['data']
            time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
            
            for _, row in df.iterrows():
                tick_time = row[time_col]
                
                # Only include ticks after signal time
                if tick_time >= signal_time:
                    events.append({
                        'timestamp': tick_time,
                        'type': 'tick',
                        'signal_id': signal_id,
                        'data': row.to_dict()
                    })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        logging.info(f"Built event queue with {len(events)} events")
        return events
    
    def _process_signal_event(self, event: Dict):
        """Process signal entry"""
        signal = event['signal']
        signal_id = signal['signal_id']
        
        if signal.get('data') is None:
            return
        
        df = signal['data']
        time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
        
        # Signal time is already in UTC from _build_event_queue
        signal_time = event['timestamp']
        
        # Find first tick after signal
        future_ticks = df[df[time_col] >= signal_time]
        
        if future_ticks.empty:
            logging.warning(f"No ticks found after signal time {signal_time} for {signal_id}")
            return
        
        first_tick = future_ticks.iloc[0].to_dict()
        entry_price = first_tick.get('mid', first_tick.get('close', 0))
        
        if entry_price <= 0:
            return
        
        # Enter position
        self.portfolio['positions'][signal_id] = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': first_tick[time_col],
            'quantity': 1,
            'highest_price': entry_price,
            'breakeven_activated': False
        }
        
        self.active_contracts[signal_id] = signal
        
        # Initialize ATR tracking if using ATR method
        if self.trail_method == 'atr':
            self.atr_values[signal_id] = None
            self.atr_stops[signal_id] = None
        
        logging.info(f"ENTRY: {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} @ ${entry_price:.2f} "
                    f"(signal: {signal['timestamp']} -> entry: {first_tick[time_col]})")
    
    def _process_tick_event(self, event: Dict):
        """Process tick and check exits"""
        signal_id = event['signal_id']
        
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        tick_data = event['data']
        
        time_col = 'timestamp' if 'timestamp' in tick_data else 'ts_event'
        current_time = tick_data[time_col] if time_col in tick_data else event['timestamp']
        
        current_price = tick_data.get('mid', (tick_data.get('bid', 0) + tick_data.get('ask', 0)) / 2)
        
        if current_price <= 0:
            return
        
        # Update highest price
        position['highest_price'] = max(position['highest_price'], current_price)
        
        # Update ATR if using ATR method
        if self.trail_method == 'atr':
            self._update_atr(signal_id, position, tick_data)
        
        # Evaluate exits
        exit_reason = self._evaluate_exit_conditions(position, current_price, signal_id)
        
        if exit_reason:
            self._close_position(signal_id, exit_reason, current_price, current_time)
    
    def _update_atr(self, signal_id: str, position: Dict, tick_data: Dict):
        """Update ATR calculation and trailing stop"""
        signal = position['signal']
        df = signal['data']
        
        # Need at least atr_period bars
        if len(df) < self.atr_period:
            return
        
        # Calculate True Range for recent bars
        recent_bars = df.tail(self.atr_period)
        
        if 'high' in recent_bars.columns and 'low' in recent_bars.columns:
            tr = pd.DataFrame()
            tr['hl'] = recent_bars['high'] - recent_bars['low']
            tr['hc'] = abs(recent_bars['high'] - recent_bars['close'].shift(1))
            tr['lc'] = abs(recent_bars['low'] - recent_bars['close'].shift(1))
            
            true_range = tr[['hl', 'hc', 'lc']].max(axis=1)
            atr = true_range.mean()
            
            self.atr_values[signal_id] = atr
            
            # Set ATR trailing stop
            highest_price = position['highest_price']
            self.atr_stops[signal_id] = highest_price - (atr * self.atr_multiplier)
    
    def _evaluate_exit_conditions(self, position: Dict, current_price: float, signal_id: str) -> Optional[str]:
        """
        Evaluate exits in CORRECT priority order:
        1. Breakeven (tightest stop)
        2. ATR/Pullback (dynamic trail)
        3. Native trail (last resort)
        """
        entry_price = position['entry_price']
        highest_price = position['highest_price']
        
        # 1. BREAKEVEN (checked first - tightest stop)
        gain_pct = ((highest_price - entry_price) / entry_price) * 100
        
        if gain_pct >= self.breakeven_trigger_percent:
            position['breakeven_activated'] = True
            
            if current_price <= entry_price:
                return "breakeven_stop"
        
        # 2. DYNAMIC TRAILING STOPS (ATR or Pullback)
        if self.trail_method == 'atr' and signal_id in self.atr_stops:
            atr_stop = self.atr_stops[signal_id]
            if atr_stop and current_price <= atr_stop:
                return "atr_trail"
        
        elif self.trail_method == 'pullback_percent':
            pullback_threshold = highest_price * (1 - self.pullback_percent / 100)
            if current_price <= pullback_threshold:
                return "pullback_stop"
        
        # 3. NATIVE TRAIL (last resort - always active)
        native_stop = highest_price * (1 - self.native_trail_percent / 100)
        if current_price <= native_stop:
            return "native_trail"
        
        return None
    
    def _close_position(self, signal_id: str, exit_reason: str, exit_price: float, exit_time):
        """Close position and log trade"""
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        signal = position['signal']
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate P&L (assuming 100 shares per contract for options)
        pnl = (exit_price - entry_price) * 100 * quantity
        
        # Log trade
        self.trade_log.append({
            'signal_id': signal_id,
            'ticker': signal['ticker'],
            'strike': signal['strike'],
            'contract_type': signal['contract_type'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'highest_price': position['highest_price'],
            'breakeven_activated': position['breakeven_activated']
        })
        
        # Update portfolio
        self.portfolio['cash'] += pnl
        del self.portfolio['positions'][signal_id]
        
        # Clean up tracking
        if signal_id in self.active_contracts:
            del self.active_contracts[signal_id]
        if signal_id in self.atr_values:
            del self.atr_values[signal_id]
        if signal_id in self.atr_stops:
            del self.atr_stops[signal_id]
        
        logging.info(f"EXIT: {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} @ ${exit_price:.2f} | "
                    f"P&L: ${pnl:.2f} | Reason: {exit_reason}")
    
    def _close_all_positions(self):
        """Close all remaining positions at end of day"""
        for signal_id in list(self.active_contracts.keys()):
            signal = self.active_contracts[signal_id]
            
            if signal.get('data') is not None:
                df = signal['data']
                time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
                last_tick = df.iloc[-1]
                exit_price = last_tick.get('mid', last_tick.get('close', 0))
                exit_time = last_tick[time_col]
                
                if exit_price > 0:
                    self._close_position(signal_id, 'eod_close', exit_price, exit_time)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results with outlier-resistant metrics"""
        if not self.trade_log:
            return self._empty_results()
        
        df = pd.DataFrame(self.trade_log)
        
        # Calculate hold times
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
            df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
            df['hold_time'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        else:
            df['hold_time'] = 0
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        
        total_pnl = df['pnl'].sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit if gross_profit > 0 else 0
        
        final_capital = self.starting_capital + total_pnl
        return_pct = (total_pnl / self.starting_capital) * 100
        
        avg_hold_minutes = df['hold_time'].mean() if len(df) > 0 else 0
        
        # Exit reason counts
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
            'exit_reasons': exit_reasons,
            'max_drawdown': 0  # Placeholder for now
        }
        
        # =================================================================
        # OUTLIER-RESISTANT METRICS
        # =================================================================
        trade_pnls = [t['pnl'] for t in self.trade_log]
        
        if len(trade_pnls) > 0:
            # Median P&L (not affected by outliers)
            results['median_pnl'] = float(np.median(trade_pnls))
            
            # Standard deviation (measures consistency)
            results['pnl_std'] = float(np.std(trade_pnls))
            
            # Max single trade (outlier detection)
            results['max_single_win'] = float(max(trade_pnls))
            results['max_single_loss'] = float(min(trade_pnls))
            
            # P&L without best trade (robustness check)
            sorted_pnls = sorted(trade_pnls, reverse=True)
            results['pnl_without_best'] = float(sum(sorted_pnls[1:])) if len(sorted_pnls) > 1 else 0
            
            # Trimmed P&L (exclude top and bottom 10%)
            if len(trade_pnls) >= 5:
                trim_count = max(1, len(trade_pnls) // 10)
                sorted_asc = sorted(trade_pnls)
                trimmed = sorted_asc[trim_count:-trim_count] if trim_count > 0 and len(sorted_asc) > 2*trim_count else sorted_asc
                results['trimmed_pnl'] = float(sum(trimmed))
            else:
                results['trimmed_pnl'] = results['total_pnl']
            
            # Consistency score (mean / std - Sharpe-like)
            if results['pnl_std'] > 0:
                results['consistency_score'] = float(np.mean(trade_pnls) / results['pnl_std'])
            else:
                results['consistency_score'] = 0.0
            
            # Outlier flag (best trade > 50% of total P&L)
            if results['total_pnl'] > 0:
                results['outlier_flag'] = results['max_single_win'] > (results['total_pnl'] * 0.5)
            else:
                results['outlier_flag'] = False
        else:
            results['median_pnl'] = 0
            results['pnl_std'] = 0
            results['max_single_win'] = 0
            results['max_single_loss'] = 0
            results['pnl_without_best'] = 0
            results['trimmed_pnl'] = 0
            results['consistency_score'] = 0
            results['outlier_flag'] = False
        
        # Log results
        logging.info("\n" + "="*80)
        logging.info("BACKTEST RESULTS")
        logging.info("="*80)
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.1f}%")
        logging.info(f"Total P&L: ${total_pnl:,.2f}")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info(f"Final Capital: ${final_capital:,.2f} ({return_pct:+.2f}%)")
        logging.info(f"Avg Hold Time: {avg_hold_minutes:.0f} minutes")
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
            'exit_reasons': {},
            'max_drawdown': 0,
            # Outlier-resistant metrics
            'median_pnl': 0,
            'pnl_std': 0,
            'max_single_win': 0,
            'max_single_loss': 0,
            'pnl_without_best': 0,
            'trimmed_pnl': 0,
            'consistency_score': 0,
            'outlier_flag': False
        }


if __name__ == "__main__":
    print("="*80)
    print("BACKTEST ENGINE - TIMEZONE FIXED v3 + OUTLIER METRICS")
    print(f"Signal timezone: {SIGNAL_TIMEZONE}")
    print("="*80)
    
    engine = BacktestEngine(
        signal_file_path="backtester/signals_to_test.txt",
        data_folder_path="backtester/historical_data"
    )
    
    results = engine.run_simulation()
    
    print(f"\n{'='*80}")
    print(f"FINAL: {results['total_trades']} trades | ${results['total_pnl']:.2f} P&L")
    print(f"Outlier Flag: {results['outlier_flag']} | P&L Without Best: ${results['pnl_without_best']:.2f}")
    print(f"{'='*80}")
