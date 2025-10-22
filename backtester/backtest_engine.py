#!/usr/bin/env python3
"""
backtest_engine.py - TIMEZONE FIXED VERSION
===================================================
CRITICAL FIX: All timestamps are now timezone-aware (UTC)
- Databento CSVs have tz-aware timestamps (UTC)
- Signal timestamps now converted to tz-aware (UTC)
- All comparisons work correctly
- No more "Cannot compare tz-naive and tz-aware" errors

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


class BacktestEngine:
    """
    Event-driven backtest engine for day trading options
    NOW WITH PROPER TIMEZONE HANDLING
    """
    
    def __init__(self, signal_file_path: str, data_folder_path: str = "backtester/historical_data"):
        self.signal_file_path = Path(signal_file_path)
        self.data_folder_path = Path(data_folder_path)
        self.starting_capital = 100000
        
        # Exit parameters (set via run_simulation)
        self.breakeven_trigger_percent = 10
        self.trail_method = 'pullback_percent'
        self.pullback_percent = 10
        self.atr_period = 14
        self.atr_multiplier = 1.5
        self.native_trail_percent = 25
        
        # State tracking
        self.portfolio = {'cash': self.starting_capital, 'positions': {}}
        self.active_contracts = {}
        self.trade_log = []
        
        # ATR calculation storage
        self.atr_values = {}  # {signal_id: current_atr}
        self.atr_stops = {}   # {signal_id: current_stop_price}
    
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
        with open(self.signal_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:'):
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    signal_text = parts[2].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    signal_text = line
                
                parsed = parser.parse_signal(signal_text, default_profile)
                if parsed:
                    parsed['timestamp'] = timestamp_str
                    signals.append(parsed)
        
        logging.info(f"Loaded {len(signals)} signals")
        return signals
    
    def _load_signal_data(self, signal: Dict):
        """
        Load historical data for a signal
        TIMEZONE FIX: Ensure all timestamps are timezone-aware (UTC)
        """
        ticker = signal['ticker']
        expiry = signal['expiry_date']
        strike = signal['strike']
        right = 'C' if signal['contract_type'] == 'CALL' else 'P'
        
        filename = get_data_filename_databento(ticker, expiry, strike, right)
        filepath = self.data_folder_path / filename
        
        if not filepath.exists():
            logging.warning(f"Data file not found: {filename}")
            signal['data'] = None
            return
        
        try:
            df = pd.read_csv(filepath)
            
            # Identify time column
            time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
            
            # TIMEZONE FIX: Parse timestamps as UTC-aware
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            
            # If timestamps were parsed without timezone, add UTC
            if df[time_col].dt.tz is None:
                df[time_col] = df[time_col].dt.tz_localize('UTC')
            
            # Calculate mid price if not present
            if 'mid' not in df.columns:
                if 'bid' in df.columns and 'ask' in df.columns:
                    df['mid'] = (df['bid'] + df['ask']) / 2
                elif 'close' in df.columns:
                    df['mid'] = df['close']
            
            signal['data'] = df
            logging.debug(f"Loaded {len(df)} bars for {ticker} (timezone-aware: {df[time_col].dt.tz})")
            
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            signal['data'] = None
    
    def _build_event_queue(self, signals: List[Dict]) -> List[Dict]:
        """
        Build chronological event queue
        TIMEZONE FIX: Ensure signal timestamps are timezone-aware (UTC)
        """
        events = []
        
        for signal in signals:
            if signal.get('data') is None:
                continue
            
            signal_id = f"{signal['ticker']}_{signal['expiry_date']}_{signal['strike']}_{signal['contract_type']}"
            signal['signal_id'] = signal_id
            
            # TIMEZONE FIX: Parse signal timestamp as UTC-aware
            signal_time = pd.to_datetime(signal['timestamp'], utc=True)
            
            # If signal_time has no timezone, add UTC
            if signal_time.tz is None:
                signal_time = signal_time.tz_localize('UTC')
            
            # Add signal entry event
            events.append({
                'timestamp': signal_time,
                'type': 'signal',
                'signal': signal
            })
            
            # Add tick events
            df = signal['data']
            time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
            
            for _, row in df.iterrows():
                tick_time = row[time_col]
                
                # COMPARISON NOW WORKS: Both are tz-aware
                if tick_time >= signal_time:
                    events.append({
                        'timestamp': tick_time,
                        'type': 'tick',
                        'signal_id': signal_id,
                        'data': row.to_dict()
                    })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        logging.info(f"Built event queue with {len(events)} events (all timestamps tz-aware)")
        return events
    
    def _process_signal_event(self, event: Dict):
        """
        Process signal entry
        TIMEZONE FIX: Signal time comparison now works
        """
        signal = event['signal']
        signal_id = signal['signal_id']
        
        if signal.get('data') is None:
            return
        
        df = signal['data']
        time_col = 'timestamp' if 'timestamp' in df.columns else 'ts_event'
        
        # TIMEZONE FIX: Parse signal time as UTC-aware
        signal_time = pd.to_datetime(signal['timestamp'], utc=True)
        if signal_time.tz is None:
            signal_time = signal_time.tz_localize('UTC')
        
        # Find first tick after signal (COMPARISON WORKS NOW)
        future_ticks = df[df[time_col] >= signal_time]
        
        if future_ticks.empty:
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
        
        logging.info(f"ENTRY: {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} @ ${entry_price:.2f}")
    
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
        
        # 3. NATIVE TRAILING STOP (last resort)
        native_stop_price = highest_price * (1 - self.native_trail_percent / 100)
        if current_price <= native_stop_price:
            return "native_trail"
        
        return None
    
    def _close_position(self, signal_id: str, exit_reason: str, exit_price: float, exit_time):
        """Close a position and log the trade"""
        if signal_id not in self.portfolio['positions']:
            return
        
        position = self.portfolio['positions'][signal_id]
        signal = position['signal']
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate P&L
        if signal['contract_type'] == 'CALL':
            pnl = (exit_price - entry_price) * quantity * 100
        else:  # PUT
            pnl = (exit_price - entry_price) * quantity * 100
        
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
            'quantity': quantity,
            'pnl': pnl,
            'exit_reason': exit_reason
        })
        
        # Update portfolio
        self.portfolio['cash'] += pnl
        del self.portfolio['positions'][signal_id]
        del self.active_contracts[signal_id]
        
        # Clean up ATR tracking
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
        """
        Calculate backtest results
        TIMEZONE FIX: Handle timezone-aware datetimes in hold time calculation
        """
        if not self.trade_log:
            return self._empty_results()
        
        df = pd.DataFrame(self.trade_log)
        
        # Calculate hold times (TIMEZONE FIX: Works with tz-aware timestamps)
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
            df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
            df['hold_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
            avg_hold_minutes = df['hold_minutes'].mean()
        else:
            avg_hold_minutes = 0
        
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
            'avg_minutes_held': avg_hold_minutes,
            'exit_reasons': exit_reasons
        }
        
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
