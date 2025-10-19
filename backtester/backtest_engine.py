#!/usr/bin/env python3
"""
options_backtest_engine.py - PROPER OPTIONS BACKTESTING
Designed specifically for 0DTE and short-dated options with correct put/call logic
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename_databento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OptionsBacktestEngine:
    """
    Properly designed backtesting engine for options trading.
    Handles puts vs calls, realistic fills, and proper trailing logic.
    """
    
    def __init__(self, signals_file, data_folder):
        self.signals_file = Path(signals_file)
        self.data_folder = Path(data_folder)
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
        # Portfolio tracking
        self.starting_capital = 100000
        self.portfolio = {
            'cash': self.starting_capital,
            'positions': {}
        }
        self.trades = []
        
    def run(self, params=None):
        """Run backtest with given parameters"""
        # Load signals
        signals = self._load_signals()
        if not signals:
            logging.error("No signals loaded")
            return
        
        logging.info(f"Testing {len(signals)} signals")
        
        # Process each signal independently (more realistic for options)
        for signal in signals:
            self._process_signal(signal, params)
        
        # Calculate results
        return self._calculate_results()
    
    def _load_signals(self):
        """Load and parse signals from file"""
        signals = []
        
        with open(self.signals_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:'):
                    continue
                
                if '|' in line:
                    try:
                        parts = line.split('|')
                        timestamp_str = parts[0].strip()
                        trader = parts[1].strip()
                        signal_text = parts[2].strip()
                        
                        # Parse the signal
                        profile = {'assume_buy_on_ambiguous': True, 'ambiguous_expiry_enabled': True}
                        parsed = self.signal_parser.parse_signal(signal_text, profile)
                        
                        if parsed:
                            parsed['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            parsed['trader'] = trader
                            signals.append(parsed)
                            logging.debug(f"Loaded: {trader} - {signal_text}")
                    except Exception as e:
                        logging.warning(f"Could not parse line: {line} - {e}")
        
        logging.info(f"Loaded {len(signals)} signals")
        return signals
    
    def _process_signal(self, signal, params=None):
        """Process a single signal through its lifecycle"""
        
        # Default parameters if none provided
        if params is None:
            params = {
                'breakeven_trigger_percent': 10,
                'trail_activate_percent': 15,
                'trail_distance_percent': 10,
                'stop_loss_percent': 50,
                'profit_target_percent': 100,
                'time_stop_minutes': 120
            }
        
        # Load price data for this signal
        price_data = self._load_price_data(signal)
        if price_data is None or price_data.empty:
            logging.warning(f"No price data for {signal['ticker']} {signal['strike']}{signal['contract_type'][0]}")
            return
        
        # Get entry price (first tick after signal)
        entry_time = signal['timestamp']
        entry_data = price_data[price_data['timestamp'] >= entry_time].iloc[0] if not price_data[price_data['timestamp'] >= entry_time].empty else None
        
        if entry_data is None:
            logging.warning(f"No entry price found for signal at {entry_time}")
            return
        
        entry_price = entry_data['close']
        entry_timestamp = entry_data['timestamp']
        
        # Determine if PUT or CALL
        is_put = signal['contract_type'].upper().startswith('P')
        
        # Calculate position size (10% of current capital)
        position_value = self.portfolio['cash'] * 0.10
        contracts = max(1, int(position_value / (entry_price * 100)))
        
        # Track position
        position = {
            'signal': signal,
            'entry_time': entry_timestamp,
            'entry_price': entry_price,
            'contracts': contracts,
            'is_put': is_put,
            'highest': entry_price,
            'lowest': entry_price,
            'breakeven_active': False,
            'trail_active': False,
            'trail_stop': None
        }
        
        # Process each tick after entry
        exit_time = None
        exit_price = None
        exit_reason = None
        
        for _, tick in price_data[price_data['timestamp'] > entry_timestamp].iterrows():
            current_price = tick['close']
            current_time = tick['timestamp']
            
            # Update high/low watermarks
            position['highest'] = max(position['highest'], current_price)
            position['lowest'] = min(position['lowest'], current_price)
            
            # Calculate P&L percentage
            if is_put:
                # Put gains when price goes down
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                profit_price = entry_price - current_price
            else:
                # Call gains when price goes up
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                profit_price = current_price - entry_price
            
            # Check exit conditions in priority order
            
            # 1. Time stop (close position after X minutes)
            minutes_held = (current_time - entry_timestamp).total_seconds() / 60
            if minutes_held >= params['time_stop_minutes']:
                exit_time = current_time
                exit_price = current_price
                exit_reason = f"Time stop ({params['time_stop_minutes']} min)"
                break
            
            # 2. Stop loss
            if pnl_pct <= -params['stop_loss_percent']:
                exit_time = current_time
                exit_price = current_price
                exit_reason = f"Stop loss (-{params['stop_loss_percent']}%)"
                break
            
            # 3. Profit target
            if pnl_pct >= params['profit_target_percent']:
                exit_time = current_time
                exit_price = current_price
                exit_reason = f"Profit target (+{params['profit_target_percent']}%)"
                break
            
            # 4. Trailing stop logic
            if not position['trail_active'] and pnl_pct >= params['trail_activate_percent']:
                position['trail_active'] = True
                # Set initial trail stop
                if is_put:
                    position['trail_stop'] = position['lowest'] * (1 + params['trail_distance_percent']/100)
                else:
                    position['trail_stop'] = position['highest'] * (1 - params['trail_distance_percent']/100)
                logging.debug(f"Trail activated at {current_time}, stop at ${position['trail_stop']:.2f}")
            
            # Update trailing stop if active
            if position['trail_active']:
                if is_put:
                    # For puts, trail above the lowest price
                    new_stop = position['lowest'] * (1 + params['trail_distance_percent']/100)
                    position['trail_stop'] = min(position['trail_stop'], new_stop)
                    if current_price >= position['trail_stop']:
                        exit_time = current_time
                        exit_price = current_price
                        exit_reason = f"Trail stop (put)"
                        break
                else:
                    # For calls, trail below the highest price
                    new_stop = position['highest'] * (1 - params['trail_distance_percent']/100)
                    position['trail_stop'] = max(position['trail_stop'], new_stop)
                    if current_price <= position['trail_stop']:
                        exit_time = current_time
                        exit_price = current_price
                        exit_reason = f"Trail stop (call)"
                        break
            
            # 5. Breakeven stop (simplified - just protects entry after profit threshold)
            if not position['breakeven_active'] and pnl_pct >= params['breakeven_trigger_percent']:
                position['breakeven_active'] = True
            
            if position['breakeven_active']:
                if (is_put and current_price >= entry_price) or (not is_put and current_price <= entry_price):
                    exit_time = current_time
                    exit_price = entry_price  # Exit at breakeven
                    exit_reason = "Breakeven stop"
                    break
        
        # If no exit triggered, use last price (market close)
        if exit_time is None:
            last_tick = price_data.iloc[-1]
            exit_time = last_tick['timestamp']
            exit_price = last_tick['close']
            exit_reason = "Market close"
        
        # Calculate P&L
        if is_put:
            pnl = (entry_price - exit_price) * contracts * 100
        else:
            pnl = (exit_price - entry_price) * contracts * 100
        
        # Record trade
        trade = {
            'trader': signal['trader'],
            'ticker': signal['ticker'],
            'strike': signal['strike'],
            'type': 'PUT' if is_put else 'CALL',
            'entry_time': entry_timestamp,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'contracts': contracts,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * contracts * 100)) * 100,
            'exit_reason': exit_reason,
            'minutes_held': (exit_time - entry_timestamp).total_seconds() / 60
        }
        
        self.trades.append(trade)
        self.portfolio['cash'] += pnl
        
        logging.info(f"[{signal['trader']}] {signal['ticker']} {signal['strike']}{signal['contract_type'][0]}: "
                    f"Entry ${entry_price:.2f} → Exit ${exit_price:.2f} | "
                    f"P&L: ${pnl:.0f} ({trade['pnl_pct']:.1f}%) | "
                    f"Held: {trade['minutes_held']:.0f}min | "
                    f"Reason: {exit_reason}")
    
    def _load_price_data(self, signal):
        """Load historical price data for a signal"""
        # Build filename
        expiry = signal['expiry_date'].replace('-', '')
        filename = get_data_filename_databento(
            signal['ticker'],
            expiry,
            signal['strike'],
            signal['contract_type'][0].upper()
        )
        
        filepath = self.data_folder / filename
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
            # For performance, sample data if too large
            if len(df) > 10000:
                # Keep every 10th row to reduce processing
                df = df.iloc[::10].reset_index(drop=True)
                logging.debug(f"Sampled data to {len(df)} rows")
            
            return df
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            return None
    
    def _calculate_results(self):
        """Calculate backtest results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'final_capital': self.portfolio['cash']
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        
        total_pnl = trades_df['pnl'].sum()
        win_rate = len(wins) / len(trades_df) * 100
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if len(losses) > 0 else float('inf')
        
        results = {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.portfolio['cash'],
            'return_pct': ((self.portfolio['cash'] - self.starting_capital) / self.starting_capital) * 100,
            'avg_minutes_held': trades_df['minutes_held'].mean(),
            'trades': trades_df
        }
        
        # Save detailed results
        output_file = self.data_folder.parent / 'options_backtest_results.csv'
        trades_df.to_csv(output_file, index=False)
        
        return results


def main():
    """Run a single backtest with example parameters"""
    
    # Example parameters optimized for 0DTE options
    params = {
        'breakeven_trigger_percent': 8,   # Activate breakeven at 8% profit
        'trail_activate_percent': 12,     # Start trailing at 12% profit
        'trail_distance_percent': 8,      # Trail 8% from high/low
        'stop_loss_percent': 30,          # Stop out at 30% loss
        'profit_target_percent': 50,      # Take profit at 50% gain
        'time_stop_minutes': 90           # Close after 90 minutes
    }
    
    engine = OptionsBacktestEngine(
        signals_file='backtester/signals_to_test.txt',
        data_folder='backtester/historical_data'
    )
    
    results = engine.run(params)
    
    print("\n" + "="*60)
    print("OPTIONS BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Total P&L: ${results['total_pnl']:,.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Return: {results['return_pct']:.1f}%")
    print(f"Avg Hold Time: {results['avg_minutes_held']:.0f} minutes")
    print("="*60)
    
    if results['total_trades'] > 0:
        print("\nTop 5 Trades by P&L:")
        top_trades = results['trades'].nlargest(5, 'pnl')[['ticker', 'strike', 'type', 'pnl', 'pnl_pct', 'exit_reason']]
        print(top_trades.to_string(index=False))
        
        print("\nWorst 5 Trades by P&L:")
        worst_trades = results['trades'].nsmallest(5, 'pnl')[['ticker', 'strike', 'type', 'pnl', 'pnl_pct', 'exit_reason']]
        print(worst_trades.to_string(index=False))


if __name__ == "__main__":
    main()
