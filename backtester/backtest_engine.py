import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta
from services.utils import get_data_filename_databento

class BacktestEngine:
    """
    Backtesting engine for options trading strategies.
    """
    
    def __init__(self, config):
        self.config = config
        self.historical_data_dir = "backtester/historical_data"
        self.results = []
        self.trades = []
        self.current_position = None
        self.portfolio_value = 100000  # Starting capital
        self.initial_capital = 100000
        
    def run_backtest(self, signals_file: str, parameters: Dict) -> Dict:
        """
        Run backtest on signals with given parameters.
        """
        # Load signals
        signals = self._load_signals(signals_file)
        if not signals:
            logging.warning(f"No signals found in {signals_file}")
            return self._generate_empty_results()
        
        # Process each signal
        for signal in signals:
            self._process_signal(signal, parameters)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _load_signals(self, signals_file: str) -> List[Dict]:
        """
        Load and parse signals from file.
        """
        signals = []
        
        try:
            with open(signals_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    signal = self._parse_signal(line)
                    if signal:
                        signals.append(signal)
            
            logging.info(f"Loaded {len(signals)} signals from {signals_file}")
            return signals
            
        except Exception as e:
            logging.error(f"Error loading signals: {e}")
            return []
    
    def _parse_signal(self, line: str) -> Optional[Dict]:
        """
        Parse a signal line into components.
        """
        try:
            # Format: YYYY-MM-DD HH:MM:SS | Trader | TICKER STRIKEP/C MM/DD
            parts = line.split('|')
            if len(parts) != 3:
                return None
            
            timestamp_str = parts[0].strip()
            trader = parts[1].strip()
            signal_text = parts[2].strip()
            
            # Parse signal text (e.g., "SPX 6650P 09/29")
            signal_parts = signal_text.split()
            if len(signal_parts) != 3:
                return None
            
            ticker = signal_parts[0]
            
            # Extract strike and right (e.g., "6650P")
            strike_str = signal_parts[1]
            if strike_str[-1] in ['C', 'P']:
                strike = float(strike_str[:-1])
                right = strike_str[-1]
            else:
                return None
            
            # Parse expiry (MM/DD)
            expiry_parts = signal_parts[2].split('/')
            if len(expiry_parts) != 2:
                return None
            
            month = int(expiry_parts[0])
            day = int(expiry_parts[1])
            
            # Determine year based on signal timestamp
            signal_time = pd.to_datetime(timestamp_str)
            year = signal_time.year
            
            # If expiry month is earlier than signal month, it's next year
            if month < signal_time.month:
                year += 1
            
            expiry = f"{year}{month:02d}{day:02d}"
            
            return {
                'timestamp': timestamp_str,
                'trader': trader,
                'ticker': ticker,
                'strike': strike,
                'right': right,
                'expiry': expiry,
                'signal_text': signal_text
            }
            
        except Exception as e:
            logging.error(f"Error parsing signal '{line}': {e}")
            return None
    
    def _process_signal(self, signal: Dict, parameters: Dict):
        """
        Process a single signal with backtesting logic.
        """
        # Load historical data for this signal
        df = self._load_signal_data(signal)
        
        if df.empty:
            logging.warning(f"No data for signal: {signal['signal_text']}")
            return
        
        # Get entry price from data
        entry_price = self._get_entry_price_from_data(df)
        if entry_price is None or entry_price <= 0:
            logging.warning(f"Invalid entry price for signal: {signal['signal_text']}")
            return
        
        # Calculate position size (10% of portfolio per trade)
        position_size = int((self.portfolio_value * 0.10) / (entry_price * 100))
        if position_size < 1:
            position_size = 1
        
        # Initialize trade
        trade = {
            'signal': signal,
            'entry_time': df.iloc[0]['ts_event'],
            'entry_price': entry_price,
            'quantity': position_size,
            'exit_time': None,
            'exit_price': None,
            'pnl': 0,
            'exit_reason': None,
            'max_profit': 0,
            'max_loss': 0
        }
        
        # Track position performance
        highest_price = entry_price
        lowest_price = entry_price
        breakeven_triggered = False
        trail_stop = None
        
        # Process each tick
        for idx, row in df.iterrows():
            current_price = (row['bid'] + row['ask']) / 2 if 'bid' in row and 'ask' in row else row.get('close', entry_price)
            
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Update extremes
            highest_price = max(highest_price, current_price)
            lowest_price = min(lowest_price, current_price)
            
            # Calculate current P&L
            if signal['right'] == 'C':
                current_pnl = (current_price - entry_price) * position_size * 100
            else:  # PUT
                current_pnl = (entry_price - current_price) * position_size * 100
            
            # Update max profit/loss
            trade['max_profit'] = max(trade['max_profit'], current_pnl)
            trade['max_loss'] = min(trade['max_loss'], current_pnl)
            
            # Check exit conditions
            exit_triggered = False
            
            # 1. Native trailing stop
            native_trail_pct = parameters.get('native_trail_percent', 25) / 100
            if signal['right'] == 'C':
                native_stop = highest_price * (1 - native_trail_pct)
                if current_price <= native_stop:
                    exit_triggered = True
                    trade['exit_reason'] = f'Native trail {parameters.get("native_trail_percent")}%'
            else:  # PUT
                native_stop = lowest_price * (1 + native_trail_pct)
                if current_price >= native_stop:
                    exit_triggered = True
                    trade['exit_reason'] = f'Native trail {parameters.get("native_trail_percent")}%'
            
            # 2. Breakeven stop
            if not exit_triggered:
                breakeven_trigger = parameters.get('breakeven_trigger_percent', 10) / 100
                if signal['right'] == 'C':
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= breakeven_trigger and not breakeven_triggered:
                        breakeven_triggered = True
                    if breakeven_triggered and current_price <= entry_price:
                        exit_triggered = True
                        trade['exit_reason'] = 'Breakeven stop'
                else:  # PUT
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= breakeven_trigger and not breakeven_triggered:
                        breakeven_triggered = True
                    if breakeven_triggered and current_price >= entry_price:
                        exit_triggered = True
                        trade['exit_reason'] = 'Breakeven stop'
            
            # 3. Pullback/ATR trailing
            if not exit_triggered and parameters.get('trail_method') == 'pullback_percent':
                pullback_pct = parameters.get('pullback_percent', 10) / 100
                if signal['right'] == 'C':
                    pullback_stop = highest_price * (1 - pullback_pct)
                    if current_price <= pullback_stop:
                        exit_triggered = True
                        trade['exit_reason'] = f'Pullback {parameters.get("pullback_percent")}%'
                else:  # PUT
                    pullback_stop = lowest_price * (1 + pullback_pct)
                    if current_price >= pullback_stop:
                        exit_triggered = True
                        trade['exit_reason'] = f'Pullback {parameters.get("pullback_percent")}%'
            
            # Exit if triggered
            if exit_triggered:
                trade['exit_time'] = row['ts_event']
                trade['exit_price'] = current_price
                trade['pnl'] = current_pnl
                break
        
        # If no exit triggered, exit at last price (market close)
        if trade['exit_time'] is None:
            trade['exit_time'] = df.iloc[-1]['ts_event']
            trade['exit_price'] = df.iloc[-1]['ask'] if 'ask' in df.columns else entry_price
            
            if signal['right'] == 'C':
                trade['pnl'] = (trade['exit_price'] - entry_price) * position_size * 100
            else:  # PUT
                trade['pnl'] = (entry_price - trade['exit_price']) * position_size * 100
            
            trade['exit_reason'] = 'Market close'
        
        # Update portfolio value
        self.portfolio_value += trade['pnl']
        
        # Store trade
        self.trades.append(trade)
        
        logging.info(f"Trade: {signal['ticker']} {signal['strike']}{signal['right']} - "
                    f"Entry: ${entry_price:.2f}, Exit: ${trade['exit_price']:.2f}, "
                    f"P&L: ${trade['pnl']:.2f}, Reason: {trade['exit_reason']}")
    
    def _load_signal_data(self, signal: Dict) -> pd.DataFrame:
        """
        Load historical tick data for the signal with UNIVERSAL COLUMN HANDLING.
        """
        ticker = signal['ticker']
        expiry = signal['expiry']
        strike = signal['strike']
        right = signal['right']
        
        # Generate the filename
        filename = get_data_filename_databento(ticker, expiry, strike, right)
        filepath = os.path.join(self.historical_data_dir, filename)
        
        if not os.path.exists(filepath):
            logging.warning(f"Historical data not found: {filepath}")
            return pd.DataFrame()
        
        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            
            # ===== UNIVERSAL COLUMN HANDLER FIX =====
            # Check which timestamp column exists and normalize to 'ts_event'
            if 'timestamp' in df.columns and 'ts_event' not in df.columns:
                # Rename 'timestamp' to 'ts_event' for consistency
                df = df.rename(columns={'timestamp': 'ts_event'})
                logging.debug(f"Renamed 'timestamp' column to 'ts_event' for {filename}")
            elif 'ts_event' not in df.columns and 'timestamp' not in df.columns:
                # Neither column exists - this is a problem
                logging.error(f"No timestamp column found in {filename}. Columns: {df.columns.tolist()}")
                return pd.DataFrame()
            # If 'ts_event' already exists, we're good - no action needed
            
            # Ensure ts_event is datetime
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            
            # Filter to only include data from signal time onwards
            signal_time = pd.to_datetime(signal['timestamp'])
            df = df[df['ts_event'] >= signal_time].copy()
            
            # Sort by timestamp
            df = df.sort_values('ts_event').reset_index(drop=True)
            
            # Add signal reference
            df['signal_id'] = signal.get('id', 'unknown')
            
            logging.info(f"Loaded {len(df)} ticks for {ticker} {strike}{right}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from {filepath}: {e}")
            return pd.DataFrame()
    
    def _get_entry_price_from_data(self, df: pd.DataFrame) -> Optional[float]:
        """
        Get entry price from the first valid tick in the data.
        """
        if df.empty:
            return None
        
        # Try to get price from first row with valid bid/ask
        for idx, row in df.head(10).iterrows():
            if 'bid' in row and 'ask' in row:
                bid = row['bid']
                ask = row['ask']
                if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
                    return (bid + ask) / 2
            elif 'close' in row:
                close = row['close']
                if not pd.isna(close) and close > 0:
                    return close
        
        # If no valid price found in first 10 ticks, return None
        return None
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from trades.
        """
        if not self.trades:
            return self._generate_empty_results()
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0
        
        # Calculate return percentage
        return_pct = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'return_pct': return_pct,
            'final_portfolio': self.portfolio_value,
            'trades': self.trades
        }
    
    def _generate_empty_results(self) -> Dict:
        """
        Generate empty results structure.
        """
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'return_pct': 0,
            'final_portfolio': self.initial_capital,
            'trades': []
        }