#!/usr/bin/env python3
"""
Databento Options Data Harvester - FINAL WORKING VERSION
Fetches data from SIGNAL DATE through EXPIRY DATE
Uses environment variable from .env file
Column output: timestamp (not ts_event) - backtest engine handles both
"""

import databento as db
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.signal_parser import SignalParser
from services.config import Config
from services.utils import get_data_filename_databento

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class DatabentoHarvester:
    def __init__(self, api_key):
        self.client = db.Historical(api_key)
        
        # Use Path for Windows compatibility
        self.signals_path = Path(__file__).parent / "signals_to_test.txt"
        self.output_dir = Path(__file__).parent / "historical_data"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
    def run(self):
        """Main execution"""
        logging.info(f"Using signals file: {self.signals_path}")
        logging.info("Starting Databento data download...")
        
        signals = self._parse_signals()
        
        if not signals:
            logging.error("No valid signals found")
            return
            
        logging.info(f"Found {len(signals)} valid signals")
        
        for signal in signals:
            self._download_signal_data(signal)
    
    def _parse_signals(self):
        """Parse signals from file"""
        signals = []
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        if not self.signals_path.exists():
            logging.error(f"Signals file not found: {self.signals_path}")
            return []
        
        with open(self.signals_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle both formats:
                # 1. Simple: TICKER STRIKEP/C MM/DD
                # 2. Full: YYYY-MM-DD HH:MM:SS | channel | TICKER STRIKEP/C MM/DD
                timestamp_str = None
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        timestamp_str = parts[0].strip()
                        signal_text = parts[2].strip()
                    else:
                        continue
                else:
                    signal_text = line
                
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                if parsed:
                    # Add signal timestamp if available
                    if timestamp_str:
                        parsed['signal_timestamp'] = timestamp_str
                    
                    signals.append(parsed)
                    
                    # Use correct keys from signal parser
                    right = parsed['contract_type'][0]
                    expiry = parsed.get('expiry_date', 'unknown')
                    logging.info(f"✓ {parsed['ticker']} {parsed['strike']}{right} {expiry}")
        
        return signals
    
    def _download_signal_data(self, signal):
        """Download data for a single signal from SIGNAL DATE to EXPIRY"""
        ticker = signal['ticker']
        strike = signal['strike']
        right = signal['contract_type'][0]  # 'CALL' -> 'C', 'PUT' -> 'P'
        
        # Signal parser returns 'expiry_date' (YYYYMMDD string)
        expiry = signal.get('expiry_date', '')
        
        logging.info(f"📥 {ticker} {strike}{right} exp {expiry}")
        
        try:
            # Parse expiry date
            exp_date = datetime.strptime(expiry, '%Y%m%d').date()
            
            # Parse signal date if available, otherwise use 2 days before expiry
            if 'signal_timestamp' in signal:
                signal_date = datetime.strptime(signal['signal_timestamp'], '%Y-%m-%d %H:%M:%S').date()
            else:
                signal_date = exp_date - timedelta(days=2)
            
            # Create OCC symbol (YYMMDD format for Databento)
            expiry_str = exp_date.strftime('%y%m%d')
            strike_str = f"{int(strike * 1000):08d}"
            
            # Handle special index tickers
            if ticker == "SPX":
                ticker_formatted = "SPXW  "  # SPXW for SPX
            elif ticker == "NDX":
                ticker_formatted = "NDXP  "  # NDXP for NDX
            else:
                ticker_formatted = ticker.ljust(6)[:6]
            
            occ_symbol = f"{ticker_formatted}{expiry_str}{right}{strike_str}"
            
            logging.info(f"   OCC: {occ_symbol}")
            logging.info(f"   Fetching: {signal_date} to {exp_date}")
            
            # Fetch data from signal date through expiry (inclusive)
            data = self.client.timeseries.get_range(
                dataset='OPRA.PILLAR',
                symbols=[occ_symbol],
                schema='trades',
                start=signal_date,
                end=exp_date + timedelta(days=1),  # End is exclusive, so add 1 day
                stype_in='raw_symbol'
            )
            
            df = data.to_df()
            
            if df.empty:
                logging.error(f"   ❌ No data returned for {occ_symbol}")
                return
            
            # Rename columns for backtest engine compatibility
            # Keep as 'timestamp' - the backtest engine fix handles both timestamp and ts_event
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'price': 'close'
            })
            
            # Add high/low if not present
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = df.get('size', 0)
            
            # Add bid/ask/mid/spread if not present (from trades data)
            if 'bid' not in df.columns:
                df['bid'] = df['close'] * 0.995  # Approximate
            if 'ask' not in df.columns:
                df['ask'] = df['close'] * 1.005  # Approximate
            if 'mid' not in df.columns:
                df['mid'] = df['close']
            if 'spread' not in df.columns:
                df['spread'] = df['ask'] - df['bid']
            
            # Save to CSV with all columns
            filename = get_data_filename_databento(ticker, expiry, strike, right)
            filepath = self.output_dir / filename
            
            # Save with proper column order
            columns_to_save = ['timestamp', 'bid', 'ask', 'mid', 'spread', 'close', 'high', 'low', 'volume']
            df[columns_to_save].to_csv(filepath, index=False)
            
            # Log statistics
            date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            price_range = f"${df['close'].min():.2f}-${df['close'].max():.2f}"
            logging.info(f"   ✅ Saved {len(df):,} ticks")
            logging.info(f"   📅 Date range: {signal_date} to {exp_date}")
            logging.info(f"   💰 Price range: {price_range}")
            
        except Exception as e:
            logging.error(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Load .env file
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', help='Databento API key (or use DATABENTO_API_KEY in .env)')
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        print("=" * 60)
        print("ERROR: API key required")
        print("=" * 60)
        print("Option 1: Add to .env file:")
        print("   DATABENTO_API_KEY=your_key_here")
        print("")
        print("Option 2: Pass as argument:")
        print("   python Databento_Harvester.py --api-key YOUR_KEY")
        print("=" * 60)
        sys.exit(1)
    
    print("=" * 60)
    print("DATABENTO DATA HARVESTER")
    print("=" * 60)
    
    harvester = DatabentoHarvester(api_key)
    harvester.run()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
