#!/usr/bin/env python3
"""
Databento Options Data Harvester - COMPLETE WORKING VERSION
Fetches data from SIGNAL DATE through EXPIRY DATE
Uses environment variable from .env file
Column output: timestamp,bid,ask,mid,spread,close,high,low,volume
"""

import databento as db
import pandas as pd
import os
import sys
from datetime import datetime, timedelta  # ✅ CRITICAL FIX: Added timedelta
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
                # 1. Simple: MSFT 515P 10/10
                # 2. Timestamped: 2025-10-07 07:46:43 | Expo | MSFT 515P 10/10
                
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    channel = parts[1].strip()
                    signal_text = parts[2].strip()
                else:
                    # Use current time for simple format
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    channel = 'test_server'
                    signal_text = line
                
                # Parse signal
                parsed = self.signal_parser.parse_signal(signal_text, channel)
                
                if parsed:
                    parsed['timestamp'] = timestamp_str
                    signals.append(parsed)
                    logging.info(f"✓ {parsed['ticker']} {parsed['strike']}{parsed['right']} {parsed['expiry']}")
        
        return signals
    
    def _download_signal_data(self, signal):
        """Download historical data for a single signal"""
        ticker = signal['ticker']
        strike = signal['strike']
        right = signal['right']
        expiry = signal['expiry']
        signal_timestamp = signal['timestamp']
        
        logging.info(f"\n📥 {ticker} {strike}{right} exp {expiry}")
        
        try:
            # Parse dates
            signal_date = datetime.strptime(signal_timestamp, '%Y-%m-%d %H:%M:%S').date()
            exp_date = datetime.strptime(expiry, '%Y%m%d').date()
            
            # Date range: signal date → expiry (inclusive)
            start_date = signal_date
            end_date = exp_date + timedelta(days=1)  # ✅ USES timedelta HERE
            
            # Build OCC symbol - YYMMDD format (2-digit year)
            expiry_str = exp_date.strftime('%y%m%d')
            strike_str = f"{int(strike * 1000):08d}"
            
            # Convert SPX → SPXW, NDX → NDXP for Databento
            root_symbol = ticker.upper()
            if root_symbol == 'SPX':
                root_symbol = 'SPXW'
            elif root_symbol == 'NDX':
                root_symbol = 'NDXP'
            
            occ_symbol = f"{root_symbol.ljust(6)}{expiry_str}{right}{strike_str}"
            
            logging.info(f"   OCC: {occ_symbol}")
            
            # Download from Databento (with bid/ask data)
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                schema='mbp-1',  # ✅ CORRECT: mbp-1 has bid/ask
                start=start_date,
                end=end_date,
                stype_in='raw_symbol'
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"   ⚠️ No data found")
                return
            
            # Calculate derived columns
            df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
            df['spread'] = df['ask_px_00'] - df['bid_px_00']
            
            # Rename and select columns to match PLTR format
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'bid_px_00': 'bid',
                'ask_px_00': 'ask'
            })
            
            # Add OHLCV if not present
            if 'close' not in df.columns:
                df['close'] = df['mid']
            if 'high' not in df.columns:
                df['high'] = df['mid']
            if 'low' not in df.columns:
                df['low'] = df['mid']
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Select final columns
            columns = ['timestamp', 'bid', 'ask', 'mid', 'spread', 'close', 'high', 'low', 'volume']
            df = df[columns]
            
            # Save to CSV
            filename = get_data_filename_databento(ticker, expiry, strike, right)
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False)
            
            logging.info(f"   ✅ Saved {len(df):,} ticks → {filename}")
            
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
    
    # Get API key
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        print("=" * 60)
        print("ERROR: DATABENTO_API_KEY not found!")
        print("")
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
