#!/usr/bin/env python3
"""
Databento Harvester - ACTUALLY FINAL VERSION
Uses correct signal parser keys: contract_type, expiry_date
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
        self.signals_path = Path(__file__).parent / "signals_to_test.txt"
        self.output_dir = Path(__file__).parent / "historical_data"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
    def run(self):
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
        """Parse signals with correct key mapping"""
        signals = []
        
        # Create default profile for parsing
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True,
            'buzzwords_buy': [],
            'buzzwords_sell': [],
            'channel_id': 'harvester'
        }
        
        if not self.signals_path.exists():
            logging.error(f"Signals file not found: {self.signals_path}")
            return []
        
        with open(self.signals_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:') or line.startswith('Format:'):
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    signal_text = parts[2].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    signal_text = line
                
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                if parsed:
                    # Add timestamp to parsed signal
                    parsed['timestamp'] = timestamp_str
                    
                    # Map parser keys to harvester keys for consistency
                    # Parser returns: contract_type="CALL"/"PUT", expiry_date="YYYYMMDD"
                    parsed['right'] = 'C' if parsed['contract_type'] == 'CALL' else 'P'
                    parsed['expiry'] = parsed['expiry_date']
                    
                    signals.append(parsed)
                    logging.info(f"✓ {parsed['ticker']} {parsed['strike']}{parsed['right']} {parsed['expiry']}")
        
        return signals
    
    def _download_signal_data(self, signal):
        """Download data for a single signal"""
        ticker = signal['ticker']
        strike = signal['strike']
        right = signal['right']  # Already converted to 'C' or 'P'
        expiry = signal['expiry']  # Already in YYYYMMDD format
        signal_timestamp = signal['timestamp']
        
        logging.info(f"\n📥 {ticker} {strike}{right} exp {expiry}")
        
        try:
            signal_date = datetime.strptime(signal_timestamp, '%Y-%m-%d %H:%M:%S').date()
            exp_date = datetime.strptime(expiry, '%Y%m%d').date()
            
            # ✅ CRITICAL FIX: Day trader - ONLY download signal date!
            start_date = signal_date
            end_date = signal_date + timedelta(days=1)  # Just one day
            
            logging.info(f"   Date: {signal_date} (signal day only)")
            
            # Build OCC symbol - YYMMDD format
            expiry_str = exp_date.strftime('%y%m%d')
            strike_str = f"{int(strike * 1000):08d}"
            
            # Convert SPX → SPXW, NDX → NDXP
            root_symbol = ticker.upper()
            if root_symbol == 'SPX':
                root_symbol = 'SPXW'
            elif root_symbol == 'NDX':
                root_symbol = 'NDXP'
            
            occ_symbol = f"{root_symbol.ljust(6)}{expiry_str}{right}{strike_str}"
            logging.info(f"   OCC: {occ_symbol}")
            
            # Download from Databento - 1-MINUTE BARS (not ticks!)
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                schema='ohlcv-1m',  # ✅ FIXED: 1-minute OHLCV bars
                start=start_date,
                end=end_date,
                stype_in='raw_symbol'
            )
            
            df = data.to_df()
            if df.empty:
                logging.warning(f"   ⚠️ No data found")
                return
            
            # OHLCV data has timestamp in the INDEX, not as a column
            # Columns: ['rtype', 'publisher_id', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Rename index column to 'timestamp' (it's usually called 'ts_event')
            if 'ts_event' in df.columns:
                df = df.rename(columns={'ts_event': 'timestamp'})
            elif df.columns[0] not in ['timestamp', 'rtype']:
                # First column is the timestamp
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # For OHLCV bars, approximate bid/ask from close
            df['mid'] = df['close']
            df['spread'] = 0.01
            df['bid'] = df['close'] - 0.005
            df['ask'] = df['close'] + 0.005
            
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
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', help='Databento API key')
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    if not api_key:
        print("=" * 60)
        print("ERROR: DATABENTO_API_KEY not found!")
        print("Add to .env file: DATABENTO_API_KEY=your_key_here")
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
