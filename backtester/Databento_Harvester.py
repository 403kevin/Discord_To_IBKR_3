#!/usr/bin/env python3
"""
Databento Options Data Harvester - WORKING VERSION
Uses environment variable from .env file
Windows compatible with Path objects
"""

import databento as db
import pandas as pd
import os
import sys
from datetime import datetime
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
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        signal_text = parts[2].strip()
                    else:
                        continue
                else:
                    signal_text = line
                
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                if parsed:
                    signals.append(parsed)
                    logging.info(f"✓ {parsed['ticker']} {parsed['strike']}{parsed['contract_type'][0]} {parsed['expiry']}")
        
        return signals
    
    def _download_signal_data(self, signal):
        """Download data for a single signal"""
        ticker = signal['ticker']
        strike = signal['strike']
        right = signal['contract_type'][0]  # 'CALL' -> 'C', 'PUT' -> 'P'
        expiry = signal['expiry']
        
        logging.info(f"📥 {ticker} {strike}{right} exp {expiry}")
        
        try:
            # Parse expiry date
            exp_date = datetime.strptime(expiry, '%Y%m%d').date()
            
            # Create OCC symbol (YYMMDD format for Databento)
            expiry_str = exp_date.strftime('%y%m%d')
            strike_str = f"{int(strike * 1000):08d}"
            occ_symbol = f"{ticker}{expiry_str}{right}{strike_str}"
            
            logging.info(f"   OCC: {occ_symbol}")
            
            # Fetch data - use trading day range
            # For day trading: get data from 2 days before expiry through expiry
            start_date = exp_date - timedelta(days=2)
            end_date = exp_date + timedelta(days=1)
            
            logging.info(f"   Fetching: {start_date} to {end_date}")
            
            data = self.client.timeseries.get_range(
                dataset='OPRA.PILLAR',
                symbols=[occ_symbol],
                schema='trades',
                start=start_date,
                end=end_date,
                stype_in='raw_symbol'
            )
            
            df = data.to_df()
            
            if df.empty:
                logging.error(f"   ❌ No data returned")
                return
            
            # Rename columns for backtest engine compatibility
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
            
            # Save to CSV
            filename = get_data_filename_databento(ticker, expiry, strike, right)
            filepath = self.output_dir / filename
            
            df[['timestamp', 'close', 'high', 'low', 'volume']].to_csv(filepath, index=False)
            
            price_range = f"${df['close'].min():.2f}-${df['close'].max():.2f}"
            logging.info(f"   ✅ Saved {len(df):,} ticks | Range: {price_range}")
            
        except Exception as e:
            logging.error(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
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
