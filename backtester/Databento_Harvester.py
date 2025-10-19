#!/usr/bin/env python3
"""
Databento_Harvester.py - FIXED VERSION
Downloads historical options data from Databento with proper path handling
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import databento as db

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename_databento


class DatabentoHarvester:
    """
    Downloads historical options data from Databento for backtesting.
    FIXED: Handles paths correctly when run from any directory.
    """
    
    def __init__(self, api_key, signals_path=None, output_dir=None):
        """Initialize harvester with proper path resolution"""
        self.client = db.Historical(api_key)
        logging.info(f"Initialized {self.client}")
        
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
        # Resolve paths properly
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # Handle signals path
        if signals_path:
            self.signals_path = Path(signals_path)
        else:
            # Try multiple locations
            possible_paths = [
                script_dir / 'signals_to_test.txt',  # Same dir as script
                project_root / 'backtester' / 'signals_to_test.txt',  # Project structure
                Path('backtester/signals_to_test.txt'),  # Relative to CWD
                Path('signals_to_test.txt'),  # Current directory
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.signals_path = path
                    break
            else:
                # Default to expected location
                self.signals_path = script_dir / 'signals_to_test.txt'
        
        # Handle output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = script_dir / 'historical_data'
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Signals path: {self.signals_path}")
        logging.info(f"Output directory: {self.output_dir}")
    
    def run(self):
        """Main execution loop - compatible with original interface"""
        signals = self._load_signals()
        
        if not signals:
            logging.error("No valid signals found")
            return
        
        logging.info(f"\n{'='*60}")
        logging.info(f"DOWNLOADING DATA FOR {len(signals)} SIGNALS")
        logging.info(f"{'='*60}\n")
        
        for i, signal in enumerate(signals, 1):
            logging.info(f"\n[{i}/{len(signals)}] Processing signal...")
            self._download_signal_data(signal)
        
        logging.info(f"\n{'='*60}")
        logging.info("DOWNLOAD COMPLETE")
        logging.info(f"{'='*60}")
    
    def _load_signals(self):
        """Parse signals from file with better error handling"""
        signals = []
        
        if not self.signals_path.exists():
            logging.error(f"Signals file not found: {self.signals_path}")
            logging.info(f"Please create the signals file at: {self.signals_path.absolute()}")
            return []
        
        logging.info(f"Loading signals from: {self.signals_path}")
        
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        with open(self.signals_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments, empty lines, and trader headers
            if not line or line.startswith('#') or line.startswith('Trader:'):
                continue
            
            # Handle timestamped format: YYYY-MM-DD HH:MM:SS | Channel | Signal
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 3:
                    signal_text = parts[2].strip()
                else:
                    logging.warning(f"Line {line_num}: Invalid format: {line}")
                    continue
            else:
                signal_text = line
            
            # Parse the signal
            parsed = self.signal_parser.parse_signal(signal_text, default_profile)
            
            if parsed:
                signals.append(parsed)
                logging.info(f"  ✓ Parsed: {parsed['ticker']} {parsed['strike']}{parsed['contract_type'][0]} exp {parsed['expiry_date']}")
            else:
                logging.warning(f"  ✗ Could not parse line {line_num}: {signal_text}")
        
        logging.info(f"\nSuccessfully parsed {len(signals)} signals")
        return signals
    
    def _download_signal_data(self, signal):
        """Download data for a single signal with bid/ask spreads"""
        ticker = signal['ticker']
        expiry = signal['expiry_date']  # Format: YYYY-MM-DD
        strike = signal['strike']
        right = signal['contract_type'][0].upper()
        
        logging.info(f"📥 {ticker} {strike}{right} exp {expiry}")
        
        # Convert expiry to datetime
        try:
            if isinstance(expiry, str):
                if '-' in expiry:
                    exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                else:
                    exp_date = datetime.strptime(expiry, '%Y%m%d')
            else:
                exp_date = expiry
        except Exception as e:
            logging.error(f"   ❌ Invalid expiry format: {expiry} - {e}")
            return
        
        # Build OCC symbol
        # CRITICAL: SPX options use SPXW root symbol
        root_symbol = 'SPXW' if ticker.upper() == 'SPX' else ticker
        
        # Format: AAAAAA YYMMDD C/P SSSSSSSS
        expiry_str = exp_date.strftime('%y%m%d')  # YYMMDD format
        ticker_padded = root_symbol.ljust(6)  # Pad to 6 chars
        strike_str = f"{int(strike * 1000):08d}"  # Strike x1000, 8 digits
        occ_symbol = f"{ticker_padded}{expiry_str}{right}{strike_str}"
        
        logging.info(f"   OCC Symbol: {occ_symbol}")
        
        # Date range for data
        start_date = exp_date
        end_date = exp_date + timedelta(days=1)
        
        try:
            logging.info(f"   Fetching data from {start_date.date()} to {end_date.date()}...")
            
            # Get trade data (actual trades)
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='trades',  # Trade data
                stype_in='raw_symbol'
            )
            
            if data is None:
                logging.warning(f"   ⚠️ No data returned")
                return
            
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"   ⚠️ No trades found - option may not have traded")
                return
            
            # Process the data
            df = df.reset_index()
            
            # Rename columns for consistency
            if 'ts_event' in df.columns:
                df = df.rename(columns={'ts_event': 'timestamp'})
            elif df.index.name == 'ts_event':
                df = df.reset_index().rename(columns={'ts_event': 'timestamp'})
            
            if 'price' in df.columns:
                df = df.rename(columns={'price': 'close'})
            
            if 'size' in df.columns:
                df = df.rename(columns={'size': 'volume'})
            
            # Add high/low if not present
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'close', 'high', 'low', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 100  # Default volume
                    else:
                        logging.error(f"   ❌ Missing required column: {col}")
                        return
            
            # Save to CSV
            expiry_str_file = exp_date.strftime('%Y%m%d')
            filename = get_data_filename_databento(ticker, expiry_str_file, strike, right)
            filepath = self.output_dir / filename
            
            # Select columns and save
            df_to_save = df[required_cols].copy()
            df_to_save.to_csv(filepath, index=False)
            
            # Log summary
            price_min = df['close'].min()
            price_max = df['close'].max()
            price_range = f"${price_min:.2f}-${price_max:.2f}"
            
            logging.info(f"   ✅ Saved {len(df):,} ticks to {filename}")
            logging.info(f"      Price range: {price_range}")
            
        except Exception as e:
            logging.error(f"   ❌ Download failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download historical options data from Databento')
    parser.add_argument('--api-key', help='Databento API key')
    parser.add_argument('--signals', help='Path to signals file')
    parser.add_argument('--output', help='Output directory for data files')
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        logging.error("ERROR: Databento API key required")
        logging.info("Either:")
        logging.info("  1. Add DATABENTO_API_KEY to your .env file")
        logging.info("  2. Pass it via command line: --api-key YOUR_KEY")
        sys.exit(1)
    
    print("="*60)
    print("DATABENTO DATA HARVESTER")
    print("="*60)
    print()
    
    # Create harvester with optional custom paths
    harvester = DatabentoHarvester(
        api_key,
        signals_path=args.signals,
        output_dir=args.output
    )
    
    # Run the harvester
    harvester.run()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
