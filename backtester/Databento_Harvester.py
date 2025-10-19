#!/usr/bin/env python3
"""
Databento_Harvester.py - PROPER VERSION WITH BID/ASK
This is what should have been provided from the start - gets bid/ask spreads for realistic backtesting
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
    Downloads historical options data from Databento WITH BID/ASK SPREADS.
    This is essential for realistic options backtesting.
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
                script_dir / 'signals_to_test.txt',
                project_root / 'backtester' / 'signals_to_test.txt',
                Path('backtester/signals_to_test.txt'),
                Path('signals_to_test.txt'),
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.signals_path = path
                    break
            else:
                self.signals_path = script_dir / 'signals_to_test.txt'
        
        # Handle output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = script_dir / 'historical_data'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Signals path: {self.signals_path}")
        logging.info(f"Output directory: {self.output_dir}")
    
    def run(self):
        """Main execution loop"""
        signals = self._load_signals()
        
        if not signals:
            logging.error("No valid signals found")
            return
        
        logging.info(f"\n{'='*60}")
        logging.info(f"DOWNLOADING DATA FOR {len(signals)} SIGNALS")
        logging.info(f"{'='*60}\n")
        
        success_count = 0
        failed_count = 0
        
        for i, signal in enumerate(signals, 1):
            logging.info(f"\n[{i}/{len(signals)}] Processing signal...")
            if self._download_signal_data(signal):
                success_count += 1
            else:
                failed_count += 1
        
        logging.info(f"\n{'='*60}")
        logging.info(f"DOWNLOAD COMPLETE")
        logging.info(f"✅ Success: {success_count} | ❌ Failed: {failed_count}")
        logging.info(f"{'='*60}")
    
    def _load_signals(self):
        """Parse signals from file"""
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
            
            # Handle timestamped format
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
        """
        Download data for a single signal WITH PROPER BID/ASK QUOTES.
        This is the correct implementation that should have been provided initially.
        """
        ticker = signal['ticker']
        expiry = signal['expiry_date']
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
            return False
        
        # Build OCC symbol
        # CRITICAL: SPX options use SPXW root symbol
        root_symbol = 'SPXW' if ticker.upper() == 'SPX' else ticker
        
        # Format: AAAAAA YYMMDD C/P SSSSSSSS
        expiry_str = exp_date.strftime('%y%m%d')
        ticker_padded = root_symbol.ljust(6)
        strike_str = f"{int(strike * 1000):08d}"
        occ_symbol = f"{ticker_padded}{expiry_str}{right}{strike_str}"
        
        logging.info(f"   OCC Symbol: {occ_symbol}")
        
        # Date range for data
        start_date = exp_date
        end_date = exp_date + timedelta(days=1)
        
        try:
            # ============================================
            # GET BID/ASK QUOTES - THIS IS CRITICAL!
            # ============================================
            logging.info(f"   Fetching BID/ASK quotes...")
            
            # Try MBP-1 first (Market By Price - includes bid/ask)
            try:
                quote_data = self.client.timeseries.get_range(
                    dataset='OPRA.pillar',
                    symbols=[occ_symbol],
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    schema='mbp-1',  # Market By Price - includes bid/ask/trades
                    stype_in='raw_symbol'
                )
            except Exception as e:
                # If MBP-1 fails, try TBBO (Top Book Bid/Offer)
                logging.info(f"   MBP-1 failed, trying TBBO schema...")
                quote_data = self.client.timeseries.get_range(
                    dataset='OPRA.pillar',
                    symbols=[occ_symbol],
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    schema='tbbo',  # Top Book Bid/Offer
                    stype_in='raw_symbol'
                )
            
            # Also get trade data for actual execution prices
            logging.info(f"   Fetching trade data...")
            trade_data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='trades',
                stype_in='raw_symbol'
            )
            
            # Convert to DataFrames
            quotes_df = quote_data.to_df() if quote_data else pd.DataFrame()
            trades_df = trade_data.to_df() if trade_data else pd.DataFrame()
            
            if quotes_df.empty and trades_df.empty:
                logging.warning(f"   ⚠️ No data found - option may not have traded")
                return False
            
            # Process quote data (bid/ask)
            if not quotes_df.empty:
                quotes_df = quotes_df.reset_index()
                
                # Handle different schema column names
                if 'bid_px_01' in quotes_df.columns:  # MBP-1 schema
                    quotes_df = quotes_df.rename(columns={
                        'ts_event': 'timestamp',
                        'bid_px_01': 'bid',
                        'ask_px_01': 'ask',
                        'bid_sz_01': 'bid_size',
                        'ask_sz_01': 'ask_size'
                    })
                elif 'bid_px_00' in quotes_df.columns:  # TBBO schema
                    quotes_df = quotes_df.rename(columns={
                        'ts_event': 'timestamp',
                        'bid_px_00': 'bid',
                        'ask_px_00': 'ask',
                        'bid_sz_00': 'bid_size',
                        'ask_sz_00': 'ask_size'
                    })
                else:
                    logging.warning("   ⚠️ Unknown quote schema columns")
                    # Try to find bid/ask columns
                    bid_cols = [col for col in quotes_df.columns if 'bid_px' in col]
                    ask_cols = [col for col in quotes_df.columns if 'ask_px' in col]
                    if bid_cols and ask_cols:
                        quotes_df = quotes_df.rename(columns={
                            'ts_event': 'timestamp',
                            bid_cols[0]: 'bid',
                            ask_cols[0]: 'ask'
                        })
                
                # Calculate mid price and spread
                if 'bid' in quotes_df.columns and 'ask' in quotes_df.columns:
                    quotes_df['mid'] = (quotes_df['bid'] + quotes_df['ask']) / 2
                    quotes_df['spread'] = quotes_df['ask'] - quotes_df['bid']
                    
                    # Log statistics
                    avg_spread = quotes_df['spread'].mean()
                    max_spread = quotes_df['spread'].max()
                    logging.info(f"   📊 Bid/Ask Stats:")
                    logging.info(f"      Avg spread: ${avg_spread:.2f}")
                    logging.info(f"      Max spread: ${max_spread:.2f}")
                    logging.info(f"      Bid range: ${quotes_df['bid'].min():.2f} - ${quotes_df['bid'].max():.2f}")
                    logging.info(f"      Ask range: ${quotes_df['ask'].min():.2f} - ${quotes_df['ask'].max():.2f}")
            
            # Process trade data
            if not trades_df.empty:
                trades_df = trades_df.reset_index()
                trades_df = trades_df.rename(columns={
                    'ts_event': 'timestamp',
                    'price': 'trade_price',
                    'size': 'trade_size'
                })
                
                logging.info(f"   📊 Trade Stats:")
                logging.info(f"      Total trades: {len(trades_df):,}")
                logging.info(f"      Price range: ${trades_df['trade_price'].min():.2f} - ${trades_df['trade_price'].max():.2f}")
            
            # Merge quotes and trades
            final_df = pd.DataFrame()
            
            if not quotes_df.empty and not trades_df.empty:
                # Merge on nearest timestamp
                quotes_df['timestamp'] = pd.to_datetime(quotes_df['timestamp'])
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # For each trade, find the most recent quote
                final_df = pd.merge_asof(
                    trades_df.sort_values('timestamp'),
                    quotes_df[['timestamp', 'bid', 'ask', 'mid', 'spread']].sort_values('timestamp'),
                    on='timestamp',
                    direction='backward'
                )
                
                # Add volume and price info
                final_df['close'] = final_df['trade_price']
                final_df['volume'] = final_df['trade_size']
                
            elif not quotes_df.empty:
                # Only quotes available
                final_df = quotes_df.copy()
                final_df['close'] = final_df['mid'] if 'mid' in final_df.columns else final_df['bid']
                final_df['volume'] = 0
                
            elif not trades_df.empty:
                # Only trades available (no bid/ask - not ideal!)
                logging.warning("   ⚠️ No bid/ask data available - using trades only")
                final_df = trades_df.copy()
                final_df['close'] = final_df['trade_price']
                final_df['volume'] = final_df['trade_size']
                # Estimate bid/ask from trade price (rough approximation)
                final_df['bid'] = final_df['trade_price'] - 0.05
                final_df['ask'] = final_df['trade_price'] + 0.05
                final_df['mid'] = final_df['trade_price']
                final_df['spread'] = 0.10
            
            # Ensure we have timestamp column
            if 'timestamp' not in final_df.columns and final_df.index.name == 'timestamp':
                final_df = final_df.reset_index()
            
            # Add high/low for compatibility
            if 'close' in final_df.columns:
                final_df['high'] = final_df['close']
                final_df['low'] = final_df['close']
            
            # Save to CSV with all important columns
            if not final_df.empty:
                expiry_str_file = exp_date.strftime('%Y%m%d')
                filename = get_data_filename_databento(ticker, expiry_str_file, strike, right)
                filepath = self.output_dir / filename
                
                # Columns to save (including bid/ask!)
                save_columns = ['timestamp', 'bid', 'ask', 'mid', 'spread', 'close', 'high', 'low', 'volume']
                
                # Only save columns that exist
                save_columns = [col for col in save_columns if col in final_df.columns]
                
                final_df[save_columns].to_csv(filepath, index=False)
                
                logging.info(f"   ✅ Saved {len(final_df):,} rows to {filename}")
                logging.info(f"      Columns: {', '.join(save_columns)}")
                
                # Verify bid/ask data was saved
                if 'bid' in save_columns and 'ask' in save_columns:
                    logging.info(f"   ✅ BID/ASK data included!")
                else:
                    logging.warning(f"   ⚠️ WARNING: No bid/ask data in output file!")
                
                return True
            else:
                logging.warning(f"   ⚠️ No data to save")
                return False
                
        except Exception as e:
            logging.error(f"   ❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False


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
    parser = argparse.ArgumentParser(description='Download historical options data from Databento WITH BID/ASK')
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
    print("DATABENTO DATA HARVESTER - WITH BID/ASK")
    print("="*60)
    print()
    
    # Create harvester
    harvester = DatabentoHarvester(
        api_key,
        signals_path=args.signals,
        output_dir=args.output
    )
    
    # Run the harvester
    harvester.run()
    
    print("\n" + "="*60)
    print("COMPLETE - Check that files include bid/ask columns!")
    print("="*60)


if __name__ == "__main__":
    main()
