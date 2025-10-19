#!/usr/bin/env python3
"""
databento_harvester_PROPER.py - Correct data harvesting for options backtesting
Gets bid/ask spreads, proper trade data, and handles options correctly
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import databento as db
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.signal_parser import SignalParser
from services.config import Config


class ProperDatabentoHarvester:
    """
    Harvests PROPER options data with bid/ask spreads and accurate pricing
    """
    
    def __init__(self, api_key):
        self.client = db.Historical(api_key)
        self.output_dir = Path("backtester/historical_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # For signal parsing
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
        logging.info("Initialized Proper Databento Harvester")
    
    def harvest_signal(self, signal_line):
        """
        Harvest data for a single signal with proper bid/ask data
        """
        # Parse signal
        if '|' not in signal_line:
            logging.warning(f"Invalid signal format: {signal_line}")
            return
        
        parts = signal_line.split('|')
        if len(parts) != 3:
            return
        
        timestamp_str = parts[0].strip()
        trader = parts[1].strip()
        signal_text = parts[2].strip()
        
        # Parse signal
        profile = {'assume_buy_on_ambiguous': True, 'ambiguous_expiry_enabled': True}
        signal = self.signal_parser.parse_signal(signal_text, profile)
        
        if not signal:
            logging.warning(f"Could not parse: {signal_text}")
            return
        
        # Extract components
        ticker = signal['ticker']
        expiry = signal['expiry_date']  # Format: YYYY-MM-DD
        strike = signal['strike']
        right = signal['contract_type'][0].upper()
        
        logging.info(f"\n📥 Harvesting: {ticker} {strike}{right} exp {expiry}")
        
        # Convert expiry to datetime
        if isinstance(expiry, str):
            if '-' in expiry:
                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
            else:
                exp_date = datetime.strptime(expiry, '%Y%m%d')
        else:
            exp_date = expiry
        
        # Build OCC symbol
        # CRITICAL: SPX options use SPXW root
        root_symbol = 'SPXW' if ticker.upper() == 'SPX' else ticker
        
        # Format: AAAAAA YYMMDD C/P SSSSSSSS (8 digits for strike)
        expiry_str = exp_date.strftime('%y%m%d')  
        ticker_padded = root_symbol.ljust(6)
        strike_str = f"{int(strike * 1000):08d}"
        occ_symbol = f"{ticker_padded}{expiry_str}{right}{strike_str}"
        
        logging.info(f"   OCC Symbol: {occ_symbol}")
        
        # Determine date range
        # For 0DTE, get data for expiry day plus buffer
        start_date = exp_date - timedelta(hours=1)  # Small buffer before market open
        end_date = exp_date + timedelta(days=1)     # Through end of expiry day
        
        try:
            logging.info(f"   Fetching BBO (Best Bid/Offer) data...")
            
            # Get BBO data - this gives us bid/ask spreads
            bbo_data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='tbbo',  # Top Book Bid/Offer - CORRECT for options
                stype_in='raw_symbol'
            )
            
            # Also get trades for actual execution prices
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
            bbo_df = bbo_data.to_df() if bbo_data else pd.DataFrame()
            trade_df = trade_data.to_df() if trade_data else pd.DataFrame()
            
            if bbo_df.empty and trade_df.empty:
                logging.warning(f"   ⚠️ No data found - symbol may not exist")
                return
            
            # Process BBO data
            if not bbo_df.empty:
                bbo_df = bbo_df.reset_index()
                
                # Rename columns for clarity
                bbo_df = bbo_df.rename(columns={
                    'ts_event': 'timestamp',
                    'bid_px_00': 'bid',
                    'ask_px_00': 'ask',
                    'bid_sz_00': 'bid_size',
                    'ask_sz_00': 'ask_size'
                })
                
                # Calculate mid price and spread
                bbo_df['mid'] = (bbo_df['bid'] + bbo_df['ask']) / 2
                bbo_df['spread'] = bbo_df['ask'] - bbo_df['bid']
                
                # Keep essential columns
                bbo_df = bbo_df[['timestamp', 'bid', 'ask', 'mid', 'spread', 'bid_size', 'ask_size']]
                
                logging.info(f"   ✅ BBO data: {len(bbo_df):,} quotes")
                logging.info(f"      Bid range: ${bbo_df['bid'].min():.2f} - ${bbo_df['bid'].max():.2f}")
                logging.info(f"      Ask range: ${bbo_df['ask'].min():.2f} - ${bbo_df['ask'].max():.2f}")
                logging.info(f"      Avg spread: ${bbo_df['spread'].mean():.2f}")
            
            # Process trade data
            if not trade_df.empty:
                trade_df = trade_df.reset_index()
                
                trade_df = trade_df.rename(columns={
                    'ts_event': 'timestamp',
                    'price': 'trade_price',
                    'size': 'trade_size'
                })
                
                # Keep essential columns
                trade_df = trade_df[['timestamp', 'trade_price', 'trade_size']]
                
                logging.info(f"   ✅ Trade data: {len(trade_df):,} trades")
                logging.info(f"      Price range: ${trade_df['trade_price'].min():.2f} - ${trade_df['trade_price'].max():.2f}")
            
            # Merge BBO and trade data
            if not bbo_df.empty and not trade_df.empty:
                # Merge on nearest timestamp
                merged_df = pd.merge_asof(
                    trade_df.sort_values('timestamp'),
                    bbo_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='backward'
                )
            elif not bbo_df.empty:
                merged_df = bbo_df
            else:
                merged_df = trade_df
            
            # Create the final dataset for backtesting
            final_df = pd.DataFrame()
            
            # Resample to regular intervals (5 seconds) for consistent backtesting
            if not merged_df.empty:
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
                merged_df = merged_df.set_index('timestamp')
                
                # Resample to 5-second bars
                resampled = merged_df.resample('5S').agg({
                    'bid': 'last',
                    'ask': 'last', 
                    'mid': 'last',
                    'spread': 'mean',
                    'trade_price': 'last',
                    'trade_size': 'sum'
                }).fillna(method='ffill')  # Forward fill missing values
                
                # Add high/low for compatibility
                resampled['high'] = resampled[['bid', 'ask', 'trade_price']].max(axis=1)
                resampled['low'] = resampled[['bid', 'ask', 'trade_price']].min(axis=1)
                
                # Use mid price as "close" for compatibility, but keep bid/ask
                resampled['close'] = resampled['mid'].fillna(resampled['trade_price'])
                resampled['volume'] = resampled['trade_size'].fillna(0)
                
                final_df = resampled.reset_index()
            
            # Save to CSV with all the important data
            if not final_df.empty:
                # Create filename
                expiry_str_file = exp_date.strftime('%Y%m%d')
                filename = f"{ticker}_{expiry_str_file}_{int(strike)}{right}_databento_PROPER.csv"
                filepath = self.output_dir / filename
                
                # Select columns to save
                columns_to_save = ['timestamp', 'bid', 'ask', 'mid', 'spread', 
                                 'close', 'high', 'low', 'volume']
                
                # Only keep columns that exist
                columns_to_save = [col for col in columns_to_save if col in final_df.columns]
                
                final_df[columns_to_save].to_csv(filepath, index=False)
                
                logging.info(f"   💾 Saved to: {filename}")
                logging.info(f"      Rows: {len(final_df):,}")
                logging.info(f"      Time range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
            
        except Exception as e:
            logging.error(f"   ❌ Error: {e}")
    
    def harvest_from_file(self, signals_file):
        """
        Harvest data for all signals in a file
        """
        signals_path = Path(signals_file)
        
        if not signals_path.exists():
            logging.error(f"Signals file not found: {signals_file}")
            return
        
        with open(signals_path, 'r') as f:
            lines = f.readlines()
        
        # Process each signal
        signal_count = 0
        for line in lines:
            line = line.strip()
            if line and '|' in line and not line.startswith('#') and not line.startswith('Trader:'):
                signal_count += 1
                logging.info(f"\n{'='*60}")
                logging.info(f"Processing signal {signal_count}: {line[:50]}...")
                self.harvest_signal(line)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"✅ Harvested data for {signal_count} signals")


def main():
    """Main entry point"""
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('DATABENTO_API_KEY')
    if not api_key:
        logging.error("DATABENTO_API_KEY not found in environment")
        sys.exit(1)
    
    # Create harvester
    harvester = ProperDatabentoHarvester(api_key)
    
    # Harvest from signals file
    signals_file = 'backtester/signals_to_test.txt'
    harvester.harvest_from_file(signals_file)
    
    logging.info("\n🎉 Data harvesting complete!")


if __name__ == "__main__":
    main()
