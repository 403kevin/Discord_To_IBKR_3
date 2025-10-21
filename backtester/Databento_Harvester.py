#!/usr/bin/env python3
"""
Databento Options Data Harvester - WINDOWS COMPATIBLE VERSION
FIXED: Uses signal date, not expiry date
FIXED: Windows file path handling

Since you day trade and exit all positions before market close,
this harvester fetches data for the SIGNAL DATE, not the expiry date.
"""

import databento as db
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Your Databento API key
API_KEY = "YOUR_DATABENTO_API_KEY_HERE"  # Replace with your actual key

def parse_signal_file(filepath):
    """
    Parse signals file to extract option details.
    Format: YYYY-MM-DD HH:MM:SS | Trader | TICKER STRIKEP/C MM/DD
    
    WINDOWS FIX: Use pathlib for cross-platform compatibility
    """
    signals = []
    
    # Convert to Path object for Windows compatibility
    filepath = Path(filepath)
    
    if not filepath.exists():
        logging.error(f"❌ File not found: {filepath}")
        logging.error(f"   Current directory: {Path.cwd()}")
        logging.error(f"   Looking for: {filepath.absolute()}")
        return []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                # Parse: "2025-10-07 07:46:43 | Expo | MSFT 515P 10/10"
                parts = line.split('|')
                if len(parts) != 3:
                    logging.warning(f"Skipping malformed line: {line}")
                    continue
                
                timestamp_str = parts[0].strip()
                trader = parts[1].strip()
                signal_text = parts[2].strip()
                
                # Parse signal timestamp to get SIGNAL DATE
                signal_datetime = pd.to_datetime(timestamp_str)
                signal_date = signal_datetime.date()
                
                # Parse signal text (e.g., "MSFT 515P 10/10")
                signal_parts = signal_text.split()
                ticker = signal_parts[0]
                
                # Extract strike and right
                strike_str = signal_parts[1]
                if strike_str[-1] in ['C', 'P']:
                    strike = float(strike_str[:-1])
                    right = strike_str[-1]
                else:
                    logging.warning(f"Invalid strike/right format: {strike_str}")
                    continue
                
                # Parse expiry date
                expiry_parts = signal_parts[2].split('/')
                month = int(expiry_parts[0])
                day = int(expiry_parts[1])
                
                # Determine expiry year
                year = signal_datetime.year
                if month < signal_datetime.month:
                    year += 1
                
                expiry_date = datetime(year, month, day).date()
                
                signals.append({
                    'ticker': ticker,
                    'strike': strike,
                    'right': right,
                    'signal_date': signal_date,  # DAY OF THE TRADE
                    'expiry_date': expiry_date,  # EXPIRATION DATE
                    'signal_time': signal_datetime,
                    'trader': trader,
                    'raw_signal': line
                })
                
            except Exception as e:
                logging.error(f"Error parsing signal '{line}': {e}")
                continue
    
    logging.info(f"✅ Parsed {len(signals)} signals from {filepath.name}")
    return signals

def get_option_symbol(ticker, expiry_date, strike, right):
    """
    Generate OCC option symbol.
    Format: UNDERLYING(6) + YYMMDD(6) + P/C(1) + PRICE(8)
    """
    # Handle special tickers
    if ticker == "SPX":
        underlying = "SPXW  "  # SPXW options for SPX
    elif ticker == "NDX":
        underlying = "NDXP  "  # NDXP for NDX
    else:
        underlying = ticker.ljust(6)[:6]
    
    # Format expiry
    expiry_str = expiry_date.strftime("%y%m%d")
    
    # Format strike price (8 digits, no decimal)
    strike_int = int(strike * 1000)
    strike_str = str(strike_int).zfill(8)
    
    symbol = f"{underlying}{expiry_str}{right}{strike_str}"
    
    return symbol

def fetch_databento_data(signal, client):
    """
    Fetch historical data for the SIGNAL DATE (day of trade).
    
    CRITICAL FIX: Uses signal_date, not expiry_date!
    """
    ticker = signal['ticker']
    strike = signal['strike']
    right = signal['right']
    signal_date = signal['signal_date']  # THIS IS THE FIX!
    expiry_date = signal['expiry_date']
    
    # Generate OCC symbol
    option_symbol = get_option_symbol(ticker, expiry_date, strike, right)
    
    logging.info(f"📥 Fetching data for {ticker} {strike}{right}")
    logging.info(f"   Signal Date: {signal_date} (DAY OF TRADE)")
    logging.info(f"   Expiry Date: {expiry_date}")
    logging.info(f"   OCC Symbol: {option_symbol}")
    
    try:
        # CRITICAL: Use signal_date for both start and end (day trading)
        start_datetime = datetime.combine(signal_date, datetime.min.time())
        end_datetime = datetime.combine(signal_date, datetime.max.time())
        
        # Get data for the TRADING DAY, not expiry day
        data = client.timeseries.get_range(
            dataset="OPRA.PILLAR",  # Options dataset
            symbols=[option_symbol],
            schema="cmbp-1",  # Consolidated Market By Price (has bid/ask)
            start=start_datetime,
            end=end_datetime,
            stype_in="raw_symbol"
        )
        
        # Convert to DataFrame
        df = data.to_df()
        
        if df.empty:
            logging.warning(f"   ⚠️ No data found for {option_symbol} on {signal_date}")
            return None
        
        # Rename columns to match backtest engine expectations
        df = df.rename(columns={
            'ts_event': 'ts_event',  # Keep as is
            'bid_px_00': 'bid',
            'ask_px_00': 'ask',
            'bid_sz_00': 'bid_size',
            'ask_sz_00': 'ask_size'
        })
        
        # Calculate mid price and spread
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        
        # Add close price (use mid as proxy)
        df['close'] = df['mid']
        
        # Filter out invalid data
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        if df.empty:
            logging.warning(f"   ⚠️ No valid bid/ask data for {option_symbol}")
            return None
        
        # Log statistics
        avg_spread = df['spread'].mean()
        bid_range = (df['bid'].min(), df['bid'].max())
        ask_range = (df['ask'].min(), df['ask'].max())
        
        logging.info(f"   ✅ Fetched {len(df)} ticks for SIGNAL DATE {signal_date}")
        logging.info(f"   📊 Avg spread: ${avg_spread:.2f}")
        logging.info(f"   📊 Bid range: ${bid_range[0]:.2f} - ${bid_range[1]:.2f}")
        logging.info(f"   📊 Ask range: ${ask_range[0]:.2f} - ${ask_range[1]:.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"   ❌ Error fetching data: {e}")
        return None

def save_data(df, signal, output_dir):
    """
    Save data to CSV file using standard naming convention.
    WINDOWS FIX: Use pathlib for path handling
    """
    ticker = signal['ticker']
    strike = signal['strike']
    right = signal['right']
    expiry_date = signal['expiry_date']
    
    # Format filename: TICKER_YYYYMMDD_STRIKE[C/P]_databento.csv
    expiry_str = expiry_date.strftime("%Y%m%d")
    
    # Handle strike formatting
    if strike == int(strike):
        strike_str = str(int(strike))
    else:
        strike_str = str(strike)
    
    filename = f"{ticker}_{expiry_str}_{strike_str}{right}_databento.csv"
    
    # Use Path for Windows compatibility
    output_path = Path(output_dir)
    filepath = output_path / filename
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    logging.info(f"   💾 Saved to {filename}")
    logging.info(f"      Data date: {signal['signal_date']} (signal date)")
    logging.info(f"      Rows: {len(df)}")
    
    return str(filepath)

def main():
    """
    Main execution function.
    WINDOWS FIX: Better path handling and error messages
    """
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    # Configuration - use Path objects
    signals_file = script_dir / "signals_to_test.txt"
    output_dir = script_dir / "historical_data"
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("DATABENTO OPTIONS DATA HARVESTER")
    logging.info("=" * 60)
    logging.info(f"Script directory: {script_dir}")
    logging.info(f"Signals file: {signals_file}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)
    
    # Check if signals file exists
    if not signals_file.exists():
        logging.error(f"❌ FATAL: signals_to_test.txt not found!")
        logging.error(f"   Expected location: {signals_file.absolute()}")
        logging.error(f"   Current directory: {Path.cwd()}")
        logging.error("")
        logging.error("📝 CREATE THE FILE:")
        logging.error(f"   1. Navigate to: {script_dir}")
        logging.error(f"   2. Create file: signals_to_test.txt")
        logging.error(f"   3. Add your signals (one per line):")
        logging.error(f"      2025-10-07 07:46:43 | Expo | MSFT 515P 10/10")
        return
    
    # Parse signals
    signals = parse_signal_file(signals_file)
    
    if not signals:
        logging.error("❌ No signals found to process")
        logging.error(f"   Check the format in: {signals_file}")
        return
    
    # Initialize Databento client
    if API_KEY == "YOUR_DATABENTO_API_KEY_HERE":
        logging.error("❌ FATAL: Please set your Databento API key!")
        logging.error("   Edit this script and replace:")
        logging.error('   API_KEY = "YOUR_DATABENTO_API_KEY_HERE"')
        logging.error("   with your actual API key")
        return
    
    client = db.Historical(API_KEY)
    
    # Process each signal
    success_count = 0
    fail_count = 0
    
    for signal in signals:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {signal['raw_signal']}")
        
        # Fetch data for SIGNAL DATE (not expiry date!)
        df = fetch_databento_data(signal, client)
        
        if df is not None:
            # Save data
            save_data(df, signal, output_dir)
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"HARVEST COMPLETE")
    logging.info(f"✅ Success: {success_count}")
    logging.info(f"❌ Failed: {fail_count}")
    logging.info(f"📁 Output: {output_dir}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
