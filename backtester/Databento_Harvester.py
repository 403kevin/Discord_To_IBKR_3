import databento as db
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.signal_parser import SignalParser
from services.config import Config
from services.utils import get_data_filename_databento

class DatabentoHarvester:
    def __init__(self, api_key, signals_path="backtester/signals_to_test.txt", output_dir="backtester/historical_data"):
        self.client = db.Historical(api_key)
        self.signals_path = Path(signals_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
    def run(self):
        logging.info("Starting Databento data download...")
        signals = self._parse_signals()
        
        for signal in signals:
            self._download_signal_data(signal)
    
    def _parse_signals(self):
        signals = []
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        with open(self.signals_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if '|' in line:
                    signal_text = line.split('|')[2].strip()
                else:
                    signal_text = line
                
                parsed = self.signal_parser.parse_signal(signal_text, default_profile)
                if parsed:
                    signals.append(parsed)
        
        return signals
    
    def _download_signal_data(self, signal):
        ticker = signal['ticker']
        expiry = signal['expiry_date']
        strike = signal['strike']
        right = signal['contract_type'][0]
        
        logging.info(f"Downloading {ticker} {expiry} {strike}{right}...")
        
        # Convert expiry to Databento format YYYYMMDD
        expiry_str = expiry.replace('-', '')
        
        # Build OCC symbol: TICKER + EXPIRY + C/P + STRIKE (padded)
        strike_str = f"{int(strike * 1000):08d}"
        symbol = f"{ticker}{expiry_str}{right}{strike_str}"
        
        # Download tick data for 1 day
        try:
            exp_date = datetime.strptime(expiry, '%Y%m%d')
        except:
            exp_date = datetime.strptime(expiry, '%Y-%m-%d')
            
        start_date = exp_date - timedelta(days=1)
        end_date = exp_date
        
        try:
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='trades'
            )
            
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"No data for {symbol}")
                return
            
            # Convert to backtest format
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'price': 'close'
            })
            
            filename = get_data_filename_databento(ticker, expiry_str, strike, right)
            filepath = self.output_dir / filename
            df[['timestamp', 'close', 'size']].to_csv(filepath, index=False)
            
            logging.info(f"✅ Saved {len(df)} ticks to {filename}")
            
        except Exception as e:
            logging.error(f"Failed to download {symbol}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load .env file
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', help='Databento API key (or set DATABENTO_API_KEY in .env)')
    args = parser.parse_args()
    
    # Get API key from args or env
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        print("ERROR: API key required")
        print("Usage: python Databento_Harvester.py --api-key YOUR_KEY")
        print("   OR: Set DATABENTO_API_KEY in .env file")
        sys.exit(1)
    
    harvester = DatabentoHarvester(api_key)
    harvester.run()
