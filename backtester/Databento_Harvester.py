import databento as db
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Get absolute project root
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from services.signal_parser import SignalParser
from services.config import Config
from services.utils import get_data_filename_databento

class DatabentoHarvester:
    def __init__(self, api_key, signals_path=None, output_dir=None):
        self.client = db.Historical(api_key)
        
        # Use absolute paths
        if signals_path:
            self.signals_path = Path(signals_path)
        else:
            self.signals_path = project_root / "backtester" / "signals_to_test.txt"
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = project_root / "backtester" / "historical_data"
            
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
        logging.info(f"Using signals file: {self.signals_path}")
        
    def run(self):
        logging.info("Starting Databento data download...")
        
        if not self.signals_path.exists():
            logging.error(f"Signals file not found: {self.signals_path}")
            return
            
        signals = self._parse_signals()
        
        if not signals:
            logging.error("No valid signals found")
            return
        
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
                    logging.info(f"✓ {parsed['ticker']} {parsed['strike']}{parsed['contract_type'][0]} {parsed['expiry_date']}")
        
        logging.info(f"\nFound {len(signals)} valid signals")
        return signals
    
    def _download_signal_data(self, signal):
        ticker = signal['ticker']
        expiry = signal['expiry_date']
        strike = signal['strike']
        right = signal['contract_type'][0]
        
        logging.info(f"\n📥 {ticker} {strike}{right} exp {expiry}")
        
        # Convert expiry to Databento format YYYYMMDD
        expiry_str = expiry.replace('-', '')
        
        # Build OCC symbol
        strike_str = f"{int(strike * 1000):08d}"
        symbol = f"{ticker}{expiry_str}{right}{strike_str}"
        logging.info(f"   OCC: {symbol}")
        
        # Get data
        try:
            exp_date = datetime.strptime(expiry, '%Y%m%d')
        except:
            exp_date = datetime.strptime(expiry, '%Y-%m-%d')
            
        start_date = exp_date - timedelta(days=1)
        end_date = exp_date
        
        try:
            logging.info(f"   Fetching: {start_date.date()} to {end_date.date()}")
            
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='trades'
            )
            
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"   ⚠️  No data")
                return
            
            # Convert to backtest format
            df = df.reset_index()
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'price': 'close'
            })
            
            # Add high/low/volume columns
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = df.get('size', 0)
            
            filename = get_data_filename_databento(ticker, expiry_str, strike, right)
            filepath = self.output_dir / filename
            
            df[['timestamp', 'close', 'high', 'low', 'volume']].to_csv(filepath, index=False)
            
            logging.info(f"   ✅ Saved {len(df)} ticks")
            
        except Exception as e:
            logging.error(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Load .env
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', help='Databento API key')
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        print("ERROR: API key required")
        print("Add DATABENTO_API_KEY to .env or use --api-key")
        sys.exit(1)
    
    print("="*60)
    print("DATABENTO DATA HARVESTER")
    print("="*60)
    
    harvester = DatabentoHarvester(api_key)
    harvester.run()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
