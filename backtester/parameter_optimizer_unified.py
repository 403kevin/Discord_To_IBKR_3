import logging
import os
import sys
from datetime import datetime, timedelta  # ✅ FIXED: Added timedelta import
from pathlib import Path
from dotenv import load_dotenv
import databento as db

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename_databento

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class DatabentoHarvester:
    """
    Downloads historical options data from Databento for backtesting.
    FIXED: Fetches from SIGNAL DATE through EXPIRY, not just expiry day.
    """
    def __init__(self, api_key):
        self.client = db.Historical(api_key)
        logging.info("Initialized Historical(gateway=https://hist.databento.com)")
        
        self.config = Config()
        self.parser = SignalParser(self.config)
        
        # File paths
        self.project_root = Path(__file__).parent.parent.absolute()
        self.signals_file = self.project_root / 'backtester' / 'signals_to_test.txt'
        self.output_dir = self.project_root / 'backtester' / 'historical_data'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Using signals file: {self.signals_file}")
    
    def _load_signals(self):
        """Load and parse signals from file."""
        signals = []
        
        if not self.signals_file.exists():
            logging.error(f"Signals file not found: {self.signals_file}")
            return signals
        
        logging.info("Loading signals...")
        
        with open(self.signals_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse signal using SignalParser
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    channel = parts[1].strip()
                    signal_text = parts[2].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    channel = 'test_server'
                    signal_text = line
                
                # Parse with SignalParser
                signal_dict = self.parser.parse_signal(signal_text, channel)
                
                if signal_dict:
                    # Add timestamp
                    signal_dict['timestamp'] = timestamp_str
                    signals.append(signal_dict)
                    
                    ticker = signal_dict['ticker']
                    strike = signal_dict['strike']
                    right = signal_dict['right']
                    expiry = signal_dict['expiry']
                    
                    logging.info(f"✓ {ticker} {strike}{right} {expiry}")
        
        logging.info(f"Found {len(signals)} valid signals")
        return signals
    
    def _download_signal_data(self, signal):
        """Download historical data for a single signal."""
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
            
            # Date range: signal date → expiry
            start_date = signal_date
            end_date = exp_date + timedelta(days=1)  # ✅ End is exclusive, so add 1 day
            
            # Build OCC symbol - use YYMMDD (2 digit year)
            expiry_str = exp_date.strftime('%y%m%d')  # ✅ FIXED: YY not YYYY
            strike_str = f"{int(strike * 1000):08d}"
            
            # Convert SPX → SPXW for Databento
            root_symbol = 'SPXW' if ticker.upper() == 'SPX' else ticker.upper()
            occ_symbol = f"{root_symbol.ljust(6)}{expiry_str}{right}{strike_str}"
            
            logging.info(f"   OCC: {occ_symbol}")
            logging.info(f"   Date range: {start_date} → {end_date}")
            
            # Download from Databento
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                schema='trades',
                start=start_date,
                end=end_date,
                stype_in='raw_symbol'
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"   ⚠️ No data found")
                return
            
            # Save to CSV
            filename = get_data_filename_databento(ticker, expiry, strike, right)
            output_path = self.output_dir / filename
            
            df.to_csv(output_path, index=False)
            logging.info(f"   ✅ Saved {len(df):,} ticks → {filename}")
            
        except Exception as e:
            logging.error(f"   ❌ Failed: {e}")
    
    def run(self):
        """Main execution - download all signals."""
        logging.info("Starting Databento data download...")
        
        signals = self._load_signals()
        
        if not signals:
            logging.error("No valid signals found!")
            return
        
        logging.info("=" * 60)
        logging.info(f"DOWNLOADING DATA FOR {len(signals)} SIGNALS")
        logging.info("=" * 60)
        
        for signal in signals:
            self._download_signal_data(signal)
        
        logging.info("\n" + "=" * 60)
        logging.info("DOWNLOAD COMPLETE")
        logging.info("=" * 60)


if __name__ == "__main__":
    # Load API key
    load_dotenv()
    api_key = os.getenv('DATABENTO_API_KEY')
    
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
