import logging
import os
import sys
from datetime import datetime, timedelta
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

class DatabentoHarvester:
    """
    Downloads historical options data from Databento for backtesting.
    Uses the battle-tested SignalParser to handle multiple signal formats.
    """
    def __init__(self, api_key, signals_path=None, output_dir=None):
        self.client = db.Historical(api_key)
        self.config = Config()
        self.signal_parser = SignalParser(self.config)
        
        script_dir = Path(__file__).parent
        self.signals_path = signals_path or script_dir / 'signals_to_test.txt'
        self.output_dir = output_dir or script_dir / 'historical_data'
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Main execution loop."""
        signals = self._load_signals()
        
        if not signals:
            logging.error("No valid signals found")
            return
        
        logging.info(f"\n{'='*60}")
        logging.info(f"DOWNLOADING DATA FOR {len(signals)} SIGNALS")
        logging.info(f"{'='*60}\n")
        
        for signal in signals:
            self._download_signal_data(signal)
        
        logging.info(f"\n{'='*60}")
        logging.info("DOWNLOAD COMPLETE")
        logging.info(f"{'='*60}")
    
    def _load_signals(self):
        """Parse signals from file."""
        signals = []
        
        logging.info("Loading signals...")
        
        default_profile = {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        with open(self.signals_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Handle format: "TIMESTAMP | CHANNEL | SIGNAL"
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
        """Download historical data for a single signal."""
        ticker = signal['ticker']
        expiry = signal['expiry_date']  # Format: YYYY-MM-DD or YYYYMMDD
        strike = signal['strike']
        right = signal['contract_type'][0]
        
        logging.info(f"\n📥 {ticker} {strike}{right} exp {expiry}")
        
        # Convert expiry to datetime object
        try:
            if '-' in expiry:
                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
            else:
                exp_date = datetime.strptime(expiry, '%Y%m%d')
        except Exception as e:
            logging.error(f"   ❌ Invalid expiry format: {expiry}")
            return
        
        # Build OCC symbol with CORRECT Databento format
        # Format: {Ticker:6chars}{Expiry:YYMMDD}{Right}{Strike:8 digits}
        # Ticker must be padded to 6 characters with spaces
        
        # CRITICAL: SPX weekly options use SPXW, not SPX
        # SPX = monthly expiries only (3rd Friday)
        # SPXW = all other expiries (Mon/Wed/Fri)
        root_symbol = ticker
        if ticker.upper() == 'SPX':
            # Check if this is the 3rd Friday (monthly expiry)
            # If not, use SPXW
            import calendar
            third_friday = self._get_third_friday(exp_date.year, exp_date.month)
            if exp_date.date() != third_friday:
                root_symbol = 'SPXW'
        
        expiry_str = exp_date.strftime('%y%m%d')  # YYMMDD
        ticker_padded = root_symbol.ljust(6)  # Pad to 6 chars with spaces
        strike_str = f"{int(strike * 1000):08d}"  # Strike × 1000, 8 digits
        occ_symbol = f"{ticker_padded}{expiry_str}{right}{strike_str}"
        
        logging.info(f"   OCC: {occ_symbol}")
        
        # Date range for data fetch
        # For 0DTE/short-dated options, we only need the expiry day itself
        start_date = exp_date
        end_date = exp_date
        
        try:
            logging.info(f"   Fetching: {start_date.date()} to {end_date.date()}")
            
            # Request data from Databento
            data = self.client.timeseries.get_range(
                dataset='OPRA.pillar',
                symbols=[occ_symbol],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                schema='trades',
                stype_in='raw_symbol'  # Use raw OCC symbols
            )
            
            df = data.to_df()
            
            if df.empty:
                logging.warning(f"   ⚠️  No data - symbol may not exist or no trading")
                return
            
            # Convert to backtest format
            df = df.reset_index()
            
            # Databento returns 'ts_event' as index
            if 'ts_event' not in df.columns:
                df['ts_event'] = df.index
            
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'price': 'close',
                'size': 'volume'
            })
            
            # Add high/low if not present
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            
            # Save to CSV
            filename = get_data_filename_databento(ticker, exp_date.strftime('%Y%m%d'), strike, right)
            filepath = self.output_dir / filename
            
            df[['timestamp', 'close', 'high', 'low', 'volume']].to_csv(filepath, index=False)
            
            price_range = f"${df['close'].min():.2f}-${df['close'].max():.2f}"
            logging.info(f"   ✅ Saved {len(df):,} ticks | Range: {price_range}")
            
        except Exception as e:
            logging.error(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
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
