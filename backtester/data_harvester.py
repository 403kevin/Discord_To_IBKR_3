import asyncio
import logging
import os
from datetime import datetime
import sys

# --- GPS FOR THE FORTRESS (PART 1) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Our Custom Tools ---
from interfaces.ib_interface import IBInterface
from services.config import Config
from services.signal_parser import SignalParser
from services.utils import get_data_filename

class DataHarvester:
    """
    Downloads historical options data from Interactive Brokers for backtesting.
    Now uses the battle-tested multi-format SignalParser.
    """
    def __init__(self, signals_path=None, output_dir=None):
        self.config = Config()
        self.ib_interface = IBInterface(self.config)
        self.signal_parser = SignalParser(self.config)
        
        # GPS-aware default paths if not provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.signals_path = signals_path or os.path.join(script_dir, 'signals_to_test.txt')
        self.output_dir = output_dir or os.path.join(script_dir, 'historical_data')
        os.makedirs(self.output_dir, exist_ok=True)

    async def run(self):
        """Main execution flow for the harvester."""
        logging.info("--- 🚀 Starting Data Harvester ---")
        if not await self.ib_interface.connect():
            logging.error("Could not connect to IBKR. Aborting.")
            return

        logging.info("Connection to IBKR for data harvesting successful.")
        
        signals_to_fetch = self._parse_signals_file()
        if not signals_to_fetch:
            logging.warning("No valid signals found to fetch data for.")
        else:
            await self._fetch_and_save_data(signals_to_fetch)

        await self.ib_interface.disconnect()
        logging.info("--- ✅ Data Harvester Finished ---")

    def _parse_signals_file(self):
        """
        Parses the signals_to_test.txt file using the multi-format SignalParser.
        Supports both simple format and timestamped format.
        """
        signals = []
        if not os.path.exists(self.signals_path):
            logging.error(f"FATAL: FileNotFoundError. Cannot find signals file at '{self.signals_path}'")
            return []

        # Use first profile as default for parsing
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }

        with open(self.signals_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Check if line has timestamp (Format 2: YYYY-MM-DD HH:MM:SS | channel | signal)
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        # Extract just the signal text for parsing
                        signal_text = parts[2].strip()
                    else:
                        logging.warning(f"Malformed timestamped line #{line_num}: '{line}'")
                        continue
                else:
                    # Simple format (Format 1: just the signal)
                    signal_text = line
                
                # Parse using the battle-tested multi-format parser
                parsed_signal = self.signal_parser.parse_signal(signal_text, default_profile)
                
                if parsed_signal:
                    signals.append(parsed_signal)
                    logging.info(f"Parsed signal #{line_num}: {parsed_signal['ticker']} {parsed_signal['expiry_date']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]}")
                else:
                    logging.warning(f"Could not parse line #{line_num}: '{signal_text}'")
        
        logging.info(f"Successfully parsed {len(signals)} valid signals from {self.signals_path}")
        return signals

    async def _fetch_and_save_data(self, signals):
        """Iterates through signals, creates contracts, and fetches data."""
        for i, signal in enumerate(signals, 1):
            logging.info(f"[{i}/{len(signals)}] Fetching data for: {signal['ticker']} {signal['expiry_date']} {signal['strike']}{signal['contract_type'][0]}")
            
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], 
                signal['expiry_date'], 
                signal['strike'], 
                signal['contract_type']
            )
            
            if not contract:
                logging.warning(f"Could not create contract for signal #{i}")
                continue

            # Fetch historical data (1-minute bars for 1 day) - CHANGED FROM 5 SECS
            data = await self.ib_interface.get_historical_data(
                contract, 
                duration='1 D', 
                bar_size='1 min'
            )

            if data is not None and not data.empty:
                # Use centralized filename generator
                filename = get_data_filename(contract)
                filepath = os.path.join(self.output_dir, filename)
                data.to_csv(filepath)
                logging.info(f"✅ Saved {len(data)} data points to {filename}")
            else:
                logging.warning(f"⚠️ No historical data returned for {contract.localSymbol}")
            
            # IBKR pacing rule: max 50 requests every 2 minutes
            # Wait 3 seconds between requests to be safe
            await asyncio.sleep(3)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    harvester = DataHarvester()
    asyncio.run(harvester.run())
