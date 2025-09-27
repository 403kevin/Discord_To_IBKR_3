import asyncio
import logging
import os
from datetime import datetime
import pandas as pd
import re

# This is a standalone script. We need to add the project root to the path
# to allow it to import from the services and interfaces directories.
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from interfaces.ib_interface import IBInterface
from services.config import Config
from services.signal_parser import SignalParser

class DataHarvester:
    """
    A standalone tool to download historical options data from Interactive Brokers
    based on a list of signals. This version is now "bilingual" and can parse
    both simple and complex signal file formats.
    """
    def __init__(self, config_path, signals_path, output_dir):
        self.config = Config()
        self.ib_interface = IBInterface(self.config)
        self.signal_parser = SignalParser(self.config) # Use our main parser
        self.signals_path = signals_path
        self.output_dir = output_dir
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
        Parses the signals_to_test.txt file, now handling both simple
        and complex formats.
        """
        signals = []
        with open(self.signals_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parsed_signal = None
                # Check for the complex format (contains '|')
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        signal_text = parts[2].strip()
                        # Use our main bot's parser for consistency
                        parsed_signal = self.signal_parser.parse_signal(signal_text, self.config.profiles[0])
                # Otherwise, assume it's the new, simple format
                else:
                    parsed_signal = self._parse_simple_format(line)

                if parsed_signal:
                    signals.append(parsed_signal)
                else:
                    logging.warning(f"Skipping malformed line #{i+1}: '{line}'")
        return signals

    def _parse_simple_format(self, text):
        """Parses the 'TICKER MM/DD STRIKE_TYPE' format."""
        # Example: NVDA 10/3 175P
        match = re.search(r'([A-Z]{1,5})\s+(\d{1,2}/\d{1,2})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text.upper())
        if not match:
            return None
        
        ticker, date_str, strike_str, _, contract_type_char = match.groups()
        
        try:
            month, day = map(int, date_str.split('/'))
            year = datetime.now().year
            # Basic year rollover logic
            if month < datetime.now().month:
                year += 1
            expiry_date = f"{year}{month:02d}{day:02d}"

            return {
                "ticker": ticker,
                "expiry_date": expiry_date,
                "strike": float(strike_str),
                "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
            }
        except ValueError:
            return None

    async def _fetch_and_save_data(self, signals):
        """Iterates through signals, creates contracts, and fetches data."""
        for signal in signals:
            logging.info(f"Fetching data for: {signal}")
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'], signal['expiry_date'], signal['strike'], signal['contract_type']
            )
            if not contract:
                continue

            # Fetching 5-second bars for a full day to power the simulator
            data = await self.ib_interface.get_historical_data(contract, duration='1 D', bar_size='5 secs')

            if data is not None and not data.empty:
                # Create a filename that the mock interface can easily parse
                filename = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{int(contract.strike)}{contract.right}_5sec_data.csv"
                filepath = os.path.join(self.output_dir, filename)
                data.to_csv(filepath)
                logging.info(f"Successfully saved {len(data)} data points to {filepath}")
            else:
                logging.warning(f"No historical data returned for {contract.localSymbol}")
            
            await asyncio.sleep(11) # IBKR pacing rule: max 50 requests every 2 minutes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Assuming this script is run from the project's root directory
    config_path = 'services/config.py' # Path is relative, might need adjustment
    signals_path = 'backtester/signals_to_test.txt'
    output_dir = 'backtester/historical_data'

    harvester = DataHarvester(config_path, signals_path, output_dir)
    asyncio.run(harvester.run())
