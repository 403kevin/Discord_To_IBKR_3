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
    based on a list of signals. This version has been hardened with a self-aware
    pathing system to make it immune to execution environment issues.
    """
    def __init__(self, config_path, signals_path, output_dir):
        self.config = Config()
        self.ib_interface = IBInterface(self.config)
        self.signal_parser = SignalParser(self.config)
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
        # This check is now robust, regardless of the working directory
        if not os.path.exists(self.signals_path):
             logging.error(f"FileNotFoundError: Cannot find signals file at '{self.signals_path}'")
             return []

        with open(self.signals_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parsed_signal = None
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        signal_text = parts[2].strip()
                        parsed_signal = self.signal_parser.parse_signal(signal_text, self.config.profiles[0])
                else:
                    parsed_signal = self._parse_simple_format(line)

                if parsed_signal:
                    signals.append(parsed_signal)
                else:
                    logging.warning(f"Skipping malformed line #{i+1}: '{line}'")
        return signals

    def _parse_simple_format(self, text):
        """Parses the 'TICKER MM/DD STRIKE_TYPE' format."""
        match = re.search(r'([A-Z]{1,5})\s+(\d{1,2}/\d{1,2})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text.upper())
        if not match:
            return None
        
        ticker, date_str, strike_str, _, contract_type_char = match.groups()
        
        try:
            month, day = map(int, date_str.split('/'))
            year = datetime.now().year
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

            data = await self.ib_interface.get_historical_data(contract, duration='1 D', bar_size='5 secs')

            if data is not None and not data.empty:
                filename = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{int(contract.strike)}{contract.right}_5sec_data.csv"
                filepath = os.path.join(self.output_dir, filename)
                data.to_csv(filepath)
                logging.info(f"Successfully saved {len(data)} data points to {filepath}")
            else:
                logging.warning(f"No historical data returned for {contract.localSymbol}")
            
            await asyncio.sleep(11)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- THE FINAL FIX: THE SELF-AWARE GPS ---
    # This logic makes the script's pathing immune to the IDE's "Working Directory".
    # It determines its own location and builds paths from there.
    
    # 1. Get the directory where THIS script lives.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define paths relative to THIS script's location.
    signals_path = os.path.join(script_dir, 'signals_to_test.txt')
    output_dir = os.path.join(script_dir, 'historical_data')

    # 3. The config_path is relative to the project root, which we already added to sys.path
    config_path = 'services/config.py'

    harvester = DataHarvester(config_path, signals_path, output_dir)
    asyncio.run(harvester.run())

