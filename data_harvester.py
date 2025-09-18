import logging
import asyncio
import pandas as pd
from ib_insync import IB, Option
import sys
import os

# --- Add project root to path to allow imports from other folders ---
# This is a professional way to manage imports in a multi-directory project.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can import from our existing, battle-hardened modules
from services.config import Config
from services.signal_parser import SignalParser

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataHarvester:
    """
    A specialist module for downloading historical options data from IBKR.
    This is the professional evolution of the successful IBKR_Historical_Options_Test.py.
    """

    def __init__(self, config):
        self.config = config
        self.ib = IB()

    async def connect(self):
        """Connects to IBKR."""
        try:
            await self.ib.connectAsync(self.config.ibkr_host, self.config.ibkr_port, clientId=self.config.ibkr_client_id + 1)
            logger.info("Connection to IBKR for data harvesting successful.")
        except Exception as e:
            logger.critical(f"Failed to connect to IBKR: {e}")
            raise

    async def disconnect(self):
        """Disconnects from IBKR."""
        self.ib.disconnect()
        logger.info("Disconnected from IBKR.")

    async def fetch_and_save_data(self, signal_file: str, output_dir: str):
        """
        Reads a file of signals, fetches historical data for each, and saves to CSV.
        
        Args:
            signal_file (str): Path to a text file with one signal per line.
            output_dir (str): Path to the directory where CSV files will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        parser = SignalParser(self.config)

        with open(signal_file, 'r') as f:
            for line in f:
                signal_text = line.strip()
                if not signal_text or signal_text.startswith('#'):
                    continue

                # Use our battle-hardened parser to translate the signal
                parsed_signal = parser.parse_signal_message(signal_text, {}) # Pass empty profile
                if not parsed_signal:
                    logger.warning(f"Could not parse signal: {signal_text}")
                    continue

                try:
                    contract = Option(
                        symbol=parsed_signal['ticker'],
                        lastTradeDateOrContractMonth=parsed_signal['expiry'],
                        strike=parsed_signal['strike'],
                        right=parsed_signal['option_type'],
                        exchange='SMART',
                        currency='USD'
                    )
                    await self.ib.qualifyContractsAsync(contract)
                    
                    logger.info(f"Fetching data for {contract.localSymbol}...")

                    # This is the core logic from your successful prototype
                    # We will fetch 1-minute bars for the last 30 days as an example
                    bars = await self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime='',
                        durationStr='30 D',
                        barSizeSetting='1 min',
                        whatToShow='TRADES',
                        useRTH=True
                    )

                    if not bars:
                        logger.warning(f"No historical data returned for {contract.localSymbol}.")
                        continue

                    # Convert the data to a pandas DataFrame and save to CSV
                    df = pd.DataFrame([vars(b) for b in bars])
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Create a unique, safe filename for the contract
                    filename = f"{contract.localSymbol.replace(' ', '_')}.csv"
                    filepath = os.path.join(output_dir, filename)
                    
                    df.to_csv(filepath, index=False)
                    logger.info(f"Successfully saved {len(df)} bars to {filepath}")

                except Exception as e:
                    logger.error(f"Failed to process signal '{signal_text}': {e}", exc_info=True)
                
                await asyncio.sleep(10) # Be polite to the API, wait 10 seconds between requests

async def main():
    """Main entry point for the script."""
    harvester = DataHarvester(Config())
    try:
        await harvester.connect()
        # You will need to create these files yourself
        await harvester.fetch_and_save_data(
            signal_file='backtester/signals_to_test.txt',
            output_dir='backtester/historical_data'
        )
    finally:
        await harvester.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
