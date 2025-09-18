import logging
import asyncio
import pandas as pd
from ib_insync import IB, Option
import sys
import os

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataHarvester:
    """
    A specialist module for downloading historical options data from IBKR.
    This is the "Smart Edition" that reads a timestamped signal log.
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
        Reads a timestamped file of signals, fetches historical data for each, and saves to CSV.
        """
        os.makedirs(output_dir, exist_ok=True)
        parser = SignalParser(self.config)
        processed_contracts = set() # To avoid downloading data for the same contract twice

        with open(signal_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # --- SURGICAL UPGRADE: The "Smart" Line Parser ---
                parts = line.split('|')
                if len(parts) != 2:
                    logger.warning(f"Skipping malformed line #{line_num}: '{line}'")
                    continue
                
                timestamp_str, signal_text = parts
                signal_text = signal_text.strip()
                # --- END UPGRADE ---

                parsed_signal = parser.parse_signal_message(signal_text, {})
                if not parsed_signal:
                    logger.warning(f"Could not parse signal on line #{line_num}: '{signal_text}'")
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

                    # --- Efficiency Upgrade: Check if we already have this data ---
                    if contract.conId in processed_contracts:
                        logger.info(f"Already processed contract for {contract.localSymbol}. Skipping download.")
                        continue
                    
                    logger.info(f"Fetching data for {contract.localSymbol}...")

                    lookback_str = f"{self.config.backtesting['lookback_days']} D"
                    bar_size_str = self.config.backtesting['bar_size']

                    bars = await self.ib.reqHistoricalDataAsync(
                        contract, endDateTime='', durationStr=lookback_str,
                        barSizeSetting=bar_size_str, whatToShow='TRADES', useRTH=True
                    )

                    if not bars:
                        logger.warning(f"No historical data returned for {contract.localSymbol}.")
                        continue

                    df = pd.DataFrame([vars(b) for b in bars])
                    df['date'] = pd.to_datetime(df['date'])
                    
                    filename = f"{contract.localSymbol.replace(' ', '_')}.csv"
                    filepath = os.path.join(output_dir, filename)
                    
                    df.to_csv(filepath, index=False)
                    logger.info(f"Successfully saved {len(df)} bars to {filepath}")
                    processed_contracts.add(contract.conId) # Mark this contract as done

                except Exception as e:
                    logger.error(f"Failed to process signal '{signal_text}': {e}", exc_info=True)
                
                await asyncio.sleep(10)

async def main():
    """Main entry point for the script."""
    os.makedirs('backtester/historical_data', exist_ok=True)
    
    signal_file_path = 'backtester/signals_to_test.txt'
    if not os.path.exists(signal_file_path):
        with open(signal_file_path, 'w') as f:
            f.write("# Format: YYYY-MM-DD HH:MM:SS | The exact signal text from Discord\n")
            f.write("2025-07-07 08:37:00 | BTO SPY 600C 09/25\n")
        logger.info(f"Created a sample signal file at: {signal_file_path}")

    harvester = DataHarvester(Config())
    try:
        await harvester.connect()
        await harvester.fetch_and_save_data(
            signal_file=signal_file_path,
            output_dir='backtester/historical_data'
        )
    finally:
        await harvester.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

