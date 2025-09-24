import logging
import asyncio
import pandas as pd
from ib_insync import IB, Option
import sys
import os

# --- SURGICAL UPGRADE: The "GPS" ---
# This tells the script how to find the other toolboxes from inside the 'backtester' folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ... (The rest of the DataHarvester class is unchanged) ...

async def main():
    # --- Paths are now relative to this script's location ---
    script_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(script_dir, 'historical_data'), exist_ok=True)

    signal_file_path = os.path.join(script_dir, 'signals_to_test.txt')
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
            output_dir=os.path.join(script_dir, 'historical_data')
        )
    finally:
        await harvester.disconnect()


if __name__ == "__main__":
    asyncio.run(main())