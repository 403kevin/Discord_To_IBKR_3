# IBKR_Historical_Options_Test.py

import logging
from ib_insync import IB, Option
from datetime import datetime, timedelta
import pandas as pd

# --- Configuration ---
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497  # 7497 for TWS, 4001 for Gateway
IBKR_CLIENT_ID = 10 # Use a different client ID from your main bot

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_options_data():
    """
    Connects to IBKR and fetches historical data for a specific, expired options contract.
    """
    ib = IB()
    try:
        logging.info("Connecting to IBKR...")
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
        logging.info("Connection successful.")

        # --- 1. Define the specific, expired contract we want to look up ---
        # We need to know the exact parameters of a contract that existed in the past.
        # Let's target an AAPL weekly option from about a month ago.
        
        # Calculate a target date roughly one month ago
        target_date = datetime.now() - timedelta(days=30)
        
        # We need to find a valid expiration date around that time.
        # For this example, let's manually specify a known past expiry.
        # In a real backtester, you would programmatically find valid expiries.
        # For AAPL, weekly options expire on Fridays. Let's find a recent Friday.
        # Example: If today is Aug 27, 2025 (Wednesday), a month ago was July 27.
        # The closest Friday expiry before that was July 25, 2025.
        
        # NOTE: You may need to adjust these values slightly depending on the current date
        # to ensure you're querying a valid, historical contract.
        symbol = 'AAPL'
        expiry = '20250725' # A Friday from last month
        strike = 210 # A reasonable strike price for AAPL at that time
        right = 'C' # 'C' for Call, 'P' for Put
        
        logging.info(f"Attempting to fetch data for: {symbol} {expiry} {strike}{right}")

        # Create the specific Option contract object
        contract = Option(symbol, expiry, strike, right, 'SMART', tradingClass='AAPL')
        
        # It's good practice to qualify the contract to get its conId
        ib.qualifyContracts(contract)
        logging.info(f"Qualified Contract: {contract.localSymbol}, conId: {contract.conId}")

        # --- 2. Request Historical Data ---
        # We will request one full day of data ending on the target date.
        end_datetime = target_date.strftime('%Y%m%d 23:59:59')
        
        logging.info(f"Requesting historical data ending on: {end_datetime}")

        # whatToShow: 'TRADES', 'MIDPOINT', 'BID', 'ASK'
        # useRTH: True means only show data from regular trading hours
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr='1 D', # Duration: 1 Day
            barSizeSetting='5 mins', # Bar size: 5 minutes
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1 # 1 for yyyy-MM-dd HH:mm:ss, 2 for epoch seconds
        )

        if not bars:
            logging.warning("No historical data was returned. This could be because:")
            logging.warning("- The contract did not trade on the requested day.")
            logging.warning("- The contract details (expiry, strike) are incorrect.")
            logging.warning("- You do not have the required market data subscriptions.")
            return

        # --- 3. Process and Display the Data ---
        # Convert the list of bars to a pandas DataFrame for nice formatting
        df = pd.DataFrame([{
            'DateTime': b.date,
            'Open': b.open,
            'High': b.high,
            'Low': b.low,
            'Close': b.close,
            'Volume': b.volume
        } for b in bars])
        
        logging.info(f"Successfully fetched {len(df)} data points.")
        print("\n--- Historical Options Data ---")
        print(df)
        print("-----------------------------\n")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if ib.isConnected():
            logging.info("Disconnecting from IBKR.")
            ib.disconnect()

if __name__ == "__main__":
    fetch_historical_options_data()
