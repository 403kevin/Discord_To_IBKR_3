# IBKR_Historical_Options_Test.py
import logging
from datetime import datetime, timedelta
from ib_insync import IB, Option, util
import pandas as pd

# --- Configuration ---
# You can change these to test different contracts
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497  # 7497 for TWS, 4001 for Gateway
IBKR_CLIENT_ID = 10 # Use a different client ID than the main bot

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    """
    Connects to IBKR and fetches historical data for a specific,
    expired options contract to test data availability.
    """
    ib = IB()
    try:
        logging.info(f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT}...")
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)

        # --- Define the Expired Contract to Fetch ---
        # We need to be very specific. Let's look for an AAPL call that expired last month.
        # This is an example; you may need to adjust the date and strike.
        today = datetime.today()
        # Go back ~30 days to find a contract that is likely expired.
        target_day = today - timedelta(days=30)
        # Find the Friday of that week (a common expiry day)
        expiry_day = target_day + timedelta(days=(4 - target_day.weekday()))

        # Format for IBKR API: YYYYMMDD
        expiry_str = expiry_day.strftime('%Y%m%d')

        contract_to_fetch = Option(
            symbol='AAPL',
            lastTradeDateOrContractMonth=expiry_str,
            strike=170, # An example strike price
            right='C',
            exchange='SMART'
        )

        logging.info(f"Attempting to qualify contract: {contract_to_fetch.localSymbol}")

        # Ask IBKR to find the official contract details
        qualified_contracts = ib.qualifyContracts(contract_to_fetch)

        if not qualified_contracts:
            logging.error("Could not find a valid contract for the specified details. Try adjusting the strike or expiry date.")
            return

        contract = qualified_contracts[0]
        logging.info(f"Successfully qualified contract: {contract.localSymbol} (conId: {contract.conId})")

        # --- Request Historical Data ---
        # We'll request 5-minute bars for the expiry day.
        logging.info(f"Requesting 5-minute bars for {expiry_str}...")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=f'{expiry_str} 20:00:00 US/Eastern',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )

        if not bars:
            logging.warning("No historical data was returned. This could be due to:")
            logging.warning("1. The contract did not trade on that day.")
            logging.warning("2. You may not have the required market data subscriptions on your account.")
            logging.warning("3. The request was made outside of historical data request hours.")
            return

        # --- Display the Data ---
        df = util.df(bars) # Convert the bar data to a pandas DataFrame
        print("\n--- Historical Data Received ---")
        print(df)
        print("\n✅ Test successful. Data was retrieved from IBKR.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if ib.isConnected():
            logging.info("Disconnecting from IBKR.")
            ib.disconnect()

if __name__ == "__main__":
    run_test()

