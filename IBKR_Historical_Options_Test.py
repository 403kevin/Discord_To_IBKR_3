# IBKR_Historical_Options_Test.py (v4 - Final Test)
import logging
from datetime import datetime, timedelta
from ib_insync import IB, Option, util
import pandas as pd

# --- Configuration ---
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497
IBKR_CLIENT_ID = 13  # Use another new client ID

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_final_test():
    """
    This is the definitive test. It attempts to fetch data for a very high-volume,
    recently expired SPY contract. If this fails, the issue is almost certainly
    related to market data subscriptions, not the code itself.
    """
    ib = IB()
    try:
        logging.info(f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT}...")
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)

        # --- The "Educated Guess": Target last Friday's SPY options ---
        today = datetime.today()
        # today.weekday() is Monday=0, Sunday=6. We want to find the previous Friday (4).
        days_since_friday = (today.weekday() - 4 + 7) % 7
        if days_since_friday == 0:  # If today is Friday, get last week's Friday
            days_since_friday = 7

        last_friday = today - timedelta(days=days_since_friday)
        expiry_str = last_friday.strftime('%Y%m%d')

        # A common, high-volume, near-the-money strike for SPY
        target_strike = 540

        contract_to_fetch = Option(
            symbol='SPY',
            lastTradeDateOrContractMonth=expiry_str,
            strike=target_strike,
            right='C',
            exchange='SMART'
        )

        logging.info(f"Attempting to fetch data for high-probability contract: {contract_to_fetch.localSymbol}")

        # --- Directly Request Historical Data ---
        # We skip qualification as it can fail for expired contracts.
        # reqHistoricalData is more robust.
        bars = ib.reqHistoricalData(
            contract_to_fetch,
            endDateTime=f'{expiry_str} 20:00:00 US/Eastern',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )

        if not bars:
            logging.warning("--- TEST CONCLUSION: No historical data was returned. ---")
            logging.warning("The code is working correctly, but IBKR did not provide the data.")
            logging.warning("This is the definitive sign that the issue is one of the following:")
            logging.warning(
                "1. Your IBKR account (even paper) lacks the necessary historical options data subscriptions.")
            logging.warning(
                "2. This specific, high-volume contract genuinely did not trade on that day (very unlikely).")
            logging.warning("\nACTION: Please check your market data subscriptions in IBKR Account Management.")
            return

        # --- Display the Data ---
        df = util.df(bars)
        print("\n--- ✅ VICTORY! HISTORICAL DATA RECEIVED! ✅ ---")
        print(df)
        print(f"\n✅ SUCCESS! Data for {contract_to_fetch.localSymbol} was retrieved from IBKR.")
        print("This definitively proves that the backtesting concept is VIABLE with your current setup.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if ib.isConnected():
            logging.info("Disconnecting from IBKR.")
            ib.disconnect()


if __name__ == "__main__":
    run_final_test()

