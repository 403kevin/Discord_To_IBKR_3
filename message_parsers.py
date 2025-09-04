# services/message_parsers.py
import re
from datetime import datetime, timedelta
import logging


def parse_message_to_signal(message_content: str, profile: dict) -> dict | None:
    """
    Parses a raw message string into a structured trade signal dictionary.
    This function is designed to be highly flexible, accommodating various formats.
    """
    # Normalize message for easier parsing
    text = message_content.upper()

    # --- 1. Find Action (BTO/STC) ---
    # Using a placeholder for now, as action logic can get complex (e.g., "scaling in")
    action = "BUY"  # Defaulting to BUY as per most use cases

    # --- 2. Find Ticker ---
    # Looks for a common stock/ETF symbol (1-5 capital letters)
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', text)
    if not ticker_match:
        return None  # A ticker is mandatory
    ticker = ticker_match.group(1)

    # Avoid common words that look like tickers
    if ticker in ["BTO", "STC", "BUY", "SELL", "CALL", "PUT", "ALERT"]:
        # Try to find the next match if the first was a buzzword
        next_ticker_match = re.search(r'\b([A-Z]{1,5})\b', text[ticker_match.end():])
        if next_ticker_match:
            ticker = next_ticker_match.group(1)
        else:
            return None

    # --- 3. Find Strike and Right (Call/Put) ---
    # Looks for a number followed by C or P, e.g., "5000C", "150.5P"
    strike_right_match = re.search(r'(\d+\.?\d*)\s*([CP])', text)
    if not strike_right_match:
        return None  # Strike and Right are mandatory
    strike = float(strike_right_match.group(1))
    right = strike_right_match.group(2)

    # --- 4. Find Expiry ---
    expiry_str = None
    # Look for mm/dd format, e.g., "12/25"
    expiry_match = re.search(r'(\d{1,2}/\d{1,2})', text)
    if expiry_match:
        # Convert mm/dd to YYYYMMDD format
        current_year = datetime.now().year
        try:
            expiry_date = datetime.strptime(f"{current_year}/{expiry_match.group(1)}", "%Y/%m/%d")
            expiry_str = expiry_date.strftime("%Y%m%d")
        except ValueError:
            logging.warning(f"Could not parse expiry date: {expiry_match.group(1)}")
            return None
    elif "0DTE" in text:
        expiry_str = datetime.now().strftime("%Y%m%d")

    # --- 5. Handle Ambiguous Expiry ---
    if not expiry_str:
        if profile.get("assume_next_expiry_on_ambiguous", False):
            # TODO: This is an advanced feature.
            # We would need to call the ib_interface here to find the next
            # available weekly expiration date for the given ticker.
            # For now, we will log a warning.
            logging.warning(
                f"No expiry found for {ticker} and 'assume_next_expiry' is enabled. This feature is not yet implemented.")
            return None  # Rejecting until implemented
        else:
            # If no expiry and the toggle is off, it's not a valid signal for us
            return None

    # --- 6. Construct and Return the Signal ---
    signal = {
        "action": action,
        "symbol": ticker,
        "strike": strike,
        "right": right,
        "expiry": expiry_str,
        "quantity": profile.get("trade_quantity", 1)  # Get quantity from profile
    }

    logging.info(f"Successfully parsed signal: {signal}")
    return signal
