# services/message_parsers.py
import re
from datetime import datetime, timedelta

class MessageParser:
    """
    The Translator. This service's only job is to take a raw string of text
    from a Discord message and attempt to translate it into a structured
    trade signal dictionary.
    """
    def __init__(self, config):
        # We can pre-compile regex patterns here for efficiency if needed
        self.config = config

    def parse(self, message_content, profile): # MODIFIED: Now accepts the profile
        """
        Parses a message to find a trade signal.
        
        Returns:
            A dictionary with signal details or None if no valid signal is found.
        """
        # --- 1. Find Action (BUY/SELL) ---
        # This is a simple example; a real implementation would be more robust
        action = None
        if re.search(r'\b(bto|buy|long)\b', message_content, re.IGNORECASE):
            action = "BUY"
        elif re.search(r'\b(stc|sell|short)\b', message_content, re.IGNORECASE):
            action = "SELL"
        
        # --- NEW LOGIC: Handle Ambiguous Signals ---
        if not action and profile.get("assume_buy_on_ambiguous", False):
            action = "BUY"
            
        if not action:
            return None # If we still have no action, we can't proceed.

        # --- 2. Find Ticker ---
        # Looks for a 2-5 letter all-caps word, common for tickers
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', message_content)
        ticker = ticker_match.group(1) if ticker_match else None
        if not ticker:
            return None

        # --- 3. Find Strike and Right (e.g., 5000C, 150.5P) ---
        strike_right_match = re.search(r'(\d+\.?\d*)\s*([CP])', message_content, re.IGNORECASE)
        if not strike_right_match:
            return None
        strike = float(strike_right_match.group(1))
        right = strike_right_match.group(2).upper()

        # --- 4. Find Expiry ---
        # Looks for MM/DD, MM-DD, or "0dte"
        expiry_match = re.search(r'(\d{1,2}[/-]\d{1,2})|0dte', message_content, re.IGNORECASE)
        expiry_date_str = self._format_expiry(expiry_match.group(0) if expiry_match else None)
        
        # If no expiry is found, we could add logic here to get next week's expiry
        # based on another config toggle, as per our README.md
        if not expiry_date_str:
            return None

        signal = {
            "action": action,
            "symbol": ticker,
            "strike": strike,
            "right": right,
            "expiry": expiry_date_str
        }
        return signal

    def _format_expiry(self, date_str):
        """Helper to format found date strings into YYYYMMDD format."""
        if not date_str:
            return None
        
        date_str = date_str.lower()
        if date_str == '0dte':
            return datetime.now().strftime('%Y%m%d')

        # Handle MM/DD or MM-DD
        try:
            # Normalize separator
            date_str = date_str.replace('-', '/')
            dt = datetime.strptime(date_str, '%m/%d')
            # Assume current year
            return dt.replace(year=datetime.now().year).strftime('%Y%m%d')
        except ValueError:
            return None

