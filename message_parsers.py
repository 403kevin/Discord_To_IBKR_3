# services/message_parsers.py
import logging
import re
from datetime import datetime

from services.utils import get_next_friday

class MessageParser:
    """
    The bot's "Translator." This is the definitive, intelligent version.
    It uses a multi-pass approach to find signal components independently,
    allowing for flexible and robust parsing of various formats (ABCD, BCDA,
    multi-line, etc.), fully respecting the project's core philosophy.
    """
    def __init__(self, config):
        self.config = config
        # Create a dynamic list of action words from the config for easy matching
        self.action_words = [keyword.upper() for keyword in config.buzzwords]

    def _find_action(self, text, profile):
        """Pass 1: Find the action keyword (A)."""
        for word in self.action_words:
            if re.search(r'\b' + word + r'\b', text):
                return "BUY" if word in self.config.buzzwords_buy else "SELL"
        
        # If no action word is found, check the ambiguity rule
        if profile["assume_buy_on_ambiguous"]:
            return "BUY"
            
        return None

    def _find_symbol(self, text):
        """Pass 2: Find the ticker symbol (B)."""
        # Look for 1-5 consecutive capital letters that are NOT action words.
        # This is a common pattern for tickers.
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        for ticker in potential_tickers:
            if ticker not in self.action_words:
                return ticker # Return the first valid ticker found
        return None

    def _find_option_details(self, text):
        """Pass 3: Find the strike price and right (C)."""
        match = re.search(r'(\d+\.?\d*)\s*([CP])', text)
        if match:
            return float(match.group(1)), match.group(2)
        return None, None

    def _find_expiry(self, text, profile):
        """Pass 4: Find the expiry date (D)."""
        # Simple mm/dd format
        match_date = re.search(r'(\d{1,2}/\d{1,2})', text)
        if match_date:
            current_year = datetime.now().year
            return datetime.strptime(f"{current_year}/{match_date.group(1)}", "%Y/%m/%d").strftime("%Y%m%d")

        # 0DTE format
        if "0DTE" in text:
            return datetime.now().strftime("%Y%m%d")
        
        # Check for ambiguous expiry toggle if no other format is found
        if profile.get("ambiguous_expiry_enabled", False):
            # You might want more specific keywords here, e.g., if "weekly" is in text
            return get_next_friday().strftime("%Y%m%d")

        return None

    def parse(self, message_content, profile):
        """
        The main parsing orchestrator. Runs each pass and assembles the signal.
        """
        text = message_content.upper().replace('\n', ' ') # Standardize text
        
        action = self._find_action(text, profile)
        symbol = self._find_symbol(text)
        strike, right = self._find_option_details(text)
        expiry = self._find_expiry(text, profile)

        # A valid signal requires at least a symbol, strike, and right.
        if all([action, symbol, strike, right, expiry]):
            return {
                "action": action,
                "symbol": symbol,
                "strike": strike,
                "right": right,
                "expiry": expiry
            }
        
        logging.debug(f"Parsing failed. Components found: A={action}, B={symbol}, C=({strike},{right}), D={expiry}")
        return None

