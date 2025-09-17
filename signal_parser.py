import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalParser:
    """
    A specialist module for parsing trading signals from raw text messages.
    This is a battle-hardened, multi-pass parser with restored intelligence
    for handling daily expiry tickers.
    """

    def __init__(self, config):
        self.config = config

    def _parse_action(self, text):
        """Finds the action (BUY or SELL) based on buzzwords."""
        for word in self.config.buzzwords_buy:
            if word in text:
                return "BUY"
        for word in self.config.buzzwords_sell:
            if word in text:
                return "SELL"
        return None

    def _parse_ticker(self, text):
        """
        Finds a potential stock ticker by locating all capitalized words and
        returning the first one that is not an action buzzword.
        """
        potential_tickers = re.findall(r'\b([A-Z]{1,5})\b', text)
        if not potential_tickers:
            return None
        
        for ticker in potential_tickers:
            if ticker not in self.config.buzzwords:
                return ticker
        
        return None

    def _parse_strike_and_type(self, text):
        """Finds the strike price and option type (C or P)."""
        match = re.search(r'(\d+(?:\.\d+)?)\s*([CP])\b', text, re.IGNORECASE)
        if match:
            strike = float(match.group(1))
            option_type = match.group(2).upper()
            return strike, option_type
        return None, None

    def _parse_expiry(self, text, ticker):
        """
        Finds and formats the expiration date.
        This is the upgraded version that understands daily expiry rules.
        """
        match = re.search(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', text)
        if not match:
            # --- SURGICAL UPGRADE for Daily Expiry ---
            # If no date is found, and it's a daily ticker, assume today.
            if ticker in self.config.daily_expiry_tickers:
                dt = datetime.now()
                # If it's a weekend, this will be caught by the check below.
                return dt.strftime('%Y%m%d')
            return None

        expiry_str = match.group(1).replace('-', '/')
        now = datetime.now()
        dt = None

        try:
            parts = expiry_str.split('/')
            if len(parts) == 3 and len(parts[2]) == 4: # MM/DD/YYYY
                dt = datetime.strptime(expiry_str, '%m/%d/%Y')
            elif len(parts) == 3 and len(parts[2]) == 2: # MM/DD/YY
                dt = datetime.strptime(expiry_str, '%m/%d/%y')
            elif len(parts) == 2: # MM/DD
                dt = datetime.strptime(expiry_str, '%m/%d').replace(year=now.year)
                if dt.date() < now.date():
                    dt = dt.replace(year=now.year + 1)
            else:
                return None
        except (IndexError, ValueError):
            return None
        
        if dt:
            # Final check: Ensure the expiry date is not a weekend.
            if dt.weekday() >= 5: # Saturday or Sunday
                logger.warning(f"Parse failed. Expiry date '{dt.strftime('%Y%m%d')}' is a weekend. Rejecting signal.")
                return None
            return dt.strftime('%Y%m%d')
        return None


    def parse_signal_message(self, message_content: str, profile: dict) -> dict or None:
        """
        The main parsing method. It orchestrates the other parsing functions.
        """
        text = message_content.upper()

        for reject_word in profile.get("reject_if_contains", []):
            if reject_word.upper() in text:
                return None

        action = self._parse_action(text)
        ticker = self._parse_ticker(text)
        strike, option_type = self._parse_strike_and_type(text)
        # --- SURGICAL UPGRADE: Pass the ticker to the expiry parser ---
        expiry = self._parse_expiry(text, ticker)

        if all([action, ticker, strike, option_type, expiry]):
            return {
                "action": action,
                "ticker": ticker,
                "strike": strike,
                "option_type": option_type,
                "expiry": expiry
            }
        
        return None

