import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalParser:
    """
    A specialist module for parsing trading signals from raw text messages.
    This is the "Genius Translator" edition, capable of reading both numeric
    and text-based dates.
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
        Finds and formats the expiration date. This is the upgraded version
        that can read both numeric (MM/DD) and text-based (Sep 18th) dates.
        """
        now = datetime.now()
        dt = None

        # --- SURGICAL UPGRADE: The "Literacy" Protocol ---
        # First, try to find a text-based date (e.g., "Sep 18th", "OCT 20")
        # This regex looks for a 3-letter month followed by a 1-2 digit day.
        text_date_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})', text, re.IGNORECASE)
        if text_date_match:
            month_str = text_date_match.group(1).capitalize()
            day_str = text_date_match.group(2)
            # We use a format string that can parse "Sep 18"
            try:
                dt = datetime.strptime(f"{month_str} {day_str}", '%b %d').replace(year=now.year)
            except ValueError:
                dt = None
        # --- END UPGRADE ---

        # If a text date wasn't found, fall back to the numeric date parser.
        if not dt:
            numeric_date_match = re.search(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', text)
            if numeric_date_match:
                expiry_str = numeric_date_match.group(1).replace('-', '/')
                try:
                    parts = expiry_str.split('/')
                    if len(parts) == 3 and len(parts[2]) == 4: # MM/DD/YYYY
                        dt = datetime.strptime(expiry_str, '%m/%d/%Y')
                    elif len(parts) == 3 and len(parts[2]) == 2: # MM/DD/YY
                        dt = datetime.strptime(expiry_str, '%m/%d/%y')
                    elif len(parts) == 2: # MM/DD
                        dt = datetime.strptime(expiry_str, '%m/%d').replace(year=now.year)
                except (IndexError, ValueError):
                    dt = None
        
        # If no date was found at all, and it's a daily ticker, assume today.
        if not dt and ticker in self.config.daily_expiry_tickers:
            dt = now

        if dt:
            # If the parsed date is in the past for this year, assume it's for next year.
            if dt.date() < now.date():
                dt = dt.replace(year=now.year + 1)
            
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

