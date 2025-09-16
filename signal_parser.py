import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalParser:
    """
    A specialist module for parsing trading signals from raw text messages.
    This is a battle-hardened, multi-pass parser designed to be flexible.
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
        """Finds a potential stock ticker (e.g., AAPL, SPX)."""
        # A common pattern is an all-caps word of 1-5 letters.
        match = re.search(r'\b([A-Z]{1,5})\b', text)
        if match:
            # Ensure the found "ticker" is not one of our action buzzwords.
            if match.group(1) not in self.config.buzzwords:
                return match.group(1)
        return None

    def _parse_strike_and_type(self, text):
        """Finds the strike price and option type (C or P)."""
        # Looks for a number followed by C or P (e.g., 450C, 120.5P)
        match = re.search(r'(\d+(?:\.\d+)?)\s*([CP])\b', text, re.IGNORECASE)
        if match:
            strike = float(match.group(1))
            option_type = match.group(2).upper()
            return strike, option_type
        return None, None

    def _parse_expiry(self, text):
        """
        Finds and formats the expiration date.
        Handles MM/DD, MM/DD/YY, and MM/DD/YYYY.
        """
        # This regex is designed to find common date formats.
        match = re.search(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', text)
        if not match:
            return None

        expiry_str = match.group(1).replace('-', '/')
        now = datetime.now()
        dt = None

        # Try to parse different date formats.
        try:
            if len(expiry_str.split('/')[2]) == 4:  # MM/DD/YYYY
                dt = datetime.strptime(expiry_str, '%m/%d/%Y')
            elif len(expiry_str.split('/')[2]) == 2:  # MM/DD/YY
                dt = datetime.strptime(expiry_str, '%m/%d/%y')
        except (IndexError, ValueError):
            try:  # MM/DD
                dt = datetime.strptime(expiry_str, '%m/%d').replace(year=now.year)
                # --- SURGICAL UPGRADE: Intelligent Year Calculation ---
                # If the parsed date is in the past for this year, assume it's for next year.
                # This prevents old signals from being misinterpreted.
                if dt < now:
                    dt = dt.replace(year=now.year + 1)
                # --- END SURGICAL UPGRADE ---
            except ValueError:
                return None

        if dt:
            # Final check: Ensure the expiry date is not a weekend.
            if dt.weekday() >= 5:  # Saturday or Sunday
                logger.warning(f"Parse failed. Expiry date '{dt.strftime('%Y%m%d')}' is a weekend. Rejecting signal.")
                return None
            return dt.strftime('%Y%m%d')
        return None

    def parse_signal_message(self, message_content: str, profile: dict) -> dict or None:
        """
        The main parsing method. It orchestrates the other parsing functions.
        """
        text = message_content.upper()

        # --- Rejection Pass ---
        for reject_word in profile.get("reject_if_contains", []):
            if reject_word.upper() in text:
                return None  # Message contains a forbidden word.

        # --- Extraction Pass ---
        action = self._parse_action(text)
        ticker = self._parse_ticker(text)
        strike, option_type = self._parse_strike_and_type(text)
        expiry = self._parse_expiry(text)

        # --- Validation Pass ---
        # A valid signal must have all its core components.
        if all([action, ticker, strike, option_type, expiry]):
            return {
                "action": action,
                "ticker": ticker,
                "strike": strike,
                "option_type": option_type,
                "expiry": expiry
            }

        return None
