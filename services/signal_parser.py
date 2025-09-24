import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalParser:
    """
    A specialist module for parsing trading signals from raw text messages.
    This is the "Historically-Aware" edition. It can use a provided timestamp
    as its "now" to correctly parse dates from the past.
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
        Finds a potential stock ticker, handling optional '$' prefixes and ignoring jargon.
        """
        potential_tickers = re.findall(r'\$?([A-Z]{1,5})\b', text)
        if not potential_tickers:
            return None

        for ticker in potential_tickers:
            if ticker not in self.config.buzzwords and ticker not in self.config.jargon_words:
                return ticker

        return None

    def _parse_strike_and_type(self, text):
        """
        Finds the strike price and option type, now understanding both
        'C'/'P' and 'CALL'/'PUT'.
        """
        match = re.search(r'(\d+(?:\.\d+)?)\s*(C|P|CALL|PUT)\b', text, re.IGNORECASE)
        if match:
            strike = float(match.group(1))
            option_type_text = match.group(2).upper()

            option_type = 'C' if option_type_text == 'CALL' else 'P' if option_type_text == 'PUT' else option_type_text

            return strike, option_type
        return None, None

    def _parse_expiry(self, text, ticker, profile, context_time=None):
        """
        Finds and formats the expiration date. If context_time is provided,
        it uses that as "now" for its calculations.
        """
        # --- SURGICAL UPGRADE: The Historical Context ---
        # If a historical time is provided (for backtesting), use it. Otherwise, use the real "now".
        now = context_time if context_time else datetime.now()
        # ---
        dt = None

        text_date_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})', text,
                                    re.IGNORECASE)
        if text_date_match:
            month_str = text_date_match.group(1).capitalize()
            day_str = text_date_match.group(2)
            try:
                dt = datetime.strptime(f"{month_str} {day_str}", '%b %d').replace(year=now.year)
            except ValueError:
                dt = None

        if not dt:
            numeric_date_match = re.search(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', text)
            if numeric_date_match:
                expiry_str = numeric_date_match.group(1).replace('-', '/')
                try:
                    parts = expiry_str.split('/')
                    if len(parts) == 3:
                        dt = datetime.strptime(expiry_str, '%m/%d/%y') if len(parts[2]) == 2 else datetime.strptime(
                            expiry_str, '%m/%d/%Y')
                    elif len(parts) == 2:
                        dt = datetime.strptime(expiry_str, '%m/%d').replace(year=now.year)
                except (IndexError, ValueError):
                    dt = None

        if not dt and profile.get("ambiguous_expiry_enabled", False):
            if ticker in self.config.daily_expiry_tickers:
                dt = now
            else:
                today = now.weekday()
                days_until_friday = (4 - today + 7) % 7
                if days_until_friday == 0 and now.time().hour > 14:
                    days_until_friday = 7
                dt = now + timedelta(days=days_until_friday)

        if dt:
            if dt.date() < now.date():
                dt = dt.replace(year=now.year + 1)

            if dt.weekday() >= 5:  # Saturday or Sunday
                logger.warning(f"Parse failed. Calculated expiry '{dt.strftime('%Y%m%d')}' is a weekend.")
                return None
            return dt.strftime('%Y%m%d')

        return None

    def parse_signal_message(self, message_content: str, profile: dict, historical_context_time=None) -> dict or None:
        """
        The main parsing method. It orchestrates the other parsing functions and
        can accept an optional historical timestamp for backtesting accuracy.
        """
        text = message_content.upper()

        for reject_word in profile.get("reject_if_contains", []):
            if reject_word.upper() in text:
                return None

        action = self._parse_action(text)
        ticker = self._parse_ticker(text)
        strike, option_type = self._parse_strike_and_type(text)
        # Pass the historical context to the expiry parser
        expiry = self._parse_expiry(text, ticker, profile, context_time=historical_context_time)

        if all([action, ticker, strike, option_type, expiry]):
            return {
                "action": action,
                "ticker": ticker,
                "strike": strike,
                "option_type": option_type,
                "expiry": expiry
            }

        return None

