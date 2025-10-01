import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This version is a "Master Linguist" with professional-grade logic for
    handling ambiguous expiries for both daily and weekly contracts.
    """
    def __init__(self, config):
        self.config = config
        # A simple list of US market holidays. A professional system would use a library.
        self.market_holidays = [
            "2025-01-01", # New Year's Day
            "2025-01-20", # Martin Luther King, Jr. Day
            "2025-02-17", # Washington's Birthday
            "2025-04-18", # Good Friday
            "2025-05-26", # Memorial Day
            "2025-06-19", # Juneteenth National Independence Day
            "2025-07-04", # Independence Day
            "2025-09-01", # Labor Day
            "2025-11-27", # Thanksgiving Day
            "2025-12-25", # Christmas Day
        ]

    def parse_signal(self, text, profile):
        # ... (rest of the function is unchanged)
        pass

    def _cleanup_text(self, text):
        # ... (unchanged)
        pass

    def _find_action(self, text, profile):
        # ... (unchanged)
        pass

    def _parse_pattern_standard(self, text, profile):
        # ... (unchanged)
        pass

    def _parse_pattern_dte(self, text, profile):
        # ... (unchanged)
        pass

    def _parse_pattern_ambiguous_expiry(self, text, profile):
        """
        THE VETERAN PILOT: Parses signals with no date and calculates the
        next available expiry, intelligently handling daily vs. weekly tickers.
        """
        if not profile.get('ambiguous_expiry_enabled', False):
            return None

        action = self._find_action(text, profile)
        if not action:
            return None

        match = re.search(r'([A-Z]{1,5})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text)
        if not match:
            return None

        ticker, strike_str, _, contract_type_char = match.groups()
        
        # --- "Next Available Expiry" Calculator ---
        today = datetime.now()
        target_date = today
        
        if ticker in self.config.daily_expiry_tickers:
            # For daily tickers, find the next valid trading day
            logging.debug(f"Ticker {ticker} found in daily expiry list. Finding next trading day.")
            # If it's a weekday and before market close (e.g., 2 PM MT), today is the target.
            # This is a simplification; a real system would have precise market hours.
            if today.weekday() < 5 and today.hour < 14:
                 pass # Today is the target
            else:
                # Otherwise, start looking from tomorrow
                target_date += timedelta(days=1)

            while target_date.weekday() >= 5 or target_date.strftime("%Y-%m-%d") in self.market_holidays:
                target_date += timedelta(days=1)
        else:
            # For weekly tickers, find the next Friday
            logging.debug(f"Ticker {ticker} not in daily list. Finding next Friday.")
            days_ahead = 4 - today.weekday() # 4 is Friday
            if days_ahead <= 0: # If it's Friday, Saturday, or Sunday
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            # Ensure the next Friday isn't a holiday
            while target_date.strftime("%Y-%m-%d") in self.market_holidays:
                target_date += timedelta(days=7)

        expiry_date = target_date.strftime("%Y%m%d")
        logging.info(f"Ambiguous expiry for {ticker}: using next available expiry '{expiry_date}'")

        return {
            "action": action, "ticker": ticker, "expiry_date": expiry_date,
            "strike": float(strike_str),
            "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
        }

