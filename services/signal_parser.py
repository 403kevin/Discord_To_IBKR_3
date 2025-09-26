import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This version is battle-hardened with more robust parsing and error handling.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, text, profile):
        """
        Main parsing function. Returns a dictionary on success, None on failure.
        This function is now guaranteed to be safe.
        """
        text = self._cleanup_text(text)
        action = self._find_action(text)
        if not action:
            return None

        # This now returns a dictionary or None
        details = self._extract_signal_details(text, profile)
        
        # If details could not be extracted, abort.
        if not details:
            return None

        # Combine the action with the extracted details
        details['action'] = action
        return details

    def _cleanup_text(self, text):
        """Standardizes text for easier parsing."""
        text = text.upper().replace('\n', ' ')
        # Remove jargon words to reduce noise
        for word in self.config.jargon_words:
            text = text.replace(word.upper(), '')
        return text

    def _find_action(self, text):
        """Determines the trade action (BTO or STC)."""
        if any(word in text for word in self.config.buzzwords_buy):
            return "BTO"
        if any(word in text for word in self.config.buzzwords_sell):
            return "STC"
        return None

    def _extract_signal_details(self, text, profile):
        """
        Extracts Ticker, Expiry, Strike, and Type from the text.
        Returns a dictionary on success, None on failure.
        """
        # --- Pattern 1: Standard "SPY 500c 9/26" ---
        match = re.search(r'([A-Z]{1,5})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])\s+(\d{1,2}/\d{1,2})', text)
        if match:
            ticker, strike_str, _, contract_type_char, date_str = match.groups()
            try:
                # Simple m/d to YYYYMMDD format
                month, day = map(int, date_str.split('/'))
                year = datetime.now().year
                # Handle year rollover
                if month < datetime.now().month:
                    year += 1
                expiry_date = f"{year}{month:02d}{day:02d}"
                
                return {
                    "ticker": ticker,
                    "expiry_date": expiry_date,
                    "strike": float(strike_str),
                    "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
                }
            except ValueError:
                logging.error(f"Could not parse date format from standard pattern: {date_str}")
                return None

        # --- Pattern 2: "AMZN Sep 18 2025 120 Call" (More robust) ---
        match = re.search(r'([A-Z]{1,5})\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\s+(\d{1,4}(\.\d{1,2})?)\s+(CALL|PUT)', text)
        if match:
            ticker, month_str, day_str, year_str, strike_str, _, contract_type = match.groups()
            try:
                date_str = f"{month_str} {day_str} {year_str}"
                expiry_obj = datetime.strptime(date_str, "%b %d %Y")
                expiry_date = expiry_obj.strftime("%Y%m%d")

                return {
                    "ticker": ticker,
                    "expiry_date": expiry_date,
                    "strike": float(strike_str),
                    "contract_type": contract_type
                }
            except ValueError:
                logging.error(f"Could not parse date format from verbose pattern: {date_str}")
                return None
        
        # If no patterns match, it's not a valid signal format we understand.
        return None

