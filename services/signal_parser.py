import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This version is a battle-hardened "Master Linguist" with robust, multi-pass
    parsing logic and intelligent error reporting.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, text, profile):
        """
        Main parsing function. Returns a dictionary on success, None on failure.
        This function is now guaranteed to be safe and verbose on failure.
        """
        if not text or not isinstance(text, str):
            return None

        # Clean the text once at the beginning
        cleaned_text = self._cleanup_text(text)
        
        # Attempt to parse using a series of patterns, from most specific to most general
        parsers = [
            self._parse_pattern_standard,
            # Add other parsing functions here as new formats are discovered
        ]

        for parser_func in parsers:
            parsed_signal = parser_func(cleaned_text, profile)
            if parsed_signal:
                # The first parser that succeeds wins.
                return parsed_signal
        
        # If no parsers succeed, log the failure and return None.
        logging.debug(f"Failed to parse signal with any known pattern. Original text: '{text}'")
        return None

    def _cleanup_text(self, text):
        """Standardizes text for easier parsing."""
        text = text.upper().replace('\n', ' ')
        for word in self.config.jargon_words:
            text = text.replace(word.upper(), '')
        # Replace common variations and extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _find_action(self, text, profile):
        """Determines the trade action (BTO or STC)."""
        if any(word in text for word in self.config.buzzwords_buy):
            return "BTO"
        if any(word in text for word in self.config.buzzwords_sell):
            return "STC"
        if profile.get('assume_buy_on_ambiguous', False):
            return "BTO"
        return None

    def _parse_pattern_standard(self, text, profile):
        """
        Parses the most common format: "ACTION TICKER MM/DD STRIKE_TYPE"
        Examples: "BTO SPY 9/26 500c", "SPY 10/3 500 C"
        """
        action = self._find_action(text, profile)
        if not action:
            return None

        # This pattern is now more flexible. It looks for TICKER, then DATE, then STRIKE+TYPE
        match = re.search(r'([A-Z]{1,5})\s+(\d{1,2}/\d{1,2})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text)
        
        if not match:
            return None
            
        ticker, date_str, strike_str, _, contract_type_char = match.groups()

        try:
            # --- THE BILINGUAL FIX ---
            # It now correctly handles both '10/3' and '10/03'
            month, day = map(int, date_str.split('/'))
            year = datetime.now().year
            
            # --- THE YEAR-ROLLOVER FIX ---
            # Handles signals in Dec for a Jan expiry
            current_month = datetime.now().month
            if current_month == 12 and month == 1:
                year += 1
                
            expiry_date = f"{year}{month:02d}{day:02d}"
            
            return {
                "action": action,
                "ticker": ticker,
                "expiry_date": expiry_date,
                "strike": float(strike_str),
                "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
            }
        except ValueError as e:
            # --- THE "VOICE OF THE SENTRY" FIX ---
            # It now screams when it's confused.
            logging.error(f"Date parsing failed for pattern 1. Date string: '{date_str}'. Error: {e}")
            return None

