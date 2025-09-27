import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This version is a "Master Linguist" upgraded with logic to handle
    ambiguous actions and expiries.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, text, profile):
        """
        Main parsing function. Returns a dictionary on success, None on failure.
        Uses a multi-pass strategy to try different parsing patterns.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned_text = self._cleanup_text(text)
        
        parsers = [
            self._parse_pattern_standard,
            self._parse_pattern_ambiguous_expiry # Try this if the first fails
        ]

        for parser_func in parsers:
            parsed_signal = parser_func(cleaned_text, profile)
            if parsed_signal:
                return parsed_signal
        
        logging.debug(f"Failed to parse signal with any known pattern. Original text: '{text}'")
        return None

    def _cleanup_text(self, text):
        """Standardizes text for easier parsing."""
        text = text.upper().replace('\n', ' ')
        for word in self.config.jargon_words:
            text = text.replace(word.upper(), '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _find_action(self, text, profile):
        """
        THE "IMPLIED INTENT" PROTOCOL: Determines the trade action (BTO or STC),
        or assumes BTO if the profile allows for it.
        """
        if any(word in text for word in self.config.buzzwords_buy):
            return "BTO"
        if any(word in text for word in self.config.buzzwords_sell):
            return "STC"
        
        # Fallback for signals with no explicit action buzzword
        if profile.get('assume_buy_on_ambiguous', False):
            logging.debug("No action buzzword found, assuming 'BTO' as per profile config.")
            return "BTO"
            
        return None

    def _parse_pattern_standard(self, text, profile):
        """Parses the common format: "ACTION TICKER MM/DD STRIKE_TYPE" """
        action = self._find_action(text, profile)
        if not action:
            return None

        match = re.search(r'([A-Z]{1,5})\s+(\d{1,2}/\d{1,2})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text)
        if not match:
            return None
            
        ticker, date_str, strike_str, _, contract_type_char = match.groups()

        try:
            month, day = map(int, date_str.split('/'))
            year = datetime.now().year
            
            current_month = datetime.now().month
            if current_month == 12 and month == 1:
                year += 1
                
            expiry_date = f"{year}{month:02d}{day:02d}"
            
            return {
                "action": action, "ticker": ticker, "expiry_date": expiry_date,
                "strike": float(strike_str),
                "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
            }
        except ValueError as e:
            logging.error(f"Date parsing failed for standard pattern. Date: '{date_str}'. Error: {e}")
            return None

    def _parse_pattern_ambiguous_expiry(self, text, profile):
        """
        THE "TIME MACHINE" PROTOCOL: Parses signals with no date, like "SPY 300P",
        and calculates the next available Friday expiry.
        """
        if not profile.get('ambiguous_expiry_enabled', False):
            return None

        action = self._find_action(text, profile)
        if not action:
            return None

        # Pattern for signals with a missing date
        match = re.search(r'([A-Z]{1,5})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])', text)
        if not match:
            return None

        ticker, strike_str, _, contract_type_char = match.groups()
        
        # --- "Next Available Expiry" Calculator ---
        # A = today, B = next Friday. We will implement B for robustness.
        today = datetime.now()
        # 4 is Friday. Calculate days until the next Friday.
        days_ahead = 4 - today.weekday()
        if days_ahead <= 0: # If it's already Friday, Saturday, or Sunday
            days_ahead += 7
        
        target_date = today + timedelta(days=days_ahead)
        expiry_date = target_date.strftime("%Y%m%d")
        
        logging.info(f"Ambiguous expiry: using next Friday '{expiry_date}' for signal '{text}'")

        return {
            "action": action,
            "ticker": ticker,
            "expiry_date": expiry_date,
            "strike": float(strike_str),
            "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
        }