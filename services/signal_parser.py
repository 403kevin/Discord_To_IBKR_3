import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This version is a "Master Linguist" upgraded with DTE parsing capabilities
    and a robust, multi-pass architecture.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, text, profile):
        """
        Main parsing function. Returns a dictionary on success, None on failure.
        It now uses a multi-pass strategy to try different parsing patterns.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned_text = self._cleanup_text(text)
        
        parsers = [
            self._parse_pattern_dte,
            self._parse_pattern_standard
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
        """Determines the trade action (BTO or STC)."""
        if any(word in text for word in self.config.buzzwords_buy):
            return "BTO"
        if any(word in text for word in self.config.buzzwords_sell):
            return "STC"
        if profile.get('assume_buy_on_ambiguous', False):
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

    def _parse_pattern_dte(self, text, profile):
        """
        Parses the DTE format: "ACTION TICKER STRIKE_TYPE XDTE"
        """
        action = self._find_action(text, profile)
        if not action:
            return None

        match = re.search(r'([A-Z]{1,5})\s+(\d{1,4}(\.\d{1,2})?)\s*([CP])\s+(\d+)DTE', text)

        if not match:
            return None

        ticker, strike_str, _, contract_type_char, dte_str = match.groups()

        try:
            days_to_expiry = int(dte_str)
            
            target_date = datetime.now()
            days_added = 0
            while days_added < days_to_expiry:
                target_date += timedelta(days=1)
                # Monday is 0, Sunday is 6. We only count weekdays.
                if target_date.weekday() < 5:
                    days_added += 1
            
            # Final check for 0DTE case, ensure it's not a weekend.
            if days_to_expiry == 0:
                today = datetime.now()
                if today.weekday() >= 5: # If it's Saturday or Sunday
                    # Roll forward to Monday
                    target_date = today + timedelta(days=(7 - today.weekday()))
                else:
                    target_date = today

            expiry_date = target_date.strftime("%Y%m%d")

            return {
                "action": action, "ticker": ticker, "expiry_date": expiry_date,
                "strike": float(strike_str),
                "contract_type": "CALL" if contract_type_char == 'C' else "PUT"
            }
        except (ValueError, IndexError) as e:
            logging.error(f"DTE parsing failed. DTE string: '{dte_str}'. Error: {e}")
            return None