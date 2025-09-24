"""
Services/signal_parser.py

Author: 403-Forbidden
Purpose: To parse raw message content from Discord into structured,
         machine-readable trade signal objects. This is the "Master Linguist".
"""
import re
import logging
from datetime import datetime, timedelta

class SignalParser:
    """
    Parses Discord messages to extract structured trade signal data.
    """
    def parse_messages(self, messages: list[str], profile: dict) -> list[dict]:
        """
        Processes a list of raw message strings and returns a list of parsed signals.
        """
        parsed_signals = []
        for message_content in messages:
            # --- Pre-computation for cleaner parsing ---
            # 1. Normalize to lowercase for case-insensitive checks.
            text_lower = message_content.lower()
            
            # 2. Get the parsing rules from the provided profile.
            reject_keywords = [word.lower() for word in profile['parsing'].get('reject_if_contains', [])]
            jargon_keywords = profile['parsing'].get('jargon_words', [])

            # --- Filtering Logic ---
            # 3. Reject if message contains any forbidden keywords. (FIXED: Case-insensitive)
            if any(keyword in text_lower for keyword in reject_keywords):
                logging.info(f"Message rejected due to keyword filter: '{message_content[:50]}...'")
                continue

            # --- Parsing Logic ---
            # 4. Clean the message by removing jargon. (IMPLEMENTED)
            cleaned_message = self._strip_jargon(message_content, jargon_keywords)
            
            # The core parsing logic attempts to extract trade details.
            signal = self._extract_signal_details(cleaned_message)
            
            if signal:
                parsed_signals.append(signal)
        
        return parsed_signals

    def _strip_jargon(self, message: str, jargon_words: list[str]) -> str:
        """
        Removes known jargon from the message string to simplify parsing.
        Uses regex for whole-word, case-insensitive replacement.
        """
        for word in jargon_words:
            # \b ensures we match whole words only (e.g., "LONG" not "LONGER")
            # re.IGNORECASE makes the replacement case-insensitive.
            message = re.sub(r'\b' + re.escape(word) + r'\b', '', message, flags=re.IGNORECASE)
        # Remove extra whitespace created by the removal.
        return ' '.join(message.split())

    def _extract_signal_details(self, message: str) -> dict | None:
        """
        Main parsing function to extract Ticker, Action, Strike, Expiry, etc.
        This is a simplified example and would need to be very robust in production.
        """
        try:
            # Example patterns - these would need to be very sophisticated.
            action_match = re.search(r'(BTO|STC)', message, re.IGNORECASE)
            ticker_match = re.search(r'\$?([A-Z]{1,5})\b', message) # Matches $TSLA or TSLA
            
            # Example: SPY 10/25 500c
            option_match = re.search(r'(\d{1,2}/\d{1,2})\s+(\d+\.?\d*)\s*(C|P|Call|Put)', message, re.IGNORECASE)

            if not all([action_match, ticker_match, option_match]):
                return None

            action = action_match.group(1).upper()
            ticker = ticker_match.group(1)
            expiry_str = option_match.group(1)
            strike = float(option_match.group(2))
            right_raw = option_match.group(3).upper()
            
            right = 'Call' if right_raw.startswith('C') else 'Put'
            
            # Convert expiry '10/25' to '2025-10-25' format
            # This is a naive implementation that assumes the current year.
            expiry_dt = datetime.strptime(expiry_str, '%m/%d').replace(year=datetime.now().year)
            expiry = expiry_dt.strftime('%Y%m%d')

            return {
                "action": action,
                "ticker": ticker,
                "expiry": expiry,
                "strike": strike,
                "right": right
            }

        except Exception as e:
            logging.warning(f"Failed to parse message into a signal: '{message[:50]}...'. Error: {e}")
            return None