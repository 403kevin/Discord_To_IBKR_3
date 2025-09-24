"""
Services/signal_parser.py

Author: 403-Forbidden
Purpose: To parse raw message content from Discord into structured,
         machine-readable trade signal objects. This is the "Master Linguist".
"""
import re
import logging
from datetime import datetime

class SignalParser:
    """
    Parses Discord messages to extract structured trade signal data.
    """
    def parse_messages(self, messages: list[str], profile: dict) -> list[tuple]:
        """
        Processes a list of raw message strings and returns a list of tuples.
        Each tuple contains: (parsed_signal_dict, raw_message_string)
        """
        parsed_signals = []
        for message_content in messages:
            text_lower = message_content.lower()
            
            parsing_rules = profile.get('parsing', {})
            reject_keywords = [word.lower() for word in parsing_rules.get('reject_if_contains', [])]
            jargon_keywords = parsing_rules.get('jargon_words', [])
            ignore_keywords = [word.lower() for word in parsing_rules.get('buzzwords_ignore', [])]

            # --- PRE-FLIGHT FILTER 1: IGNORE BUZZWORDS (NEW LOGIC) ---
            if any(keyword in text_lower for keyword in ignore_keywords):
                logging.info(f"Message ignored due to buzzword filter: '{message_content[:50]}...'")
                continue

            # --- PRE-FLIGHT FILTER 2: REJECT KEYWORDS ---
            if any(keyword in text_lower for keyword in reject_keywords):
                logging.info(f"Message rejected due to keyword filter: '{message_content[:50]}...'")
                continue

            # --- PARSING ---
            cleaned_message = self._strip_jargon(message_content, jargon_keywords)
            signal = self._extract_signal_details(cleaned_message)
            
            if signal:
                # Return the parsed signal AND the original raw text for sentiment analysis
                parsed_signals.append((signal, message_content))
        
        return parsed_signals

    def _strip_jargon(self, message: str, jargon_words: list[str]) -> str:
        """Removes known jargon from the message string to simplify parsing."""
        for word in jargon_words:
            message = re.sub(r'\b' + re.escape(word) + r'\b', '', message, flags=re.IGNORECASE)
        return ' '.join(message.split())

    def _extract_signal_details(self, message: str) -> dict | None:
        """Main parsing function to extract Ticker, Action, Strike, Expiry, etc."""
        try:
            action_match = re.search(r'(BTO|STC)', message, re.IGNORECASE)
            ticker_match = re.search(r'\$?([A-Z]{1,5})\b', message)
            option_match = re.search(r'(\d{1,2}/\d{1,2})\s+(\d+\.?\d*)\s*(C|P|Call|Put)', message, re.IGNORECASE)

            if not all([action_match, ticker_match, option_match]):
                return None

            action = action_match.group(1).upper()
            ticker = ticker_match.group(1)
            expiry_str = option_match.group(1)
            strike = float(option_match.group(2))
            right_raw = option_match.group(3).upper()
            
            right = 'Call' if right_raw.startswith('C') else 'Put'
            
            # TODO: Future Task B - Fix year-rollover bug
            expiry_dt = datetime.strptime(expiry_str, '%m/%d').replace(year=datetime.now().year)
            expiry = expiry_dt.strftime('%Y%m%d')

            return { "action": action, "ticker": ticker, "expiry": expiry, "strike": strike, "right": right }
        except Exception as e:
            logging.warning(f"Failed to parse message into a signal: '{message[:50]}...'. Error: {e}")
            return None