# services/message_parsers.py
import logging
import re
from datetime import datetime

from services.utils import get_next_friday

class MessageParser:
    """
    The bot's "Translator." This is the final, intelligent version that
    correctly distinguishes between action words (like BTO) and the actual
    ticker symbol.
    """
    def __init__(self, config):
        self.config = config

    def _parse_expiry(self, text):
        """Parses various expiry date formats."""
        # Simple mm/dd format
        match = re.search(r'(\d{1,2}/\d{1,2})', text)
        if match:
            current_year = datetime.now().year
            return datetime.strptime(f"{current_year}/{match.group(1)}", "%Y/%m/%d").strftime("%Y%m%d")

        # 0DTE format
        if "0dte" in text.lower():
            return datetime.now().strftime("%Y%m%d")
        
        # Next Friday for ambiguous signals (e.g., "weekly")
        if any(keyword in text.lower() for keyword in ["next week", "weekly"]):
            return get_next_friday().strftime("%Y%m%d")

        return None

    def parse(self, message_content, profile):
        """
        The main parsing logic. It now correctly identifies the action first,
        then finds the ticker that follows it.
        """
        text = message_content.upper()
        
        # --- THIS IS THE CRITICAL FIX ---
        # The new regex now looks for an action word, then captures the ticker
        # that comes AFTER it, preventing the action word from being mistaken
        # for the ticker symbol.
        action_pattern = r'\b(BTO|STC|BUY|SELL)\b\s+([A-Z]{1,5})'
        action_match = re.search(action_pattern, text)

        action = None
        symbol = None
        
        if action_match:
            action_word = action_match.group(1)
            symbol = action_match.group(2)
            action = "BUY" if action_word in ["BTO", "BUY"] else "SELL"
        elif profile["assume_buy_on_ambiguous"]:
            # If no action word, find the first likely ticker
            symbol_match = re.search(r'\b([A-Z]{1,5})\b', text)
            if symbol_match:
                symbol = symbol_match.group(1)
                action = "BUY"
        
        if not action or not symbol:
            return None # Could not determine a valid action and symbol

        # Now, find the strike and right (e.g., 5000C, 150P)
        option_pattern = re.search(r'(\d+\.?\d*)\s*([CP])', text)
        if not option_pattern:
            return None
            
        strike = float(option_pattern.group(1))
        right = option_pattern.group(2)

        # Find the expiry date
        expiry = self._parse_expiry(text)
        
        # Handle ambiguous expiry toggle from config
        if not expiry and profile.get("ambiguous_expiry_enabled"):
            expiry = get_next_friday().strftime("%Y%m%d")
            logging.info(f"No expiry found, defaulting to next Friday: {expiry}")
        
        if not expiry:
            return None

        return {
            "action": action,
            "symbol": symbol,
            "strike": strike,
            "right": right,
            "expiry": expiry
        }

