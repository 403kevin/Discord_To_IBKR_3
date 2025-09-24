import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw Discord messages to extract structured trading signal data.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, message_content, profile):
        """
        Main parsing function. Cleans the message and extracts signal details.
        Returns a dictionary with the parsed signal or None if invalid.
        """
        # --- Pre-flight Checks ---
        # 1. Ignore Filter (using buzzwords_ignore)
        if any(word.lower() in message_content.lower() for word in self.config.buzzwords_ignore):
            logging.info("Message ignored due to ignore buzzword: %s", message_content)
            return None, None # Return None for both signal and raw message

        # 2. Rejection Filter (using reject_if_contains from profile)
        # Fix: Make comparison case-insensitive
        if any(word.lower() in message_content.lower() for word in profile['reject_if_contains']):
            logging.info("Signal rejected due to reject word in profile '%s': %s", profile['channel_name'], message_content)
            return None, None

        # --- Cleaning and Extraction ---
        cleaned_content = self._clean_message(message_content)
        signal_details = self._extract_signal_details(cleaned_content, profile)

        if not signal_details:
            return None, None

        # --- Post-flight Validation ---
        if not all(k in signal_details for k in ['ticker', 'action', 'option_type', 'strike', 'expiry_str']):
            logging.warning("Parsed signal is missing one or more key details: %s", signal_details)
            return None, None
        
        logging.info("Successfully parsed signal: %s", signal_details)
        return signal_details, message_content # Return both the parsed details and the original raw message

    def _clean_message(self, message):
        """
        Strips jargon and standardizes the message for easier parsing.
        """
        # Convert to uppercase for consistent processing
        message = message.upper()
        # Remove jargon words
        for word in self.config.jargon_words:
            message = message.replace(word.upper(), "")
        # Remove common characters that can interfere with parsing
        message = re.sub(r'[@\n\t]', ' ', message)
        # Standardize spacing
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def _extract_signal_details(self, cleaned_content, profile):
        """
        Uses regex to find the core components of a trading signal.
        """
        # --- Regex Patterns ---
        # Ticker: Looks for 1-5 uppercase letters, often preceded by $
        ticker_match = re.search(r'\$?([A-Z]{1,5})\b', cleaned_content)
        ticker = ticker_match.group(1) if ticker_match else None

        # Action: Buy or Sell
        action = self._determine_action(cleaned_content, profile)

        # Strike and Option Type: e.g., 500C, 120.5P
        option_match = re.search(r'(\d{1,5}(?:\.\d{1,2})?)\s?(C|P|CALL|PUT)\b', cleaned_content)
        strike = float(option_match.group(1)) if option_match else None
        option_type_str = option_match.group(2) if option_match else None
        option_type = self._normalize_option_type(option_type_str)

        # Expiry: Various formats like 9/20, 09/20, Sep 20
        expiry_match = re.search(r'(\d{1,2}[/-]\d{1,2})|([A-Z]{3}\s\d{1,2})', cleaned_content)
        expiry_str = expiry_match.group(0) if expiry_match else None
        
        # If no expiry is found, check if ambiguous expiry is allowed for this profile
        if not expiry_str and profile.get("ambiguous_expiry_enabled", False):
            expiry_date = self._get_next_friday()
            expiry_str_formatted = expiry_date.strftime('%Y%m%d')
        elif expiry_str:
            expiry_date = self._parse_date(expiry_str)
            expiry_str_formatted = expiry_date.strftime('%Y%m%d') if expiry_date else None
        else:
            expiry_str_formatted = None


        if not all([ticker, action, option_type, strike, expiry_str_formatted]):
            return None

        return {
            "ticker": ticker,
            "action": action,
            "option_type": option_type,
            "strike": strike,
            "expiry_str": expiry_str_formatted
        }
        
    def _determine_action(self, content, profile):
        """Determines if the action is BUY or SELL based on buzzwords."""
        if any(word in content for word in self.config.buzzwords_buy):
            return "BUY"
        if any(word in content for word in self.config.buzzwords_sell):
            return "SELL"
        # Handle ambiguous cases based on profile setting
        if profile.get("assume_buy_on_ambiguous", False):
            return "BUY"
        return None

    def _normalize_option_type(self, type_str):
        """Normalizes CALL/C to 'Call' and PUT/P to 'Put'."""
        if not type_str:
            return None
        if type_str in ["C", "CALL"]:
            return "Call"
        if type_str in ["P", "PUT"]:
            return "Put"
        return None

    def _parse_date(self, date_str):
        """
        Parses various date formats (e.g., '9/20', 'Sep 20') into a datetime object.
        Handles year-rollover for future-dated options.
        """
        now = datetime.now()
        date_str = date_str.replace('/', '-')
        
        try:
            # Try parsing formats like 'MM-DD'
            dt = datetime.strptime(date_str, '%m-%d').replace(year=now.year)
        except ValueError:
            try:
                # Try parsing formats like 'Mon DD', e.g., 'Sep 20'
                dt = datetime.strptime(date_str, '%b %d').replace(year=now.year)
            except ValueError:
                logging.error("Could not parse date format: %s", date_str)
                return None
        
        # --- YEAR-ROLLOVER FIX ---
        # If the parsed date is in the past (e.g., it's December and the signal is for January),
        # assume it's for the next year.
        if dt < now:
            dt = dt.replace(year=now.year + 1)
            
        return dt

    def _get_next_friday(self):
        """Calculates the date of the next Friday."""
        today = datetime.now()
        # weekday() returns 0 for Monday, 4 for Friday.
        days_until_friday = (4 - today.weekday() + 7) % 7
        if days_until_friday == 0 and today.hour >= 14: # If it's Friday past market close, get next Friday
            days_until_friday = 7
        next_friday = today + timedelta(days=days_until_friday)
        return next_friday