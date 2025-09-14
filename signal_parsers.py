import logging
import re
from services.config import Config
from datetime import datetime, timedelta

class SignalParser:
    """
    A specialist class for parsing trading signals from Discord messages.
    It uses a multi-pass approach to handle various signal formats.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def parse_signal_message(self, message_content, profile):
        """
        Parses a message content based on a given profile.
        This is the primary entry point for this class.
        """
        # --- Pre-computation for ambiguous expiry ---
        today = datetime.now()
        
        # Check for rejection keywords first
        if any(keyword.lower() in message_content.lower() for keyword in profile.get('reject_if_contains', [])):
            self.logger.info(f"Message rejected due to keyword filter: '{message_content}'")
            return None

        # --- First Pass: Standard Parser (Action-Ticker-Strike-Expiry) ---
        # Example: BTO SPX 5200C 06/21
        # Example: SELL NVDA 120P 6/21
        pattern = re.compile(
            r'\b(' + '|'.join(self.config.buzzwords) + r')\b'  # Action (BTO, STC, etc.)
            r'\s+([A-Z]{1,5})'                            # Ticker (1-5 capital letters)
            r'\s+(\d{2,5}(?:\.\d{1,2})?)\s*([PC])'         # Strike and Type (e.g., 5200C, 120.5P)
            r'\s+(\d{1,2}/\d{1,2})',                       # Expiry (e.g., 06/21, 6/21)
            re.IGNORECASE
        )
        
        match = pattern.search(message_content)

        if match:
            action, ticker, strike, option_type, expiry_raw = match.groups()
            
            # --- Date Processing ---
            month, day = map(int, expiry_raw.split('/'))
            year = today.year
            # Handle year rollover
            if month < today.month or (month == today.month and day < today.day):
                year += 1
            expiry_str = f"{year}{month:02d}{day:02d}"

            # --- SURGICAL FIX: Validate the extracted ticker ---
            # Check if the extracted "ticker" is actually a buzzword.
            if ticker.upper() in self.config.buzzwords:
                self.logger.warning(f"Parse failed. Ticker '{ticker}' is a buzzword. Rejecting signal.")
                return None
            # --- END SURGICAL FIX ---
            
            # --- SURGICAL FIX: Validate the expiry date ---
            try:
                expiry_dt = datetime.strptime(expiry_str, '%Y%m%d')
                # Monday is 0 and Sunday is 6. We reject Saturdays (5) and Sundays (6).
                if expiry_dt.weekday() >= 5:
                    self.logger.warning(f"Parse failed. Expiry date '{expiry_str}' is a weekend. Rejecting signal.")
                    return None
            except ValueError:
                self.logger.error(f"Parse failed. Invalid date format for expiry: '{expiry_str}'")
                return None
            # --- END SURGICAL FIX ---

            return {
                "action": "BUY" if action.upper() in self.config.buzzwords_buy else "SELL",
                "ticker": ticker.upper(),
                "strike": float(strike),
                "option_type": option_type.upper(),
                "expiry": expiry_str
            }

        # --- Second Pass: Ambiguous Signal (Ticker-Strike-Expiry only) ---
        # This handles signals where the action (BTO/STC) is implied.
        if profile.get('assume_buy_on_ambiguous', False):
            pattern_ambiguous = re.compile(
                r'([A-Z]{1,5})'                               # Ticker
                r'\s+(\d{2,5}(?:\.\d{1,2})?)\s*([PC])'          # Strike and Type
                r'\s+(\d{1,2}/\d{1,2})',                        # Expiry
                re.IGNORECASE
            )
            match_ambiguous = pattern_ambiguous.search(message_content)
            if match_ambiguous:
                ticker, strike, option_type, expiry_raw = match_ambiguous.groups()
                
                # --- Date Processing ---
                month, day = map(int, expiry_raw.split('/'))
                year = today.year
                if month < today.month or (month == today.month and day < today.day):
                    year += 1
                expiry_str = f"{year}{month:02d}{day:02d}"

                # --- SURGICAL FIX (Duplicated for this path): Validate Ticker ---
                if ticker.upper() in self.config.buzzwords:
                    self.logger.warning(f"Ambiguous parse failed. Ticker '{ticker}' is a buzzword.")
                    return None
                # --- END SURGICAL FIX ---

                # --- SURGICAL FIX (Duplicated for this path): Validate Expiry ---
                try:
                    expiry_dt = datetime.strptime(expiry_str, '%Y%m%d')
                    if expiry_dt.weekday() >= 5:
                        self.logger.warning(f"Ambiguous parse failed. Expiry '{expiry_str}' is a weekend.")
                        return None
                except ValueError:
                    self.logger.error(f"Ambiguous parse failed. Invalid date format: '{expiry_str}'")
                    return None
                # --- END SURGICAL FIX ---

                return {
                    "action": "BUY", # Assumption based on profile
                    "ticker": ticker.upper(),
                    "strike": float(strike),
                    "option_type": option_type.upper(),
                    "expiry": expiry_str
                }

        # --- Third Pass: Ambiguous Expiry (e.g., "SPX 0DTE", "NVDA weekly") ---
        if profile.get('ambiguous_expiry_enabled', False):
            # This logic can be complex. For now, we'll handle simple cases.
            # Example: "0DTE" means "zero days to expiry"
            if "0dte" in message_content.lower():
                # Find the parts of the signal around the 0DTE text
                pattern_0dte = re.compile(
                    r'([A-Z]{1,5})\s+(\d{2,5}(?:\.\d{1,2})?)\s*([PC])',
                    re.IGNORECASE
                )
                match_0dte = pattern_0dte.search(message_content)
                if match_0dte:
                    ticker, strike, option_type = match_0dte.groups()
                    
                    # Set expiry to today's date
                    expiry_str = today.strftime('%Y%m%d')
                    
                    # (No need to validate if today is a weekend, IB will reject it anyway,
                    # but a robust implementation would check against a market calendar)
                    
                    return {
                        "action": "BUY", # Assuming buy
                        "ticker": ticker.upper(),
                        "strike": float(strike),
                        "option_type": option_type.upper(),
                        "expiry": expiry_str
                    }

        self.logger.debug(f"Message did not match any known signal format: '{message_content}'")
        return None
