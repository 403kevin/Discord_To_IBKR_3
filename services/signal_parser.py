import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    This is the battle-tested parser ported from the working Scraping_Old repo,
    adapted to the new architecture. Supports all 8 format variations and XDTE logic.
    """
    def __init__(self, config):
        self.config = config

    def parse_signal(self, text, profile):
        """
        Main parsing function. Returns a dictionary on success, None on failure.
        This uses the proven multi-step extraction approach from the old repo.
        """
        if not text or not isinstance(text, str):
            return None

        # Clean the text once at the beginning
        cleaned_text = self._cleanup_text(text)
        
        # Use the multi-step extraction method (format-agnostic)
        parsed_signal = self._parse_multi_step(cleaned_text, profile)
        
        if parsed_signal:
            return parsed_signal
        
        # If parsing fails, log for debugging
        logging.debug(f"Failed to parse signal. Original text: '{text}'")
        return None

    def _cleanup_text(self, text):
        """Standardizes text for easier parsing."""
        # Preserve newlines for splitting, but uppercase first
        text = text.upper()
        
        # Remove dollar signs before strikes
        text = text.replace('$', '')
        
        # Normalize "CALL"/"PUT" to "C"/"P" for consistent matching
        text = text.replace(' CALL', 'C')
        text = text.replace(' PUT', 'P')
        text = text.replace(' CALLS', 'C')
        text = text.replace(' PUTS', 'P')
        
        # Remove jargon words
        for word in self.config.jargon_words:
            text = text.replace(word.upper(), '')
        
        # Keep newlines for multiline format support, but normalize other whitespace
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text

    def _parse_multi_step(self, text, profile):
        """
        Multi-step extraction method ported from Scraping_Old/message_parsers.py.
        This is format-agnostic - extracts components independently regardless of order.
        """
        # Split text into parts for component extraction
        msg_parts = [p.strip().upper() for p in re.split(r'[\s\n:*]+|\*\*', text) if p.strip()]
        if not msg_parts:
            return None

        # Initialize variables
        action, ticker, strike, contract_type = None, None, None, None
        exp_month, exp_day = None, None
        temp_parts = list(msg_parts)

        # Step 1: Extract ACTION (BTO, STC, etc.)
        action = self._find_action(text, profile)
        if action:
            # FIX: Remove only BUY action words from temp_parts
            # We don't check buzzwords_sell anymore since it doesn't exist
            for part in list(temp_parts):
                if part in self.config.buzzwords_buy:
                    temp_parts.remove(part)

        # Step 2: Extract DATE (MM/DD or XDTE format)
        for part in list(temp_parts):
            if "/" in part and len(part) >= 3:
                try:
                    m, d = part.split('/')
                    exp_month, exp_day = int(m), int(d)
                    temp_parts.remove(part)
                    break
                except (ValueError, IndexError):
                    continue
            elif "DTE" in part:
                try:
                    # Handle 0DTE, 1DTE, 2DTE, etc.
                    dte_value = int(part.replace("DTE", ""))
                    expiry = self._get_business_day(dte_value)
                    exp_month, exp_day = expiry.month, expiry.day
                    temp_parts.remove(part)
                    break
                except (ValueError, IndexError):
                    continue

        # Step 3: Extract STRIKE + TYPE (combined or separate)
        # Try combined format first (e.g., "170C", "4500P")
        for part in list(temp_parts):
            match = re.match(r'^(\d+(\.\d+)?)(C|P|CALL|CALLS|PUT|PUTS)$', part)
            if match:
                strike = float(match.group(1))
                contract_type = "C" if match.group(3).startswith('C') else "P"
                temp_parts.remove(part)
                break

        # If not found, try separate format (e.g., "170 C")
        if not strike:
            for i, part in enumerate(temp_parts):
                if part in ['C', 'P', 'CALL', 'PUT', 'CALLS', 'PUTS'] and i > 0:
                    try:
                        strike = float(temp_parts[i - 1])
                        contract_type = "C" if part.startswith('C') else "P"
                        temp_parts.pop(i)
                        temp_parts.pop(i - 1)
                        break
                    except (ValueError, IndexError):
                        continue

        # Step 4: Extract TICKER (alphabetic-only parts, longest wins)
        potential_tickers = [part for part in temp_parts if part.isalpha() and len(part) <= 5]
        if potential_tickers:
            potential_tickers.sort(key=len, reverse=True)
            ticker = potential_tickers[0]

        # Step 5: Validate we have minimum required components
        if not all([action, ticker, strike, contract_type]):
            logging.debug(f"Parser failed validation. Found: action={action}, ticker={ticker}, strike={strike}, type={contract_type}")
            return None

        # Step 6: Handle ambiguous expiry (no date found)
        if not exp_month and profile.get('ambiguous_expiry_enabled', False):
            if ticker in self.config.daily_expiry_tickers:
                # Default to 0DTE for daily expiry tickers
                logging.info(f"No expiry found for daily ticker {ticker}. Defaulting to 0DTE.")
                today = datetime.now()
                exp_month, exp_day = today.month, today.day
            else:
                # Default to next Friday for weekly options
                logging.info(f"No expiry found for {ticker}. Defaulting to next Friday.")
                next_friday = self._get_next_friday()
                exp_month, exp_day = next_friday.month, next_friday.day

        # Final validation: Must have a date
        if not all([exp_month, exp_day]):
            logging.debug(f"Parser failed to determine expiry date for ticker {ticker}.")
            return None

        # Step 7: Handle year rollover
        year = datetime.now().year
        current_month = datetime.now().month
        if current_month == 12 and exp_month == 1:
            year += 1
        
        expiry_date = f"{year}{exp_month:02d}{exp_day:02d}"

        # Return the structured signal
        return {
            "action": action,
            "ticker": ticker,
            "expiry_date": expiry_date,
            "strike": strike,
            "contract_type": "CALL" if contract_type == 'C' else "PUT"
        }

    def _find_action(self, text, profile):
        """Determines the trade action (BTO or STC)."""
        if any(word in text for word in self.config.buzzwords_buy):
            return "BTO"
        # FIX: We no longer check buzzwords_sell since it was moved to buzzwords_ignore
        # Any sell words in the message would have already been rejected at the signal_processor level
        if profile.get('assume_buy_on_ambiguous', False):
            return "BTO"
        return None

    def _get_business_day(self, dte: int) -> datetime:
        """
        Calculates a future business day by adding DTE days to today,
        skipping weekends. Ported from Scraping_Old/utils.py
        """
        target_date = datetime.today() + timedelta(days=dte)
        # weekday() returns 5 for Saturday and 6 for Sunday
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
        return target_date

    def _get_next_friday(self) -> datetime:
        """
        Calculates the date of the upcoming Friday.
        If today is Friday, returns today's date.
        Ported from Scraping_Old/utils.py
        """
        today = datetime.today()
        # weekday() returns 0 for Monday and 4 for Friday
        days_to_friday = (4 - today.weekday() + 7) % 7
        next_friday = today + timedelta(days=days_to_friday)
        return next_friday
