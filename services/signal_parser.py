"""
services/signal_parser.py

The Master Linguist - FIXED WITH TICKER WHITELIST
Combines aggressive tokenization with whitelist validation.
Handles 17+ signal formats including Discord embeds.

FIXES:
1. Now validates tickers against whitelist (prevents SELL, WATCH, EOD etc from being parsed as tickers)
2. Aggressive tokenization for embed formats
"""

import re
import logging
from datetime import datetime, timedelta

# Import ticker whitelist
from services.tickers import VALID_TICKERS, is_valid_ticker


class SignalParser:
    """
    The Master Linguist - FIXED FOR EMBED FORMATS + TICKER VALIDATION
    Combines aggressive tokenization from old script with modern multi-format parsing.
    Handles 17+ signal formats including Discord embeds with emojis and extra text.
    """

    def __init__(self, config):
        self.config = config

    def parse_signal(self, raw_message, profile):
        """Main entry point for signal parsing."""
        cleaned_text = self._preprocess_message(raw_message)
        if not cleaned_text:
            return None

        # Attempt multi-step parsing
        parsed_signal = self._parse_multi_step(cleaned_text, profile)

        if parsed_signal:
            logging.debug(f"Successfully parsed signal: {parsed_signal}")
        else:
            logging.debug(f"Failed to parse signal from: '{raw_message}'")

        return parsed_signal

    def _preprocess_message(self, text):
        """
        ENHANCED: Aggressive preprocessing to handle embed formats.
        Strips emojis, "Contract:", "Price:", and other noise.
        """
        if not text:
            return None

        # Remove emojis (non-ASCII characters)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Remove common embed prefixes that break parsing
        text = re.sub(r'\bContract:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bPrice:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bComments?:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bExpiry:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bTicker:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bStrike:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bExpiration:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'__', '', text)
        text = re.sub(r'```', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _parse_multi_step(self, text, profile):
        """
        FIXED: Uses aggressive tokenization + TICKER WHITELIST VALIDATION.
        Splits on spaces, newlines, colons, asterisks to handle embed formats.
        Only accepts tickers that are in the whitelist.
        """
        # Step 1: Aggressive tokenization (from old script)
        # Split on spaces, newlines, colons, asterisks, dollar signs
        parts = re.split(r'[\s\n:*$]+|\*\*', text.upper())
        parts = [p.strip() for p in parts if p.strip()]

        logging.debug(f"Tokenized parts: {parts}")

        # Initialize variables
        ticker = None
        strike = None
        contract_type = None
        exp_month = None
        exp_day = None
        action = self._find_action(text, profile)

        # Create working copy for extraction
        temp_parts = parts.copy()

        # Step 2: Extract DATE first (most specific pattern)
        for part in list(temp_parts):
            # Pattern: "10/15" or "10-15" or "1015"
            date_match = re.match(r'^(\d{1,2})[/-]?(\d{1,2})$', part)
            if date_match:
                exp_month = int(date_match.group(1))
                exp_day = int(date_match.group(2))
                temp_parts.remove(part)
                break
            
            # Pattern: "OCT15" or "OCT 15"
            month_match = re.match(r'^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{1,2})?$', part)
            if month_match:
                month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
                exp_month = month_map[month_match.group(1)]
                if month_match.group(2):
                    exp_day = int(month_match.group(2))
                temp_parts.remove(part)
                break

        # Check for xDTE patterns (0DTE, 1DTE, etc.)
        for part in list(temp_parts):
            dte_match = re.match(r'^(\d+)DTE$', part)
            if dte_match:
                dte = int(dte_match.group(1))
                target_date = self._get_business_day(dte)
                exp_month, exp_day = target_date.month, target_date.day
                temp_parts.remove(part)
                break

        # Step 3: Extract STRIKE and CONTRACT TYPE
        # Pattern 1: "427.5C" or "500P" (combined)
        for part in list(temp_parts):
            match = re.match(r'^(\d+\.?\d*)(C|P|CALL|PUT|CALLS|PUTS)$', part)
            if match:
                strike = float(match.group(1))
                contract_type = 'C' if match.group(2).startswith('C') else 'P'
                temp_parts.remove(part)
                break

        # Pattern 2: "427.5 P" (strike and type separated)
        if strike is None or contract_type is None:
            for i, part in enumerate(temp_parts):
                if part in ['C', 'P', 'CALL', 'PUT', 'CALLS', 'PUTS'] and i > 0:
                    try:
                        strike = float(temp_parts[i - 1])
                        contract_type = 'C' if part.startswith('C') else 'P'
                        temp_parts.pop(i)
                        temp_parts.pop(i - 1)
                        break
                    except (ValueError, IndexError):
                        continue

        # Pattern 3: Separated strike and type (fallback)
        if strike is None or contract_type is None:
            for part in list(temp_parts):
                if re.match(r'^\d+(\.\d+)?$', part):
                    strike = float(part)
                    temp_parts.remove(part)
                    break
            for part in list(temp_parts):
                if part in ['C', 'P', 'CALL', 'PUT', 'CALLS', 'PUTS']:
                    contract_type = 'C' if part.startswith('C') else 'P'
                    temp_parts.remove(part)
                    break

        # Step 4: Extract TICKER (FIXED: Validate against whitelist)
        # Get potential tickers: alphabetic, 1-5 chars
        potential_tickers = [p for p in temp_parts if p.isalpha() and 1 <= len(p) <= 5]
        
        # CRITICAL FIX: Filter to only VALID tickers from whitelist
        valid_potential_tickers = [t for t in potential_tickers if is_valid_ticker(t)]
        
        if valid_potential_tickers:
            # Sort by length (longest first) to prefer full tickers over fragments
            valid_potential_tickers.sort(key=len, reverse=True)
            ticker = valid_potential_tickers[0]
            logging.debug(f"Valid ticker found: {ticker} (from candidates: {potential_tickers})")
        else:
            # Log what was rejected
            if potential_tickers:
                logging.debug(f"Rejected invalid ticker candidates: {potential_tickers}")
            logging.debug(f"No valid ticker found in whitelist")

        # Validation
        if not all([ticker, strike, contract_type]):
            logging.debug(f"Missing components: ticker={ticker}, strike={strike}, type={contract_type}")
            return None

        # Handle expiry date (from old script logic)
        if not exp_month:  # If no date was found...
            if profile.get('ambiguous_expiry_enabled', False):
                if ticker in self.config.daily_expiry_tickers:
                    # Daily expiry tickers (SPX, SPY, QQQ) → 0DTE
                    logging.info(f"Ticker {ticker} is a daily expiry ticker. Defaulting to 0DTE.")
                    today = datetime.now()
                    exp_month, exp_day = today.month, today.day
                else:
                    # Regular stocks → next Friday
                    logging.info(f"No expiry found for {ticker}. Defaulting to next Friday.")
                    next_friday = self._get_next_friday()
                    exp_month, exp_day = next_friday.month, next_friday.day
            else:
                # ambiguous_expiry is disabled and no expiry provided
                logging.debug(f"Parser requires explicit expiry date for {ticker} (ambiguous_expiry_enabled=False)")
                return None

        # Final validation
        if not all([exp_month, exp_day]):
            logging.debug(f"Parser failed to determine expiry date for ticker {ticker}.")
            return None

        # Handle year rollover
        year = datetime.now().year
        current_month = datetime.now().month
        if current_month == 12 and exp_month == 1:
            year += 1

        expiry_date = f"{year}{exp_month:02d}{exp_day:02d}"

        return {
            "action": action,
            "ticker": ticker,
            "expiry_date": expiry_date,
            "strike": strike,
            "contract_type": "CALL" if contract_type == 'C' else "PUT"
        }

    def _find_action(self, text, profile):
        """Determines trade action using channel-specific buzzwords."""
        text_upper = text.upper()
        
        # FIX: Get channel-specific buy words with safe empty list fallback
        channel_buy_words = profile.get('buzzwords_buy', [])

        # Check if any buy word is in the text
        if channel_buy_words and any(word.upper() in text_upper for word in channel_buy_words):
            return "BTO"

        # If no buy word found but profile assumes buy on ambiguous
        if profile.get('assume_buy_on_ambiguous', False):
            return "BTO"

        return None

    def _get_business_day(self, dte: int) -> datetime:
        """Calculate future business day skipping weekends."""
        target_date = datetime.today() + timedelta(days=dte)
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
        return target_date

    def _get_next_friday(self) -> datetime:
        """Get next Friday date."""
        today = datetime.today()
        days_to_friday = (4 - today.weekday() + 7) % 7
        if days_to_friday == 0:  # If today is Friday, get next Friday
            days_to_friday = 7
        next_friday = today + timedelta(days=days_to_friday)
        return next_friday
