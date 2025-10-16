import re
import logging
from datetime import datetime, timedelta


class SignalParser:
    """
    The Master Linguist - now with per-channel buzzword support.
    Parses 17+ signal formats with surgical precision.
    FIXED: ambiguous_expiry now works for ALL tickers, not just daily expiry tickers.
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
        """Clean and normalize the input text."""
        if not text:
            return None

        # Remove markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'__', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _parse_multi_step(self, text, profile):
        """
        The "Step-by-Step Surgeon" - now with strict ticker validation.
        FIXED: ambiguous_expiry now calculates next Friday for ALL tickers when enabled.
        """
        # Split text into parts for component extraction
        msg_parts = [p.strip().upper() for p in re.split(r'[\s\n:*]+|\*\*', text) if p.strip()]
        if not msg_parts:
            return None

        # Initialize variables
        action, ticker, strike, contract_type = None, None, None, None
        exp_month, exp_day = None, None
        temp_parts = list(msg_parts)

        # Step 1: Extract ACTION using channel-specific buzzwords
        action = self._find_action(text, profile)
        if action:
            # Get channel-specific buy words with safe fallback
            channel_buy_words = profile.get('buzzwords_buy', [])
            # Remove buy action words from temp_parts
            for part in list(temp_parts):
                if part in [w.upper() for w in channel_buy_words]:
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
                    dte_value = int(part.replace("DTE", ""))
                    expiry = self._get_business_day(dte_value)
                    exp_month, exp_day = expiry.month, expiry.day
                    temp_parts.remove(part)
                    break
                except (ValueError, IndexError):
                    continue

        # Step 3: Extract STRIKE + TYPE (combined or separate)
        for part in list(temp_parts):
            match = re.match(r'^(\d+(\.\d+)?)(C|P)$', part)
            if match:
                strike = float(match.group(1))
                contract_type = match.group(3)
                temp_parts.remove(part)
                break

        if strike is None or contract_type is None:
            for part in list(temp_parts):
                if re.match(r'^\d+(\.\d+)?$', part):
                    strike = float(part)
                    temp_parts.remove(part)
                    break
            for part in list(temp_parts):
                if part in ['C', 'P', 'CALL', 'PUT']:
                    contract_type = 'C' if part in ['C', 'CALL'] else 'P'
                    temp_parts.remove(part)
                    break

        # Step 4: Extract TICKER (what's left)
        potential_tickers = [p for p in temp_parts if len(p) >= 1 and len(p) <= 5 and p.isalpha()]
        if potential_tickers:
            ticker = potential_tickers[0]

        # Validation
        if not all([ticker, strike, contract_type]):
            logging.debug(f"Missing components: ticker={ticker}, strike={strike}, type={contract_type}")
            return None

        # FIXED: Handle expiry date with corrected logic
        if exp_month and exp_day:
            # Expiry was explicitly provided in the signal
            pass
        elif profile.get('ambiguous_expiry_enabled', False):
            # Calculate expiry when ambiguous_expiry is enabled
            if ticker in self.config.daily_expiry_tickers:
                # Daily expiry tickers (SPX, SPY, QQQ) → 0DTE
                logging.info(f"Ticker {ticker} is a daily expiry ticker. Defaulting to 0DTE.")
                today = datetime.now()
                exp_month, exp_day = today.month, today.day
            else:
                # Regular stocks → next Friday
                logging.info(f"No expiry found for {ticker}. Calculating next Friday.")
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
        # FIX: Get channel-specific buy words with safe empty list fallback
        channel_buy_words = profile.get('buzzwords_buy', [])

        # Check if any buy word is in the text
        if channel_buy_words and any(word.upper() in text.upper() for word in channel_buy_words):
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
