import re
from datetime import datetime, timedelta
import logging

class SignalParser:
    """
    Parses raw text from Discord messages into structured trade signals.
    ENHANCED: Now uses per-channel buzzwords from profile configuration.
    """
    def __init__(self, config):
        self.config = config
        # Define common trading terms that should NOT be treated as tickers
        self.trading_terms = {
            'SIZED', 'PLACE', 'BELOW', 'ABOVE', 'SWEEP', 'BLOCK', 'TRADE',
            'ALERT', 'FLOW', 'LARGE', 'SMALL', 'HUGE', 'MEGA', 'SUPER',
            'HEAVY', 'LIGHT', 'FAST', 'SLOW', 'QUICK', 'PRINT', 'ORDER',
            'FILLED', 'PARTIAL', 'FULL', 'SCALE', 'LAYER', 'ADDED', 'MORE',
            'LESS', 'SOLD', 'BOUGHT', 'CLOSED', 'OPENED', 'TRIMMED',
            'UNUSUAL', 'OPTION', 'OPTIONS', 'ACTIVITY', 'BULLISH', 'BEARISH',
            'NEUTRAL', 'MIXED', 'STRONG', 'WEAK', 'INSIDE', 'OUTSIDE'
        }
        # Common valid tickers (expand this list as needed)
        self.known_tickers = {
            'SPY', 'SPX', 'SPXW', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY',
            'TSLA', 'AAPL', 'NVDA', 'AMD', 'AMZN', 'META', 'GOOGL', 'MSFT',
            'NFLX', 'BABA', 'SHOP', 'SQ', 'PYPL', 'COIN', 'ROKU', 'SNAP',
            'UBER', 'LYFT', 'ABNB', 'HOOD', 'SOFI', 'PLTR', 'AI', 'PATH',
            'BA', 'GS', 'JPM', 'BAC', 'WFC', 'C', 'MS', 'V', 'MA', 'AXP',
            'PENN', 'DKNG', 'WYNN', 'MGM', 'CZR', 'LVS', 'CHPT', 'RIVN',
            'F', 'GM', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'RIDE',
            'XOM', 'CVX', 'COP', 'OXY', 'SLB', 'HAL', 'MRO', 'DVN',
            'PFE', 'MRNA', 'JNJ', 'LLY', 'ABBV', 'BMY', 'MRK', 'GILD',
            'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'NTES', 'BILI',
            'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
            'AAL', 'DAL', 'UAL', 'LUV', 'JETS', 'CCL', 'RCL', 'NCLH',
            'MARA', 'RIOT', 'BTBT', 'MSTR', 'CLSK', 'CAN', 'HUT', 'BITF',
            'ORCL', 'CRM', 'ADBE', 'NOW', 'INTU', 'CSCO', 'IBM', 'INTC',
            'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ASML',
            'CONS', 'DOWN'
        }

    def parse_signal(self, text, profile):
        """
        Main parsing function using per-channel buzzwords from profile.
        Returns a dictionary on success, None on failure.
        """
        if not text or not isinstance(text, str):
            return None

        # Clean the text once at the beginning
        cleaned_text = self._cleanup_text(text)
        
        # Use the multi-step extraction method
        parsed_signal = self._parse_multi_step(cleaned_text, profile)
        
        if parsed_signal:
            return parsed_signal
        
        # If parsing fails, log for debugging
        logging.debug(f"Failed to parse signal. Original text: '{text}'")
        return None

    def _cleanup_text(self, text):
        """Standardizes text for easier parsing."""
        text = text.upper()
        text = text.replace('$', '')
        text = text.replace(' CALL', 'C')
        text = text.replace(' PUT', 'P')
        text = text.replace(' CALLS', 'C')
        text = text.replace(' PUTS', 'P')
        
        # Remove jargon words
        for word in self.config.jargon_words:
            text = text.replace(word.upper(), '')
        
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text

    def _parse_multi_step(self, text, profile):
        """
        Multi-step extraction method using per-channel buzzwords.
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
            # Get channel-specific buy words or use global defaults
            channel_buy_words = profile.get('buzzwords_buy', self.config.buzzwords_buy)
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
            match = re.match(r'^(\d+(\.\d+)?)(C|P|CALL|CALLS|PUT|PUTS)$', part)
            if match:
                strike = float(match.group(1))
                contract_type = "C" if match.group(3).startswith('C') else "P"
                temp_parts.remove(part)
                break

        # If not found, try separate format
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

        # Step 4: Extract TICKER with validation
        potential_tickers = [
            part for part in temp_parts 
            if part.isalpha() 
            and len(part) <= 5 
            and part not in self.trading_terms
        ]
        
        if potential_tickers:
            # First, check for known tickers
            for part in potential_tickers:
                if part in self.known_tickers:
                    ticker = part
                    logging.debug(f"Found known ticker: {ticker}")
                    break
            
            # If no known ticker found, validate remaining candidates
            if not ticker and potential_tickers:
                valid_candidates = []
                for candidate in potential_tickers:
                    if (len(candidate) >= 1 and len(candidate) <= 5 and 
                        not candidate.startswith('X') and
                        candidate not in ['A', 'I', 'U', 'Y']):
                        valid_candidates.append(candidate)
                
                if valid_candidates:
                    valid_candidates.sort(key=len, reverse=True)
                    ticker = valid_candidates[0]
                    logging.debug(f"Selected ticker by length: {ticker}")

        # Step 5: Validate we have minimum required components
        if not all([action, ticker, strike, contract_type]):
            logging.debug(f"Parser failed validation. Found: action={action}, ticker={ticker}, strike={strike}, type={contract_type}")
            return None

        # Step 6: Handle ambiguous expiry
        if not exp_month and profile.get('ambiguous_expiry_enabled', False):
            if ticker in self.config.daily_expiry_tickers:
                logging.info(f"No expiry found for daily ticker {ticker}. Defaulting to 0DTE.")
                today = datetime.now()
                exp_month, exp_day = today.month, today.day
            else:
                logging.info(f"No expiry found for {ticker}. Defaulting to next Friday.")
                next_friday = self._get_next_friday()
                exp_month, exp_day = next_friday.month, next_friday.day

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
        # Get channel-specific buy words or use global defaults
        channel_buy_words = profile.get('buzzwords_buy', self.config.buzzwords_buy)
        
        # Check if any buy word is in the text
        if any(word.upper() in text.upper() for word in channel_buy_words):
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
        next_friday = today + timedelta(days=days_to_friday)
        return next_friday