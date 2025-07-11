import logging
import re
from datetime import datetime, timedelta

import config
import utils

class CommonParser:
    """
    A parser designed to extract trade signal details from a raw Discord message.
    It uses a series of regular expressions and keyword checks to find the
    action, ticker, strike, expiry, and option type.
    """
    def __init__(self):
        # We can pre-compile regex patterns here for efficiency if needed
        pass

    def parse_message(self, message: dict, reject_keywords: list, assume_buy: bool) -> dict:
        """
        Parses a raw Discord message dictionary.

        Args:
            message (dict): The raw message object from Discord.
            reject_keywords (list): A list of words that, if found, will cause the signal to be rejected.
            assume_buy (bool): If True, signals without an explicit action keyword will be treated as BUY.

        Returns:
            A dictionary containing the parsed signal details, or an empty dictionary if no valid signal is found.
        """
        # Combine content from message body and any embeds into a single string
        full_content = ""
        try:
            full_content += message.get('content', '')
            for embed in message.get('embeds', []):
                if embed.get('title'):
                    full_content += f" {embed['title']}"
                if embed.get('description'):
                    full_content += f" {embed['description']}"
        except Exception as e:
            logging.error(f"Error combining message content: {e}")
            return {}

        # 1. Reject Filter: Check for reject keywords first
        for word in reject_keywords:
            if word.lower() in full_content.lower():
                logging.info(f"Signal rejected due to keyword: '{word}'")
                return {}

        # 2. Clean and split the message content into processable parts
        # This regex splits by spaces, newlines, colons, and asterisks
        msg_parts = re.split(r'[\s\n:*]+|\*\*', full_content)
        msg_parts = [part.strip().upper() for part in msg_parts if part.strip()]

        if not msg_parts:
            return {}

        # 3. Find Action (BUY, SELL, TRIM)
        instr = ""
        for word in msg_parts:
            if word in config.BUY_KEYWORDS:
                instr = "BUY"
                break
            if word in config.SELL_KEYWORDS:
                instr = "SELL"
                break
            if word in config.TRIM_KEYWORDS:
                instr = "TRIM"
                break
        
        # Apply the "assume buy" logic if no instruction was found
        if not instr and assume_buy:
            instr = "BUY"

        # If still no instruction, we can't proceed
        if not instr:
            logging.debug("No valid action keyword found in message.")
            return {}

        # 4. Find Ticker, Strike, and Option Type (C/P)
        # This is a simplified example; your original complex regex logic can be adapted here.
        # We'll look for a common pattern like "SPY 500C"
        ticker, strike, p_or_c = None, None, None
        for i, part in enumerate(msg_parts):
            # Regex to find patterns like 500C, 450.5P, 120CALL, etc.
            match = re.match(r'^(\d+(\.\d+)?)(C|P|CALL|PUTS|PUT|CALLS)$', part)
            if match and i > 0:
                strike = float(match.group(1))
                p_or_c = "C" if match.group(3).startswith('C') else "P"
                ticker = msg_parts[i-1] # Assume the ticker is the word before the strike/call
                break

        if not all([ticker, strike, p_or_c]):
            logging.debug(f"Could not parse Ticker/Strike/Type from message parts: {msg_parts}")
            return {}

        # 5. Find Expiry Date (e.g., MM/DD or DTE format)
        exp_month, exp_day = None, None
        for part in msg_parts:
            if "/" in part:
                try:
                    month_str, day_str = part.split('/')
                    exp_month, exp_day = int(month_str), int(day_str)
                    break
                except ValueError:
                    continue # Not a valid date format
            elif "DTE" in part:
                try:
                    dte_days = int(part.replace("DTE", ""))
                    expiry_date = utils.get_business_day(dte_days)
                    exp_month, exp_day = expiry_date.month, expiry_date.day
                    break
                except (ValueError, IndexError):
                    continue

        # Default to daily expiry for specific tickers if no date found
        if not exp_month and ticker in config.DAILY_EXPIRY_TICKERS:
            today = datetime.now()
            exp_month, exp_day = today.month, today.day
        
        if not all([exp_month, exp_day]):
            logging.debug(f"Could not parse expiry date from message parts: {msg_parts}")
            return {}

        # 6. Assemble and return the final parsed signal
        parsed_signal = {
            'underlying': ticker,
            'exp_month': exp_month,
            'exp_day': exp_day,
            'strike': strike,
            'p_or_c': p_or_c,
            'instr': instr,
            'id': message['id']
        }
        return parsed_signal

