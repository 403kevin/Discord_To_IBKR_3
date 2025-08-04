import logging
import re
from datetime import datetime

import config
import utils


class CommonParser:
    def parse_message(self, message: dict, reject_keywords: list, assume_buy: bool) -> dict:
        full_content = ""
        try:
            full_content += message.get('content', '')
            for embed in message.get('embeds', []):
                if embed.get('title'): full_content += f" {embed['title']}"
                if embed.get('description'): full_content += f" {embed['description']}"
        except Exception as e:
            logging.error(f"Error combining message content: {e}")
            return {}

        for word in reject_keywords:
            if word.lower() in full_content.lower():
                logging.info(f"Signal rejected due to keyword: '{word}'")
                return {}

        # Use the more robust parsing logic from your original script
        try:
            # This is a simplified adaptation of your original logic.
            # It can be further refined if needed.
            msg_parts = [p.strip().upper() for p in re.split(r'[\s\n:*]+|\*\*', full_content) if p.strip()]
            if not msg_parts: return {}

            # Find Ticker, Strike, and Option Type
            ticker, strike, p_or_c = None, None, None
            for i, part in enumerate(msg_parts):
                # Regex for patterns like: SPY, 500C, 450.5P, TSLA, 120CALL
                match = re.match(r'^(\d+(\.\d+)?)(C|P|CALL|PUTS|PUT|CALLS)$', part)
                if match and i > 0:
                    # Check if the preceding part is a likely ticker (alphabetic)
                    if msg_parts[i - 1].isalpha():
                        strike = float(match.group(1))
                        p_or_c = "C" if match.group(3).startswith('C') else "P"
                        ticker = msg_parts[i - 1]
                        break

            if not all([ticker, strike, p_or_c]):
                logging.debug("Parser could not find a valid Ticker/Strike/Type pattern.")
                return {}

            # Find Expiry
            exp_month, exp_day = None, None
            for part in msg_parts:
                if "/" in part and len(part) >= 3:  # Basic check for MM/DD
                    try:
                        m_str, d_str = part.split('/')
                        exp_month, exp_day = int(m_str), int(d_str)
                        break
                    except (ValueError, IndexError):
                        continue
                elif "DTE" in part:
                    try:
                        expiry = utils.get_business_day(int(part.replace("DTE", "")))
                        exp_month, exp_day = expiry.month, expiry.day
                        break
                    except (ValueError, IndexError):
                        continue

            if not exp_month and ticker in config.DAILY_EXPIRY_TICKERS:
                today = datetime.now()
                exp_month, exp_day = today.month, today.day

            if not all([exp_month, exp_day]):
                logging.debug("Parser could not find a valid expiry date.")
                return {}

            # Find Instruction
            instr = ""
            for word in msg_parts:
                if word in config.BUY_KEYWORDS: instr = "BUY"; break
                if word in config.SELL_KEYWORDS: instr = "SELL"; break
                if word in config.TRIM_KEYWORDS: instr = "TRIM"; break
            if not instr and assume_buy: instr = "BUY"
            if not instr: return {}

            return {
                'underlying': ticker, 'exp_month': exp_month, 'exp_day': exp_day,
                'strike': strike, 'p_or_c': p_or_c.upper(), 'instr': instr, 'id': message['id']
            }

        except Exception as e:
            logging.error(f"An unexpected error occurred during parsing: {e}")
            return {}
