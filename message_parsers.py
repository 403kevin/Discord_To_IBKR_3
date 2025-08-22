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

        try:
            msg_parts = [p.strip().upper() for p in re.split(r'[\s\n:*]+|\*\*', full_content) if p.strip()]
            if not msg_parts: return {}

            instr, ticker, strike, p_or_c, exp_month, exp_day = None, None, None, None, None, None
            temp_parts = list(msg_parts)

            for part in temp_parts:
                if part in config.BUY_KEYWORDS: instr = "BUY"; temp_parts.remove(part); break
                if part in config.SELL_KEYWORDS: instr = "SELL"; temp_parts.remove(part); break
                if part in config.TRIM_KEYWORDS: instr = "TRIM"; temp_parts.remove(part); break

            for part in temp_parts:
                if "/" in part and len(part) >= 3:
                    try:
                        m, d = part.split('/'); exp_month, exp_day = int(m), int(d); temp_parts.remove(part); break
                    except (ValueError, IndexError):
                        continue
                elif "DTE" in part:
                    try:
                        expiry = utils.get_business_day(int(part.replace("DTE", "")))
                        exp_month, exp_day = expiry.month, expiry.day;
                        temp_parts.remove(part);
                        break
                    except (ValueError, IndexError):
                        continue

            for part in temp_parts:
                match = re.match(r'^(\d+(\.\d+)?)(C|P|CALL|PUTS|PUT|CALLS)$', part)
                if match:
                    strike = float(match.group(1))
                    p_or_c = "C" if match.group(3).startswith('C') else "P"
                    temp_parts.remove(part);
                    break

            if not strike:
                for i, part in enumerate(temp_parts):
                    if part in ['C', 'P', 'CALL', 'PUT', 'CALLS', 'PUTS'] and i > 0:
                        try:
                            strike = float(temp_parts[i - 1])
                            p_or_c = "C" if part.startswith('C') else "P"
                            temp_parts.pop(i);
                            temp_parts.pop(i - 1);
                            break
                        except (ValueError, IndexError):
                            continue

            potential_tickers = [part for part in temp_parts if part.isalpha()]
            if potential_tickers:
                potential_tickers.sort(key=len, reverse=True)
                ticker = potential_tickers[0]

            if not instr and assume_buy: instr = "BUY"
            if not all([instr, ticker, strike, p_or_c]):
                logging.debug(f"Parser failed validation. Found: {instr}, {ticker}, {strike}, {p_or_c}")
                return {}

            # --- THIS IS THE NEW LOGIC ---
            if not exp_month:  # If no date was found...
                if ticker in config.DAILY_EXPIRY_TICKERS:
                    # Rule Exception: Default to 0DTE for daily tickers
                    logging.info(f"No expiry found for daily ticker {ticker}. Defaulting to 0DTE.")
                    today = datetime.now()
                    exp_month, exp_day = today.month, today.day
                else:
                    # Rule: Default to the next Friday for all other tickers
                    logging.info(f"No expiry found for {ticker}. Defaulting to next Friday.")
                    next_friday = utils.get_next_friday()
                    exp_month, exp_day = next_friday.month, next_friday.day

            if not all([exp_month, exp_day]):
                logging.debug(f"Parser failed to find expiry date for ticker {ticker}.")
                return {}

            return {
                'underlying': ticker, 'exp_month': exp_month, 'exp_day': exp_day,
                'strike': strike, 'p_or_c': p_or_c.upper(), 'instr': instr, 'id': message['id']
            }
        except Exception as e:
            logging.error(f"An unexpected error occurred during parsing: {e}", exc_info=True)
            return {}
