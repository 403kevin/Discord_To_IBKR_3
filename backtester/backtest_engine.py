#!/usr/bin/env python3
"""
backtest_engine.py - FIXED VERSION 
BUG FIX #3: Corrects year calculation for historical signals
When backtesting past signals (e.g., September signals in October),
the parser was using datetime.now().year which caused wrong expiry dates.
Now uses the signal timestamp's year instead.
"""

# In the _load_signals() method, add this fix after parsing:

                    if parsed:
                        # FIX BUG #3: For historical signals, correct the expiry year
                        # The parser uses datetime.now().year which breaks for past signals
                        exp_str = parsed['expiry_date']  # Format: YYYYMMDD
                        parsed_year = int(exp_str[:4])
                        signal_year = timestamp.year
                        
                        # If parsed expiry year doesn't match signal year, fix it
                        if parsed_year != signal_year:
                            # Rebuild expiry_date using signal's year
                            parsed['expiry_date'] = f"{signal_year}{exp_str[4:]}"
                            logging.info(f"  ⚠️ Corrected expiry year from {parsed_year} to {signal_year}")
                        
                        parsed['timestamp'] = timestamp
                        parsed['trader'] = trader
                        parsed['raw_signal'] = signal_text
                        signals.append(parsed)
                        logging.info(f"  Line {line_num}: {parsed['ticker']} {parsed['expiry_date']} {parsed['strike']}{parsed['contract_type'][0]}")
                    else:
                        logging.warning(f"  Line {line_num}: Failed to parse: {signal_text}")
