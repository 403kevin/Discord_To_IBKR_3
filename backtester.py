import logging
from datetime import datetime
import time
import csv

import ib_insync

ib_insync.util.patchAsyncio()

import config
import custom_logger
from ib_interface import IBInterface
from message_parsers import CommonParser
from trailing_stop_manager import TrailingStopManager
from trade_logger import log_trade

# --- CONFIGURATION ---
SIGNALS_FILE = 'signals_to_test.txt'
RESULTS_FILE = 'backtest_results.csv'


# ---------------------

class Backtester:
    def __init__(self):
        self.ib_interface = IBInterface()
        self.parser = CommonParser()

        class DummyNotifier:
            def send_message(self, text): pass

        self.trailing_manager = TrailingStopManager(self.ib_interface, DummyNotifier())

        # Use the same trade logger, but point it to our results file
        log_trade.LOG_FILE = RESULTS_FILE
        log_trade.HEADER = [
            'signal_timestamp', 'symbol', 'entry_price', 'exit_price', 'pnl',
            'reason_for_exit', 'trader_name', 'strategy_details'
        ]
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_trade.HEADER)

    def run(self):
        logging.info("--- Starting Backtest Engine ---")

        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # --- FIX 1: Skip empty lines and comments ---
                if not line or line.startswith('#') or '|' not in line:
                    continue

                try:
                    timestamp_str, signal_content = line.split('|', 1)
                    signal_timestamp = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M:%S")

                    logging.info(f"\nProcessing signal from {signal_timestamp}: {signal_content}")
                    self.simulate_trade(signal_timestamp, signal_content.strip())

                    # Pause between requests to respect IBKR's pacing rules
                    self.ib_interface.ib.sleep(2)

                except Exception as e:
                    logging.error(f"Failed to process line: '{line}'. Error: {e}", exc_info=True)

        logging.info("--- Backtest Complete ---")
        self.ib_interface.disconnect()

    def simulate_trade(self, timestamp, content):
        trader_name = "Backtest_Trader"

        dummy_message = {'content': content, 'id': 'backtest'}
        parsed_signal = self.parser.parse_message(dummy_message, [], False)
        if not parsed_signal: return

        # --- FIX 2: Pass the signal's date as the reference date ---
        contract = self.ib_interface.create_contract_from_parsed_signal(parsed_signal, reference_date=timestamp.date())
        if not contract: return

        logging.info(f"Fetching historical 1-min data for {contract.localSymbol} on {timestamp.date()}...")
        bars = self.ib_interface.ib.reqHistoricalData(
            contract,
            endDateTime=timestamp.date().strftime("%Y%m%d 23:59:59"),
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True
        )
        if not bars:
            logging.warning("No historical data found for this contract.")
            return

        entry_bar = next((bar for bar in bars if bar.date.replace(tzinfo=None) >= timestamp), None)
        if not entry_bar:
            logging.warning("Signal timestamp is after market close. No entry bar found.")
            return

        entry_price = entry_bar.open
        logging.info(f"Simulating BUY at {entry_bar.date} for ${entry_price:.2f}")

        exit_price, exit_reason = None, "Market Close"

        sim_trail_manager = TrailingStopManager(self.ib_interface, self.trailing_manager.telegram_notifier)
        sim_trail_manager.add_position(
            symbol=contract.localSymbol, contract=contract, entry_price=entry_price, qty=1,
            rules=config.CHANNEL_PROFILES[0]['exit_strategy'],
            trader_name=trader_name, strategy_details="backtest_trail"
        )

        start_index = bars.index(entry_bar) + 1
        for bar in bars[start_index:]:
            # Manually feed the historical price to the checker's stream function
            sim_trail_manager.ib.get_price_from_stream = lambda symbol: bar.close

            sim_trail_manager.check_trailing_stops()

            if not sim_trail_manager.active_trails:
                exit_price = bar.close
                # A more robust way to get the reason would be needed in a full implementation
                exit_reason = "Dynamic Trail"
                logging.info(f"Simulating SELL at {bar.date} for ${exit_price:.2f} due to {exit_reason}")
                break

        if exit_price is None:
            exit_price = bars[-1].close

        pnl = exit_price - entry_price

        log_trade(
            symbol=contract.localSymbol, qty=1, price=entry_price,
            action=f"{exit_price:.2f}", reason=exit_reason,
            trader_name=trader_name, strategy_details=f"pnl:{pnl:.2f}"
        )


if __name__ == '__main__':
    custom_logger.setup_logging(console_log_output="stdout", console_log_level="info",
                                console_log_color=True, logfile_file='backtest.log',
                                logfile_log_level="info", logfile_log_color=False,
                                log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s")

    backtester = Backtester()
    backtester.run()
