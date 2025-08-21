import logging
from datetime import datetime
import time

import ib_insync
ib_insync.util.patchAsyncio()

import config
import custom_logger
from ib_interface import IBInterface
from message_parsers import CommonParser
from trailing_stop_manager import TrailingStopManager
from trade_logger import log_trade # We can reuse the logger for the output file

# --- CONFIGURATION ---
SIGNALS_FILE = 'signals_to_test.txt'
RESULTS_FILE = 'backtest_results.csv'
# ---------------------

class Backtester:
    def __init__(self):
        self.ib_interface = IBInterface()
        self.parser = CommonParser()
        # We create a dummy notifier since we don't need real-time alerts for a backtest
        class DummyNotifier:
            def send_message(self, text): pass
        self.trailing_manager = TrailingStopManager(self.ib_interface, DummyNotifier())

        # Use the same trade logger, but point it to our results file
        log_trade.LOG_FILE = RESULTS_FILE
        log_trade.HEADER = [
            'signal_timestamp', 'symbol', 'entry_price', 'exit_price', 'pnl', 
            'reason_for_exit', 'trader_name', 'strategy_details'
        ]
        # Clear the results file and write headers
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_trade.HEADER)

    def run(self):
        logging.info("--- Starting Backtest Engine ---")
        
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue

                try:
                    timestamp_str, signal_content = line.split('|', 1)
                    signal_timestamp = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M:%S")
                    
                    logging.info(f"\nProcessing signal from {signal_timestamp}: {signal_content}")
                    self.simulate_trade(signal_timestamp, signal_content.strip())
                    
                    # Pause between requests to respect IBKR's pacing rules
                    self.ib_interface.ib.sleep(2) 

                except Exception as e:
                    logging.error(f"Failed to process line: '{line}'. Error: {e}")

        logging.info("--- Backtest Complete ---")
        self.ib_interface.disconnect()

    def simulate_trade(self, timestamp, content):
        # For now, we assume all signals come from one trader for the test
        trader_name = "Backtest_Trader"
        
        # 1. Parse the signal
        # We create a dummy message object for the parser
        dummy_message = {'content': content, 'id': 'backtest'}
        parsed_signal = self.parser.parse_message(dummy_message, [], False)
        if not parsed_signal: return

        # 2. Create the contract
        contract = self.ib_interface.create_contract_from_parsed_signal(parsed_signal)
        if not contract: return

        # 3. Fetch historical data for the entire day
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

        # 4. Find the entry bar and simulate the BUY
        entry_bar = next((bar for bar in bars if bar.date.replace(tzinfo=None) >= timestamp), None)
        if not entry_bar:
            logging.warning("Signal timestamp is after market close. No entry bar found.")
            return
        
        entry_price = entry_bar.open
        logging.info(f"Simulating BUY at {entry_bar.date} for ${entry_price:.2f}")

        # 5. Simulate the trailing stop logic minute by minute
        exit_price, exit_reason = None, "Market Close"
        # Create a dummy trail manager for this single trade simulation
        sim_trail_manager = TrailingStopManager(self.ib_interface, DummyNotifier())
        sim_trail_manager.add_position(
            symbol=contract.localSymbol, contract=contract, entry_price=entry_price, qty=1,
            rules=config.CHANNEL_PROFILES[0]['exit_strategy'], # Use first profile's rules for now
            trader_name=trader_name, strategy_details="backtest_trail"
        )

        # Find the index of the entry bar to start our loop from there
        start_index = bars.index(entry_bar) + 1
        for bar in bars[start_index:]:
            # This is the magic: we manually feed the historical price to the checker
            sim_trail_manager.ib.get_price_from_stream = lambda symbol: bar.close
            
            sim_trail_manager.check_trailing_stops()
            
            # If the trade was closed, the position will be removed from active_trails
            if not sim_trail_manager.active_trails:
                exit_price = bar.close
                # This is a simplified way to get the reason; a real implementation would be more robust
                exit_reason = "Dynamic Trail" 
                logging.info(f"Simulating SELL at {bar.date} for ${exit_price:.2f} due to {exit_reason}")
                break
        
        # If the loop finishes and we're still in the trade, assume we sold at the close
        if exit_price is None:
            exit_price = bars[-1].close

        # 6. Log the results
        pnl = exit_price - entry_price
        log_trade.log_trade(
            symbol=contract.localSymbol,
            qty=1, # Qty is 1 for simulation
            price=entry_price, # This field is repurposed for entry price
            action=f"{exit_price:.2f}", # Repurposed for exit price
            reason=exit_reason,
            trader_name=trader_name,
            strategy_details=f"pnl:{pnl:.2f}" # Repurposed for P&L
        )

if __name__ == '__main__':
    custom_logger.setup_logging(console_log_output="stdout", console_log_level="info",
                                console_log_color=True, logfile_file='backtest.log',
                                logfile_log_level="info", logfile_log_color=False,
                                log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s")
    
    backtester = Backtester()
    backtester.run()
