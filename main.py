import os
import time
import logging
import threading
from math import floor, isnan
from datetime import date, datetime, timezone, timedelta
from typing import Dict

from dotenv import load_dotenv
import ib_insync

ib_insync.util.patchAsyncio()

import config
import custom_logger
import message_parsers
from ib_interface import IBInterface, DiscordScraper
from trailing_stop_manager import TrailingStopManager
from trade_logger import log_trade
from telegram_notifier import TelegramNotifier

RUNTIME_LOG_FILE = 'runtime.log'


class Main:
    def __init__(self):
        load_dotenv()
        self.DISCORD_AUTH_TOKEN = os.getenv("DISCORD_AUTH_TOKEN")
        if not self.DISCORD_AUTH_TOKEN:
            logging.critical("DISCORD_AUTH_TOKEN not found in .env file. Exiting.")
            exit()

        self.channel_profiles = {p["channel_id"]: p for p in config.CHANNEL_PROFILES if p.get("enabled", True)}
        self.channel_ids_to_poll = list(self.channel_profiles.keys())

        self.discord_client = DiscordScraper(self.DISCORD_AUTH_TOKEN)
        self.ib_interface = IBInterface()
        self.parser = message_parsers.CommonParser()
        self.telegram_notifier = TelegramNotifier()
        self.trailing_manager = TrailingStopManager(self.ib_interface, self.telegram_notifier)

        logging.info("Initializing... Fetching latest message IDs to prevent processing old signals.")
        self.last_message_ids = self.get_initial_message_ids()
        logging.info("Initialization complete. Last message IDs are set.")

        if config.EOD_CLOSE_ENABLED:
            threading.Thread(target=self.run_eod_close_loop, daemon=True).start()

        logging.info(f'IBKR client and Scraper initiated for {len(self.channel_profiles)} channels.')

    def get_initial_message_ids(self) -> Dict[str, str]:
        initial_ids = {}
        for channel_id in self.channel_ids_to_poll:
            try:
                messages = self.discord_client.poll_new_messages(channel_id, limit=1)
                if messages:
                    initial_ids[channel_id] = messages[0]['id']
                else:
                    initial_ids[channel_id] = "0"
                self.ib_interface.ib.sleep(1)
            except Exception as e:
                logging.error(f"Could not fetch initial message ID for channel {channel_id}: {e}")
                initial_ids[channel_id] = "0"
        return initial_ids

    def run_eod_close_loop(self):
        eod_time_str = f"{config.EOD_CLOSE_HOUR:02d}:{config.EOD_CLOSE_MINUTE:02d}"
        while True:
            if datetime.now().strftime("%H:%M") >= eod_time_str:
                logging.info(f"EOD CLOSE TRIGGERED. Closing all positions.")
                self.ib_interface.close_all_positions()
                break
            time.sleep(30)

    def run(self):
        logging.info('Initiating main loop... BEHOLD!!')
        while True:
            for channel_id in self.channel_ids_to_poll:
                profile = self.channel_profiles[channel_id]
                logging.debug(f"Polling channel: {profile.get('channel_name', channel_id)}")
                try:
                    messages = self.discord_client.poll_new_messages(channel_id, limit=10)
                    if not messages: continue
                    last_id = self.last_message_ids[channel_id]
                    new_messages = [msg for msg in messages if int(msg['id']) > int(last_id)]
                    if new_messages:
                        self.last_message_ids[channel_id] = new_messages[0]['id']
                        for message in reversed(new_messages): self.process_signal(message, profile)
                except Exception as e:
                    logging.error(f"Error polling channel {channel_id}: {e}", exc_info=True)
                self.ib_interface.ib.sleep(config.DELAY_BETWEEN_CHANNELS)

            try:
                self.trailing_manager.check_trailing_stops()
            except Exception as e:
                logging.error(f"[TRAIL LOOP ERROR] {e}", exc_info=True)

            logging.debug("Full polling cycle complete. Pausing.")
            self.ib_interface.ib.sleep(config.DELAY_AFTER_FULL_CYCLE)

    def process_signal(self, message: Dict, profile: Dict):
        signal_id = message['id']
        channel_name = profile.get('channel_name', message['channel_id'])
        log_prefix = f"Signal #{signal_id} from {channel_name}"

        msg_timestamp = datetime.fromisoformat(message['timestamp']).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - msg_timestamp > timedelta(seconds=config.SIGNAL_MAX_AGE_SECONDS):
            return

        try:
            parsed_signal = self.parser.parse_message(
                message,
                reject_keywords=profile.get('reject_if_contains', []),
                assume_buy=profile.get('assume_buy_on_ambiguous', False)
            )
            if not parsed_signal: return
        except Exception as e:
            logging.error(f'[{log_prefix}] Exception during parsing: {e}', exc_info=True)
            return

        logging.info(f"[{log_prefix}] Successfully parsed signal: {parsed_signal}")

        if parsed_signal['instr'] not in ["BUY", "ADD"]:
            return

        contract = self.ib_interface.create_contract_from_parsed_signal(parsed_signal)
        if not contract: return

        price = self.ib_interface.get_snapshot_price(contract)
        if not price or not (config.MIN_PRICE <= price <= config.MAX_PRICE):
            return

        qty = floor(config.PER_SIGNAL_FUNDS_ALLOCATION / (price * 100))
        if qty <= 0: return

        exit_strategy = profile.get("exit_strategy", {})
        strategy_type = exit_strategy.get("type")
        order_details = {'parsed_symbol': parsed_signal, 'qty': qty}
        trader_name = profile.get("channel_name", "Unknown")

        logging.info(
            f"[{log_prefix}] Placing {profile.get('entry_order_type', 'MKT')} order for {qty}x {contract.localSymbol}")

        trade = self.ib_interface.submit_entry_order(order_details, profile)

        if not trade:
            logging.error(f"[{log_prefix}] Order submission failed.")
            return

        while not trade.isDone():
            self.ib_interface.ib.waitOnUpdate()

        fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice > 0 else price
        logging.info(f"[{log_prefix}] Order filled at average price: {fill_price:.2f}")

        notification_message = (
            f"✅ *BUY Order Executed*\n\n"
            f"*Trader:* `{trader_name}`\n"
            f"*Contract:* `{contract.localSymbol}`\n"
            f"*Quantity:* `{qty}`\n"
            f"*Fill Price:* `${fill_price:.2f}`"
        )
        self.telegram_notifier.send_message(notification_message)

        strategy_details_str = f"{strategy_type}_{exit_strategy.get('pullback_stop_percent', 15)}%"
        log_trade(
            symbol=contract.localSymbol, qty=qty, price=fill_price, action="BUY", reason="entry",
            trader_name=trader_name, strategy_details=strategy_details_str
        )

        safety_net_order_id = None
        if strategy_type == "dynamic_trail":
            safety_net_config = exit_strategy.get("safety_net", {})
            if safety_net_config.get("enabled"):
                trail_percent = safety_net_config.get("native_trail_percent")
                safety_net_trade = self.ib_interface.submit_native_trail_order(order_details, trail_percent)
                if safety_net_trade:
                    safety_net_order_id = safety_net_trade.order.orderId

            self.trailing_manager.add_position(
                symbol=contract.localSymbol, entry_price=fill_price, qty=qty, contract=contract,
                rules=exit_strategy, trader_name=trader_name, strategy_details=strategy_details_str,
                safety_net_order_id=safety_net_order_id
            )
        elif strategy_type == "bracket":
            self.ib_interface.submit_bracket_order(order_details, fill_price, exit_strategy)
        elif strategy_type == "native_trail":
            self.ib_interface.submit_native_trail_order(order_details, exit_strategy.get("trailing_percent"))


if __name__ == '__main__':
    import colorama

    colorama.init()
    custom_logger.setup_logging(console_log_output="stdout", console_log_level="info",
                                console_log_color=True, logfile_file=RUNTIME_LOG_FILE,
                                logfile_log_level="info", logfile_log_color=False,
                                log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s")
    logging.info(f'===================================================================\n')
    main_app = Main()
    main_app.run()
