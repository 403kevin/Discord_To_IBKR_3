import os
import time
import logging
import threading
from math import floor, isnan
from datetime import date, datetime, timezone, timedelta
from typing import Dict

import ib_insync
from dotenv import load_dotenv

import config
import custom_logger
import message_parsers
import ib_interface
from trailing_stop_manager import TrailingStopManager
from trade_logger import log_trade

RUNTIME_LOG_FILE = 'runtime.log'

class Main:
    def __init__(self):
        """
        Initializes the bot, loading configurations and setting up interfaces.
        """
        load_dotenv()
        self.DISCORD_AUTH_TOKEN = os.getenv("DISCORD_AUTH_TOKEN")
        if not self.DISCORD_AUTH_TOKEN:
            logging.critical("DISCORD_AUTH_TOKEN not found in .env file. Exiting.")
            exit()

        # Build a lookup dictionary from channel profiles for quick access
        self.channel_profiles = {
            profile["channel_id"]: profile
            for profile in config.CHANNEL_PROFILES if profile.get("enabled", True)
        }
        # Keep a simple list of channel IDs to iterate over
        self.channel_ids_to_poll = list(self.channel_profiles.keys())

        # State management to track the last seen message ID for each channel
        self.last_message_ids = {channel_id: "0" for channel_id in self.channel_ids_to_poll}

        # Initialize interfaces
        self.discord_client = ib_interface.DiscordScraper(self.DISCORD_AUTH_TOKEN)
        self.ib_interface = ib_interface.IBInterface()
        self.parser = getattr(message_parsers, 'CommonParser')()
        self.trailing_manager = TrailingStopManager(self.ib_interface)

        # Start the trailing stop manager loop in a separate thread
        threading.Thread(target=self.run_trailing_loop, daemon=True).start()

        logging.info(f'IBKR client and Scraper initiated.')
        logging.info(f'Monitoring {len(self.channel_profiles)} enabled channel profiles.')

    def run_trailing_loop(self):
        """
        Runs the trailing stop manager in a background thread.
        """
        while True:
            try:
                self.trailing_manager.check_trailing_stops()
            except Exception as e:
                logging.error(f"[TRAIL LOOP ERROR] {e}", exc_info=True)
            time.sleep(5) # Check stops every 5 seconds

    def run(self):
        """
        Main sequential polling loop.
        Cycles through the configured channels, scrapes messages, and processes signals.
        """
        logging.info(f'Initiating sequential polling loop... BEHOLD!!')
        while True:
            # Iterate through each channel profile
            for channel_id in self.channel_ids_to_poll:
                try:
                    profile = self.channel_profiles[channel_id]
                    logging.debug(f"Polling channel: {profile.get('channel_name', channel_id)}")

                    # Scrape new messages
                    all_messages = self.discord_client.poll_new_messages(channel_id, limit=10)
                    if not all_messages:
                        continue

                    # Filter for messages that are actually new
                    last_id = self.last_message_ids[channel_id]
                    new_messages = [msg for msg in all_messages if int(msg['id']) > int(last_id)]

                    if new_messages:
                        # Update the last seen message ID for this channel
                        self.last_message_ids[channel_id] = new_messages[0]['id']
                        
                        # Process newest to oldest
                        new_messages.reverse()
                        for message in new_messages:
                            self.process_signal(message, profile)

                except Exception as e:
                    logging.error(f"Error polling channel {channel_id}: {e}", exc_info=True)

                # Wait before polling the next channel
                time.sleep(config.DELAY_BETWEEN_CHANNELS)

            # Wait after completing a full cycle
            logging.debug("Full polling cycle complete. Pausing.")
            time.sleep(config.DELAY_AFTER_FULL_CYCLE)

    def process_signal(self, message: Dict, profile: Dict):
        """
        Processes a single scraped message based on the rules from its channel profile.
        """
        signal_id = message['id']
        channel_name = profile.get('channel_name', message['channel_id'])
        log_prefix = f"Signal #{signal_id} from {channel_name}"

        # Check for stale message
        msg_timestamp = datetime.fromisoformat(message['timestamp']).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - msg_timestamp > timedelta(seconds=config.SIGNAL_MAX_AGE_SECONDS):
            logging.info(f"[{log_prefix}] Skipping stale signal.")
            return

        # Parse the message content
        try:
            # Pass the profile-specific keywords to the parser
            parsed_signal = self.parser.parse_message(
                message,
                reject_keywords=profile.get('reject_if_contains', []),
                assume_buy=profile.get('assume_buy_on_ambiguous', False)
            )
            if not parsed_signal:
                return # Parser rejected or found no valid signal
        except Exception as e:
            logging.error(f'[{log_prefix}] Exception during parsing: {e}', exc_info=True)
            return

        # --- Execution Logic (simplified for clarity, will be expanded) ---
        logging.info(f"[{log_prefix}] Successfully parsed signal: {parsed_signal}")
        
        # Here we would add the logic for:
        # 1. Creating the contract
        # 2. Getting the price
        # 3. Calculating quantity
        # 4. Placing the order based on the profile's exit_strategy
        # 5. Logging the trade to the CSV
        # 6. Adding to the trailing_stop_manager if needed


if __name__ == '__main__':
    # Setup logging
    import colorama
    colorama.init()
    custom_logger.setup_logging(console_log_output="stdout", console_log_level="info",
                                console_log_color=True, logfile_file=RUNTIME_LOG_FILE,
                                logfile_log_level="info", logfile_log_color=False,
                                log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s]"
                                                  " %(message)s%(color_off)s")
    logging.info(f'===================================================================\n')
    
    # Run the application
    main_app = Main()
    main_app.run()

