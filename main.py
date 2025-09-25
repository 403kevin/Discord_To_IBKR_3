# --- SURGICAL FIX: The "GPS" Protocol ---
# This block makes the script self-aware of its location and fixes all
# potential import errors, regardless of where it's run from.
import sys
import os
# Get the absolute path of the directory containing main.py
project_root = os.path.dirname(os.path.abspath(__file__))
# Add this path to the list of places Python looks for modules
sys.path.insert(0, project_root)
# --- END SURGICAL FIX ---
import logging
import sys
import asyncio
from datetime import datetime, time as dt_time
import pytz
from collections import deque

# --- MODULE IMPORTS ---
# These are the specialist modules that perform specific jobs.
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from bot_engine.signal_processor import SignalProcessor
from interfaces.telegram_interface import TelegramInterface
from services.trade_logger import TradeLogger
from services.state_manager import StateManager

# --- 1. SETUP ---
# Custom logger configuration. All events will be recorded in `runtime.log`.
log_file_path = "runtime.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info("Custom logger initialized.")

# --- DATA STRUCTURES ---
# A simple in-memory dictionary to hold the state of active trades.
# The key is a unique trade identifier, and the value is an object with trade details.
active_trades = {}
# A deque (a list with a maximum size) to store recent message IDs.
# This acts as a short-term memory to prevent processing the same signal twice.
processed_message_ids = deque(maxlen=Config().processed_message_cache_size)


# --- UTILITY FUNCTIONS ---
def is_market_hours(timezone="US/Eastern"):
    """
    Checks if the current time is within regular US market hours (9:30 AM to 4:00 PM Eastern).
    Also checks that it's a weekday.
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    # Monday is 0, Sunday is 6. We only trade on weekdays.
    return market_open <= now.time() <= market_close and now.weekday() < 5


# THIS IS THE CHANGE FOR main.py

async def main():
    """
    The main asynchronous function that initializes and runs the bot.
    This is the "Orchestrator" of the entire application.
    """
    telegram_interface = None
    ib_interface = None
    discord_interface = None
    try:
        logging.info("--- 🚀 LAUNCHING TRADING BOT 🚀 ---")

        # --- Initialize Components ---
        config = Config()
        state_manager = StateManager(config)
        sentiment_analyzer = SentimentAnalyzer()

        ib_interface = IBInterface(config)
        telegram_interface = TelegramInterface(config)
        discord_interface = DiscordInterface(config)

        # --- Pre-flight Connections ---
        await ib_interface.connect()
        await telegram_interface.initialize()
        await discord_interface.initialize_and_login()

        # --- Load State ---
        loaded_trades, loaded_ids = state_manager.load_state()

        # --- Instantiate the Brain ---
        signal_processor = SignalProcessor(
            config=config,
            ib_interface=ib_interface,
            telegram_interface=telegram_interface,
            discord_interface=discord_interface,
            state_manager=state_manager,
            sentiment_analyzer=sentiment_analyzer,
            initial_positions=loaded_trades,
            initial_processed_ids=loaded_ids
        )

        # --- GO LIVE ---
        # The start() method now contains all the logic and loops.
        # It will run indefinitely until a shutdown is triggered.
        logging.info("Starting main event loop...")
        await signal_processor.start()

    except Exception as e:
        logging.critical("A critical error occurred in the main setup or loop: %s", e, exc_info=True)
        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message(f"🚨 CRITICAL ERROR 🚨\nBot has crashed. Check logs.\n\nError: {e}")
    finally:
        logging.info("--- 😴 Bot is shutting down. ---")
        # Gracefully close all connections
        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message("😴 Bot is shutting down.")
            await telegram_interface.shutdown()
        if ib_interface and ib_interface.is_connected():
            await ib_interface.disconnect()
        if discord_interface and discord_interface.is_initialized():
            await discord_interface.shutdown()
def legacy_main_disabled():
    """
    This function represents the old, single-threaded architecture.
    It is not called and will not run. It is for historical reference.
    """
    config = Config()
    ib_interface = IBInterface(config)
    discord_interface = DiscordInterface(config)
    sentiment_analyzer = SentimentAnalyzer(config)

    try:
        # --- 2. INITIALIZATION & CONNECTION (Legacy Sync) ---
        # In the old model, this `connect` call would block the entire script.
        ib_interface.connect_sync()
        logger.info("VADER sentiment analyzer initialized successfully.")

        # --- 3. MAIN EVENT LOOP (Legacy Sync) ---
        logger.info("Starting main event loop...")
        while True:
            # --- Global Shutdown Check (Legacy) ---
            if config.master_shutdown_enabled:
                try:
                    # This would require a synchronous version of the discord fetcher.
                    # It highlights the difficulty of mixing async-style tasks
                    # in a synchronous loop.
                    pass
                except Exception as e:
                    logger.error(f"Error checking for shutdown command: {e}")

            # --- Discord Polling (Legacy Sync) ---
            for profile in config.profiles:
                if not profile.get('enabled', False):
                    continue

                try:
                    # The old script would have to block here, waiting for a response.
                    messages = []  # Placeholder for a synchronous fetch
                    if messages:
                        for message in reversed(messages):
                            # The entire trade logic would block here. If a trade
                            # took 3 seconds, the bot could not poll other channels.
                            pass
                except Exception as e:
                    logger.error(f"Error fetching messages for {profile['channel_name']}: {e}")

                # A hard, blocking sleep.
                time.sleep(config.delay_between_channels)

            # --- Active Trade Monitoring (The biggest flaw of the legacy model) ---
            # In a sync model, monitoring trades while also polling for new ones
            # is extremely difficult and inefficient, often requiring complex
            # threading that we proved was a failure point. The new async
            # `monitor_active_trades` is architecturally superior.

            time.sleep(config.delay_after_full_cycle)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if ib_interface.ib.isConnected():
            ib_interface.disconnect_sync()
        logger.info("Bot has shut down.")