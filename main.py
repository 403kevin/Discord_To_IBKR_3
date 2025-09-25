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


async def main():
    """
    The main asynchronous entry point and orchestrator for the trading bot.
    This function initializes all components and runs the primary event loop.
    """
    # Initialize interfaces to None to ensure they exist in the `finally` block
    # for a clean shutdown, even if initialization fails.
    ib_interface = None
    discord_interface = None
    telegram_interface = None
    eod_close_triggered_today = False
    try:
        # --- COMPONENT INITIALIZATION ---
        config = Config()
        state_manager = StateManager(config)
        ib_interface = IBInterface(config)
        discord_interface = DiscordInterface(config)
        sentiment_analyzer = SentimentAnalyzer()
        telegram_interface = TelegramInterface(config)

        # --- SURGICAL UPGRADE: The "Amnesia Vaccine" Startup Sequence ---
        # 1. Connect to the broker first. We need a live connection to understand state.
        await ib_interface.connect()

        # 2. Command the Scribe to load the last known state.
        # It needs the live ib instance to correctly "rehydrate" trade objects.
        loaded_trades, loaded_ids = state_manager.load_state()

        # 3. Brief the General ("brain") with the restored memory and the Scribe.
        signal_processor = SignalProcessor(
            config, ib_interface, discord_interface,
            sentiment_analyzer, telegram_interface, state_manager,  # Pass the Scribe
            initial_trades=loaded_trades,  # Pass the restored memory
            initial_processed_ids=loaded_ids
        )
        # --- END UPGRADE ---

        # --- 2. CONNECTION & STARTUP (Continued) ---
        await discord_interface.initialize()
        await telegram_interface.initialize()
        logger.info("VADER sentiment analyzer initialized successfully.")

        # ... (The rest of the main function, the while True: loop, follows here) ...

        # --- 3. MAIN EVENT LOOP ---
        logger.info("Starting main event loop...")
        await telegram_interface.send_message("✅ **Bot is online and running.**")
        while True:
            # --- Global Shutdown Check ---
            # This allows for a remote shutdown command via a specific Discord channel.
            if config.master_shutdown_enabled:
                try:
                    shutdown_messages = await discord_interface.get_latest_messages(
                        config.master_shutdown_channel_id, limit=1
                    )
                    if shutdown_messages and shutdown_messages[0][
                        'content'].strip().lower() == config.master_shutdown_command:
                        logger.info("Master shutdown command received. Terminating.")
                        await telegram_interface.send_message("⚠️ **Shutdown command received. Bot is terminating.**")
                        break  # Exit the main while loop
                except Exception as e:
                    logger.error(f"Error checking for shutdown command: {e}")

                    # --- SURGICAL UPGRADE: The Corrected EOD Safety Net Check ---
                    if config.eod_close["enabled"] and not eod_close_triggered_today:
                        tz = pytz.timezone("US/Eastern")
                        now_eastern = datetime.now(tz)

                        # Create a simple time object for the closing time
                        close_time = dt_time(config.eod_close["hour"], config.eod_close["minute"])

                        # The two checks are now clean and separate
                        is_after_close_time = now_eastern.time() >= close_time
                        is_weekday = now_eastern.weekday() < 5  # Monday is 0, Friday is 4

                        if is_after_close_time and is_weekday:
                            logger.warning("EOD CLOSE TRIGGERED. Initiating closing of all positions.")
                            await telegram_interface.send_message(
                                "⚠️ **EOD CLOSE TRIGGERED** ⚠️\nInitiating closing of all open positions.")

                            try:
                                await ib_interface.close_all_positions()
                                eod_close_triggered_today = True  # Mark as triggered to prevent re-running
                                await telegram_interface.send_message("✅ **EOD CLOSE COMPLETE** ✅")
                            except Exception as e:
                                logger.critical(f"EOD close procedure failed: {e}", exc_info=True)
                                await telegram_interface.send_message(f"🚨 **EOD CLOSE FAILED** 🚨\nReason: `{e}`")
                    # --- END SURGICAL UPGRADE ---

                    # --- Discord Polling Cycle ---

            # --- Discord Polling Cycle ---
            # Iterate through each channel profile defined in the config.
            for profile in config.profiles:
                if not profile.get('enabled', False):
                    continue  # Skip disabled profiles

                # --- Consecutive Loss Cooldown Check ---
                # If a channel has too many losses in a row, it's put on a timeout.
                cooldown_info = signal_processor.get_cooldown_status(profile['channel_id'])
                if cooldown_info['on_cooldown']:
                    now_utc = datetime.now(pytz.utc)
                    if now_utc < cooldown_info['end_time']:
                        continue  # Still on cooldown, skip this profile
                    else:
                        # Cooldown has expired, reset the counter.
                        signal_processor.reset_consecutive_losses(profile['channel_id'])
                        logger.info(f"Cooldown for channel {profile['channel_name']} has ended.")

                try:
                    # Fetch the latest messages from the Discord channel.
                    messages = await discord_interface.get_latest_messages(profile['channel_id'], limit=5)
                    if messages:
                        # Process messages from oldest to newest to maintain order.
                        for message in reversed(messages):
                            await signal_processor.process_signal(message, profile)
                except Exception as e:
                    logger.error(f"Error fetching or processing messages for channel {profile['channel_name']}: {e}")

                # A brief, polite pause between polling different channels.
                await asyncio.sleep(config.delay_between_channels)

            # --- Active Trade Monitoring & Management ---
            # After checking for new signals, manage all ongoing trades.
            try:
                await signal_processor.monitor_active_trades()
            except Exception as e:
                logger.error(f"Error during active trade monitoring: {e}", exc_info=True)

            # A longer pause after a full cycle of polling and monitoring.
            await asyncio.sleep(config.delay_after_full_cycle)

    except Exception as e:
        logger.critical(f"A critical error occurred in the main setup or loop: {e}", exc_info=True)
        if telegram_interface:  # <-- ADD THIS BLOCK
            await telegram_interface.send_message(f"🚨 **FATAL BOT CRASH** 🚨\n\n`{e}`")
        logger.info("Bot has shut down.")
        if telegram_interface:  # <-- ADD THIS BLOCK
            await telegram_interface.send_message(f"🚨 **FATAL BOT CRASH** 🚨\n\n`{e}`")

        if discord_interface:
            await discord_interface.close()
        if telegram_interface:  # <-- ADD THIS BLOCK
            await telegram_interface.close()


# --- SCRIPT ENTRY POINT ---
# This is the standard Python way to make a script runnable.
if __name__ == "__main__":
    # This is a small but important fix for asyncio on Windows.
    # It prevents a `RuntimeError: Event loop is closed` on shutdown.
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        # `asyncio.run()` starts the asynchronous event loop.
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # This catches manual shutdown signals (like Ctrl+C).
        logger.info("Shutdown signal received. Exiting.")


# ====================================================================================
# --- LEGACY MAIN BLOCK (FOR REFERENCE & EDUCATIONAL PURPOSES ONLY) ---
# The code below this point represents the previous, synchronous version of the bot.
# It is NOT executed. It is preserved here as a "battle scar" to show the project's
# architectural evolution from a simple, blocking script to a more robust,
# asynchronous application. This is a key part of the project's history.
# ====================================================================================

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