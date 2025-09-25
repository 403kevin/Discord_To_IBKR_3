import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os

# --- Core Components ---
from services.config import Config
from services.state_manager import StateManager
from services.sentiment_analyzer import SentimentAnalyzer
from bot_engine.signal_processor import SignalProcessor

# --- Interfaces ---
from bot_engine.interfaces.discord_interface import DiscordInterface
from bot_engine.interfaces.ib_interface import IBInterface
from bot_engine.interfaces.telegram_interface import TelegramInterface


def setup_logging():
    # ... (logging setup is unchanged)
    pass

# = a bunch of unchanged code

async def main():
    """
    The main asynchronous function that initializes and runs the bot.
    This is the "Orchestrator" of the entire application.
    """
    # Initialize all components to None for robust cleanup in the finally block
    config = None
    state_manager = None
    sentiment_analyzer = None
    ib_interface = None
    telegram_interface = None
    discord_interface = None
    signal_processor = None
    
    # This will hold our main running tasks
    main_tasks = []

    try:
        logging.info("--- 🚀 LAUNCHING TRADING BOT 🚀 ---")
        
        # --- 1. Assemble the Machine ---
        config = Config()
        state_manager = StateManager(config)
        sentiment_analyzer = SentimentAnalyzer()
        
        ib_interface = IBInterface(config)
        telegram_interface = TelegramInterface(config)
        discord_interface = DiscordInterface(config)
        
        # --- 2. Pre-flight Connections & State Load ---
        await ib_interface.connect()
        await telegram_interface.initialize()
        await discord_interface.initialize_and_login()
        loaded_trades, loaded_ids = state_manager.load_state()

        # --- 3. Instantiate the Brain ---
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

        # --- 4. GO LIVE: Create the main task for the SignalProcessor ---
        # THE FINAL FIX: We now create the processor's main loop as a background
        # task instead of directly awaiting it. This allows the main() function
        # to continue running and keep the program alive.
        logging.info("Starting main event loop...")
        processor_task = asyncio.create_task(signal_processor.start())
        main_tasks.append(processor_task)

        # --- 5. Keep the main function alive ---
        # The main() function will now run forever, monitoring the primary task.
        # This is the "heartbeat" that prevents the script from exiting prematurely.
        await asyncio.gather(*main_tasks)

    except asyncio.CancelledError:
        logging.info("Main task was cancelled, initiating shutdown.")
    except Exception as e:
        logging.critical("A critical error occurred in the main setup or loop: %s", e, exc_info=True)
        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message(f"🚨 CRITICAL ERROR 🚨\nBot has crashed. Check logs.\n\nError: {e}")
    finally:
        logging.info("--- 😴 Bot is shutting down. ---")
        
        # Gracefully cancel all running tasks
        for task in main_tasks:
            task.cancel()
        if signal_processor:
            await signal_processor.shutdown() # Ensure internal shutdown event is set

        # Gracefully close all connections
        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message("😴 Bot is shutting down.")
            await telegram_interface.shutdown()
        if ib_interface and ib_interface.is_connected():
            await ib_interface.disconnect()
        if discord_interface and discord_interface.is_initialized():
            await discord_interface.shutdown()

if __name__ == "__main__":
    setup_logging()
    # To handle Ctrl+C gracefully
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutdown initiated by user (Ctrl+C).")

