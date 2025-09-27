import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

# --- GPS for our fortress ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Core Components ---
from services.config import Config
from services.state_manager import StateManager
from services.sentiment_analyzer import SentimentAnalyzer
from bot_engine.signal_processor import SignalProcessor

# --- Interfaces ---
from interfaces.discord_interface import DiscordInterface
from interfaces.ib_interface import IBInterface
from interfaces.telegram_interface import TelegramInterface


def setup_logging():
    """Sets up a robust, rotating logger for the application."""
    log_formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    
    log_file = os.getenv("TRADE_BOT_LOG_FILE", "logs/trading_bot.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # --- THE "HEAD CHEF" PROTOCOL ---
    # Seize control of the root logger to prevent duplicate messages.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any handlers that may have been pre-configured by imported libraries.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Now, add our own clean handlers.
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("Custom logger initialized.")


async def main():
    """
    The main asynchronous function that initializes and runs the bot.
    """
    telegram_interface = None
    ib_interface = None
    discord_interface = None
    signal_processor = None
    main_tasks = []

    try:
        logging.info("--- 🚀 LAUNCHING TRADING BOT 🚀 ---")
        
        config = Config()
        state_manager = StateManager(config)
        sentiment_analyzer = SentimentAnalyzer()
        
        ib_interface = IBInterface(config)
        telegram_interface = TelegramInterface(config)
        discord_interface = DiscordInterface(config)
        
        await ib_interface.connect()
        await telegram_interface.initialize()
        await discord_interface.initialize()
        
        loaded_trades, loaded_ids = state_manager.load_state()

        await telegram_interface.send_message("*🚀 Bot is starting up\\.\\.\\.*")

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

        logging.info("Starting main event loop...")
        processor_task = asyncio.create_task(signal_processor.start())
        main_tasks.append(processor_task)

        await asyncio.gather(*main_tasks)

    except asyncio.CancelledError:
        logging.info("Main task was cancelled, initiating shutdown.")
    except Exception as e:
        logging.critical("A critical error occurred in the main setup or loop: %s", e, exc_info=True)
        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message(f"🚨 CRITICAL ERROR 🚨\nBot has crashed\\. Check logs\\.")
    finally:
        logging.info("--- 😴 Bot is shutting down. ---")
        
        for task in main_tasks:
            task.cancel()
        if signal_processor:
            await signal_processor.shutdown()

        if telegram_interface and telegram_interface.is_initialized():
            await telegram_interface.send_message("😴 Bot is shutting down\\.")
            await telegram_interface.shutdown()
        if ib_interface and ib_interface.is_connected():
            await ib_interface.disconnect()
        if discord_interface and discord_interface.is_initialized():
            await discord_interface.shutdown()

if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutdown initiated by user (Ctrl+C).")

