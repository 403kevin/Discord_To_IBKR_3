# services/custom_logger.py
import logging
import sys

def setup_logger():
    """
    Sets up a basic, universal logger that works on all operating systems.
    This version removes the non-Windows compatible code.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to print to the console (standard error)
    # This is compatible with all systems.
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)
    
    # Also add a file handler to log to runtime.log
    file_handler = logging.FileHandler('runtime.log', mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Custom logger initialized.")
