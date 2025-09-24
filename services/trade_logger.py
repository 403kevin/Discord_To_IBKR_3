import csv
import os
from datetime import datetime
from typing import Dict, Any

from services import custom_logger

class TradeLogger:
    """
    Handles logging of trade signals to a CSV file.
    """

    def __init__(self, file_path: str):
        """
        Initializes the TradeLogger.

        Args:
            file_path: The path to the CSV log file.
        """
        self.file_path = file_path
        self._setup_logging()

    def _setup_logging(self):
        """
        Sets up the log file and writes the header if the file doesn't exist.
        """
        try:
            # Check if the file exists and is not empty
            file_exists = os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0
            if not file_exists:
                with open(self.file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Define the header row
                    header = [
                        'timestamp_utc', 'author', 'channel_id', 'action', 'ticker',
                        'asset_type', 'strike', 'contract_type', 'expiry_date',
                        'entry_price', 'sentiment_score', 'raw_message'
                    ]
                    writer.writerow(header)
                custom_logger.log_info(f"Trade log file created with header at {self.file_path}")
        except IOError as e:
            custom_logger.log_critical(f"Failed to set up trade log file: {e}")

    def log_signal(self, author: str, channel_id: str, raw_message: str, parsed_signal: Dict[str, Any]):
        """
        Logs a parsed trade signal to the CSV file.

        Args:
            author: The author of the signal.
            channel_id: The Discord channel ID where the signal was posted.
            raw_message: The raw text of the Discord message.
            parsed_signal: The dictionary containing the parsed signal details.
        """
        try:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                log_entry = [
                    datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    author,
                    channel_id,
                    parsed_signal.get('action'),
                    parsed_signal.get('ticker'),
                    parsed_signal.get('asset_type'),
                    parsed_signal.get('strike'),
                    parsed_signal.get('contract_type'),
                    parsed_signal.get('expiry_date'),
                    parsed_signal.get('entry_price'),
                    parsed_signal.get('sentiment_score', 'N/A'),
                    raw_message.replace('\n', ' ')  # Replace newlines for clean CSV format
                ]
                writer.writerow(log_entry)
            custom_logger.log_info(f"Successfully logged signal for {parsed_signal.get('ticker')}")
        except IOError as e:
            custom_logger.log_error(f"Failed to write to trade log file: {e}")
        except Exception as e:
            custom_logger.log_error(f"An unexpected error occurred during signal logging: {e}")
