import logging
import requests
from services.config import Config

class TelegramInterface:
    """
    Handles sending notifications to a Telegram chat.
    This is a specialist class for communication.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}"

    async def send_message(self, message_text):
        """
        Sends a message to the configured Telegram chat ID.
        Uses a synchronous request in a thread-safe manner.
        """
        api_url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.config.telegram_chat_id,
            'text': message_text,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            self.logger.info(f"Successfully sent message to Telegram: '{message_text}'")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send message to Telegram: {e}")
            return False

    async def test_connection(self):
        """Tests the connection to the Telegram API."""
        api_url = f"{self.base_url}/getMe"
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            self.logger.info("Telegram API connection test successful.")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Telegram API connection test failed: {e}")
            return False
