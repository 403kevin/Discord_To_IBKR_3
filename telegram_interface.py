import logging
import aiohttp

logger = logging.getLogger(__name__)

class TelegramInterface:
    """
    A specialist module for sending notifications to Telegram.
    Uses the aiohttp library for asynchronous, non-blocking HTTP requests.
    """

    def __init__(self, config):
        self.config = config
        self.token = self.config.telegram_bot_token
        self.chat_id = self.config.telegram_chat_id
        self.session = None
        if self.token and self.chat_id:
            self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        else:
            self.base_url = None
            logger.warning("Telegram token or chat ID is missing. Telegram notifications will be disabled.")

    async def initialize(self):
        """Initializes the aiohttp session."""
        if self.base_url:
            self.session = aiohttp.ClientSession()
            logger.info("Telegram interface initialized.")

    async def send_message(self, text: str):
        """
        Sends a message to the configured Telegram chat.
        """
        if not self.session or self.session.closed:
            logger.error("Telegram session not initialized or has been closed. Cannot send message.")
            return

        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        }
        try:
            async with self.session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Failed to send Telegram message. Status: {response.status}, Response: {response_text}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending Telegram message: {e}", exc_info=True)

    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Telegram session closed.")

