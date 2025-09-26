import asyncio
import logging
from aiohttp import ClientSession, ClientError
import re

class TelegramInterface:
    """
    Manages all communication with the Telegram Bot API.
    Includes robust sanitization to prevent Markdown parsing errors.
    """
    def __init__(self, config):
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = None
        self._is_initialized = False

    async def initialize(self):
        """Initializes the aiohttp session for sending messages."""
        if not self.session or self.session.closed:
            self.session = ClientSession()
            self._is_initialized = True
            logging.info("Telegram session initialized.")

    def is_initialized(self):
        """Checks if the Telegram session is active."""
        return self._is_initialized and self.session and not self.session.closed

    async def shutdown(self):
        """Closes the aiohttp session gracefully."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._is_initialized = False
        logging.info("Telegram session closed.")

    def _sanitize_markdown(self, text: str) -> str:
        """
        Escapes all special characters for Telegram MarkdownV2.
        This is a comprehensive and robust implementation.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Characters to escape for MarkdownV2.
        # The backslash must be escaped first.
        escape_chars = r'\_*[]()~`>#+-=|{}.!'
        
        # This regex will find any of the characters in the string and prepend a backslash.
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    async def send_message(self, text):
        """Sends a text message to the configured Telegram chat, using MarkdownV2."""
        if not self.is_initialized():
            logging.error("Telegram session not initialized. Cannot send message.")
            return

        url = f"{self.api_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'MarkdownV2'
        }
        try:
            async with self.session.post(url, json=payload, timeout=10) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logging.error(f"Failed to send Telegram message. Status: {response.status}, Response: {response_text}")
        except Exception as e:
            logging.error(f"An error occurred while sending Telegram message: {e}", exc_info=True)

    async def send_trade_notification(self, trade_info, status):
        """Sends a professionally formatted and sanitized trade notification."""
        header_map = {
            "OPENED": "✅ TRADE OPENED ✅",
            "CLOSED": "❌ TRADE CLOSED ❌",
            "VETOED": "🚫 TRADE VETOED 🚫"
        }
        # Sanitize the header just in case status is an unexpected value
        header = self._sanitize_markdown(header_map.get(status, status.upper()))

        # Sanitize all dynamic parts of the message
        ticker = self._sanitize_markdown(trade_info.get('ticker'))
        option = self._sanitize_markdown(trade_info.get('option'))
        expiry = self._sanitize_markdown(trade_info.get('expiry'))
        source = self._sanitize_markdown(trade_info.get('source'))

        # Use f-strings for a clean, multi-line message
        message = (
            f"*{header}*\n\n"
            f"*Ticker:* `{ticker}`\n"
            f"*Option:* `{option}`\n"
            f"*Expiry:* `{expiry}`\n"
            f"*Source:* `{source}`\n"
        )
        
        if status == "VETOED":
            reason = self._sanitize_markdown(trade_info.get('reason'))
            message += f"*Reason:* `{reason}`\n"
        if status == "CLOSED":
            pnl = self._sanitize_markdown(trade_info.get('pnl', 'N/A'))
            exit_reason = self._sanitize_markdown(trade_info.get('exit_reason', 'N/A'))
            message += f"*P/L:* `{pnl}`\n"
            message += f"*Exit Reason:* `{exit_reason}`\n"

        await self.send_message(message.strip())

