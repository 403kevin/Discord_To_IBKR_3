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
        SURGICAL FIX: Escapes special characters in a string to prevent
        Telegram MarkdownV2 parsing errors.
        """
        if not isinstance(text, str):
            text = str(text)
        # Characters to escape for MarkdownV2
        escape_chars = r'\_*[]()~`>#+-=|{}.!'
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
        header = self._sanitize_markdown(header_map.get(status, status.upper()))

        # Sanitize all dynamic parts of the message
        ticker = self._sanitize_markdown(trade_info.get('ticker'))
        option = self._sanitize_markdown(trade_info.get('option'))
        expiry = self._sanitize_markdown(trade_info.get('expiry'))
        source = self._sanitize_markdown(trade_info.get('source'))

        # Use f-strings and triple quotes for a clean, multi-line message
        message = f"""
*{header}*

*Ticker:* `{ticker}`
*Option:* `{option}`
*Expiry:* `{expiry}`
*Source:* `{source}`
"""
        
        if status == "VETOED":
            reason = self._sanitize_markdown(trade_info.get('reason'))
            message += f"*Reason:* `{reason}`\n"
        if status == "CLOSED":
            pnl = self._sanitize_markdown(trade_info.get('pnl', 'N/A'))
            exit_reason = self._sanitize_markdown(trade_info.get('exit_reason', 'N/A'))
            message += f"*P/L:* `{pnl}`\n"
            message += f"*Exit Reason:* `{exit_reason}`\n"

        # Telegram API requires the message to be clean, without leading/trailing whitespace
        await self.send_message(message.strip())

