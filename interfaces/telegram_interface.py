import asyncio
import logging
from aiohttp import ClientSession, ClientError

class TelegramInterface:
    """
    Manages all communication with the Telegram Bot API.
    Handles sending formatted messages and notifications.
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

    async def send_message(self, text):
        """Sends a plain text message to the configured Telegram chat."""
        if not self.is_initialized():
            logging.error("Telegram session not initialized or has been closed. Cannot send message.")
            return

        url = f"{self.api_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown' # Enable Markdown for better formatting
        }
        try:
            async with self.session.post(url, json=payload, timeout=10) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logging.error(f"Failed to send Telegram message. Status: {response.status}, Response: {response_text}")
        except ClientError as e:
            logging.error(f"An error occurred while sending Telegram message: {e}")
        except asyncio.TimeoutError:
            logging.error("Request to Telegram API timed out.")

    async def send_trade_notification(self, trade_info, status):
        """
        Sends a professionally formatted trade notification.
        This is the new, structured method called by the SignalProcessor.
        """
        header = ""
        if status == "OPENED":
            header = "✅ TRADE OPENED ✅"
        elif status == "CLOSED":
            header = "❌ TRADE CLOSED ❌"
        elif status == "VETOED":
             header = "🚫 TRADE VETOED 🚫"

        # Using Markdown for bolding and fixed-width font
        message = (
            f"*{header}*\n\n"
            f"*Ticker:* `{trade_info.get('ticker')}`\n"
            f"*Option:* `{trade_info.get('option')}`\n"
            f"*Expiry:* `{trade_info.get('expiry')}`\n"
            f"*Source:* `{trade_info.get('source')}`\n"
        )
        
        # Add extra details for specific statuses
        if status == "VETOED":
            message += f"*Reason:* `{trade_info.get('reason')}`\n"
        if status == "CLOSED":
            message += f"*P/L:* `{trade_info.get('pnl', 'N/A')}`\n"
            message += f"*Exit Reason:* `{trade_info.get('exit_reason', 'N/A')}`\n"


        await self.send_message(message)