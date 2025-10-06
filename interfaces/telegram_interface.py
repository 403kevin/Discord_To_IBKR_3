import asyncio
import logging
from aiohttp import ClientSession, ClientError
import re

class TelegramInterface:
    """
    Manages all communication with the Telegram Bot API.
    This version includes the professional, multi-format notification system.
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
        """Escapes all special characters for Telegram MarkdownV2."""
        if not isinstance(text, str):
            text = str(text)
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
        """
        THE VETERAN UPGRADE: Sends a professionally formatted notification
        based on the new, multi-option format.
        FIX: Corrected sentiment_score formatting to handle both numeric and 'N/A' values.
        """
        message = ""
        # Sanitize all data first for safety
        source_channel = self._sanitize_markdown(trade_info.get('source_channel', 'N/A'))
        contract_details = self._sanitize_markdown(trade_info.get('contract_details', 'N/A'))

        if status == "OPENED":
            quantity = self._sanitize_markdown(str(trade_info.get('quantity', 'N/A')))
            entry_price = self._sanitize_markdown(f"${trade_info.get('entry_price'):.2f}" if trade_info.get('entry_price') is not None else 'N/A')
            
            # FIX: Handle sentiment_score properly - check if it's numeric before formatting
            sentiment_raw = trade_info.get('sentiment_score')
            if sentiment_raw is None or sentiment_raw == 'N/A':
                sentiment_score = self._sanitize_markdown('N/A')
            elif isinstance(sentiment_raw, (int, float)):
                sentiment_score = self._sanitize_markdown(f"{sentiment_raw:.4f}")
            else:
                sentiment_score = self._sanitize_markdown(str(sentiment_raw))
            
            trail_method = self._sanitize_markdown(trade_info.get('trail_method', 'N/A'))
            momentum_exit = self._sanitize_markdown(trade_info.get('momentum_exit', 'None'))
            
            message = f"""
*✅ Trade Entry Confirmed ✅*

*Source Channel:* `{source_channel}`
*Contract:* `{contract_details}`
*Quantity:* `{quantity}`
*Entry Price:* `{entry_price}`
*Vader Sentiment:* `{sentiment_score}`
*Trail Method:* `{trail_method}`
*Momentum Exit:* `{momentum_exit}`
"""

        elif status == "CLOSED":
            exit_price = self._sanitize_markdown(f"${trade_info.get('exit_price'):.2f}" if trade_info.get('exit_price') is not None else 'N/A')
            reason = self._sanitize_markdown(trade_info.get('reason', 'N/A'))
            
            message = f"""
*🔴 SELL Order Executed*

*Contract:* `{contract_details}`
*Exit Price:* `{exit_price}`
*Reason:* `{reason}`
"""

        elif status == "VETOED":
            reason = self._sanitize_markdown(trade_info.get('reason', 'N/A'))

            message = f"""
*❌ Trade Vetoed ❌*

*Source Channel:* `{source_channel}`
*Contract:* `{contract_details}`
*Reason:* `{reason}`
"""

        if message:
            await self.send_message(message.strip())
