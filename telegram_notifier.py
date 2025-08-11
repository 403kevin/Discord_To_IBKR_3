import os
import logging
import requests
import config

class TelegramNotifier:
    """
    Handles sending messages to a specified Telegram chat.
    """
    def __init__(self):
        """
        Initializes the notifier, loading settings and secrets.
        """
        self.settings = config.TELEGRAM_SETTINGS
        if not self.settings.get("enabled"):
            self.bot_token = None
            self.chat_id = None
            logging.warning("[Telegram] Notifier is disabled in config.")
            return

        token_name = self.settings.get("bot_token_name")
        chat_id_name = self.settings.get("chat_id_name")

        self.bot_token = os.getenv(token_name)
        self.chat_id = os.getenv(chat_id_name)

        if not self.bot_token or not self.chat_id:
            logging.error(f"[Telegram] Bot token ('{token_name}') or chat ID ('{chat_id_name}') not found in .env file.")
            self.bot_token = None # Disable if secrets are missing

    def send_message(self, text: str):
        """
        Sends a text message to the configured Telegram chat.
        """
        if not self.bot_token:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"[Telegram] Message sent successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"[Telegram] Failed to send message: {e}")

