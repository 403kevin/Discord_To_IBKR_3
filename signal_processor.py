# interfaces/telegram_notifier.py
import logging
import requests


class TelegramNotifier:
    """
    Handles all communication with the Telegram Bot API.
    Responsible for sending formatted messages and alerts.
    """

    def __init__(self, config):
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text):
        """Sends a simple, general-purpose message to the Telegram chat."""
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            logging.info("Successfully sent general message to Telegram.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send message to Telegram: {e}")

    def send_fill_confirmation(self, fill, pos_data):
        """
        NEW: Sends a detailed, formatted message upon a confirmed trade fill.
        This is the new standard for entry notifications.
        """
        contract = fill.contract
        execution = fill.execution
        profile = pos_data["profile"]
        sentiment_score = pos_data["sentiment_score"]

        # Format the message with rich details
        message = (
            f"✅ *Trade Entry Confirmed* ✅\n\n"
            f"*Symbol:* `{contract.localSymbol}`\n"
            f"*Quantity:* `{int(execution.shares)}`\n"
            f"*Fill Price:* `${execution.price:.2f}`\n\n"
            f"*Source Channel:* `{profile['channel_name']}`\n"
            f"*Sentiment Score:* `{sentiment_score:.4f}`"
        )

        # Use the general send_message method to dispatch it
        self.send_message(message)

