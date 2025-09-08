# interfaces/telegram_notifier.py
import logging
import requests

class TelegramNotifier:
    """
    The bot's "Mouth." This is the definitive, corrected version that can
    handle the sentiment score in fill confirmations.
    """
    def __init__(self, config):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"

    def send_message(self, message_text):
        """Sends a general-purpose message to the configured Telegram chat."""
        params = {
            'chat_id': self.config.telegram_chat_id,
            'text': message_text,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                logging.info("Successfully sent general message to Telegram.")
                return True
            else:
                logging.error(f"Failed to send Telegram message. Status: {response.status_code}, Response: {response.text}")
                return False
        except requests.RequestException as e:
            logging.error(f"Network error sending Telegram message: {e}")
            return False

    def send_fill_confirmation(self, fill, sentiment_score, channel_name):
        """
        Sends a detailed, formatted message upon a confirmed trade entry.
        THIS IS THE CORRECTED FUNCTION SIGNATURE.
        """
        contract = fill.contract
        execution = fill.execution
        
        message = (
            f"✅ *Trade Entry Confirmed* ✅\n\n"
            f"*Symbol:* `{contract.localSymbol}`\n"
            f"*Quantity:* `{int(execution.shares)}`\n"
            f"*Entry Price:* `${execution.price:.2f}`\n"
            f"*Source Channel:* `{channel_name}`\n"
            f"*Sentiment Score:* `{sentiment_score:.4f}`"
        )
        return self.send_message(message)

