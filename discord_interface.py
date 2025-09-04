# interfaces/discord_interface.py
import logging
import requests
import time
from threading import Thread

class DiscordInterface:
    """
    The bot's "Ears." This class is responsible for polling target Discord
    channels for new messages using a custom, lightweight polling engine
    based on direct HTTP requests. This version is guaranteed to be
    cross-platform and compatible with Windows.
    """
    def __init__(self, config, callback):
        self.config = config
        self.callback = callback
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.config.discord_user_token,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self._polling_thread = None
        self._is_polling = False
        self.last_message_ids = {str(p["channel_id"]): None for p in self.config.profiles}

    def _poll_channel(self, profile):
        """Fetches the latest messages from a single channel."""
        channel_id = str(profile["channel_id"])
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=10"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                messages = response.json()
                if not messages:
                    return

                # Process messages from oldest to newest
                new_messages = []
                last_id = self.last_message_ids.get(channel_id)
                for msg in reversed(messages):
                    if not last_id or msg['id'] > last_id:
                        new_messages.append(msg)

                if new_messages:
                    self.last_message_ids[channel_id] = new_messages[-1]['id']
                    for msg in new_messages:
                        message_data = {
                            "channel_id": msg["channel"]["id"],
                            "content": msg["content"]
                        }
                        if self.callback:
                            self.callback(message_data)
            else:
                logging.error(f"Error polling channel {channel_id}: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            logging.error(f"Network error while polling channel {channel_id}: {e}")

    def _polling_loop(self):
        """The main loop that cycles through all target channels."""
        logging.info("Discord polling thread started.")
        while self._is_polling:
            for profile in self.config.profiles:
                if profile["enabled"]:
                    self._poll_channel(profile)
                    time.sleep(self.config.delay_between_channels)
            
            time.sleep(self.config.delay_after_full_cycle)
        logging.info("Discord polling thread stopped.")

    def check_connection(self):
        """A simple check to validate the user token."""
        url = "https://discord.com/api/v9/users/@me"
        try:
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def start_polling(self):
        if not self._is_polling:
            self._is_polling = True
            self._polling_thread = Thread(target=self._polling_loop, name="DiscordPollingThread", daemon=True)
            self._polling_thread.start()

    def stop_polling(self):
        logging.info("Stopping Discord polling.")
        self._is_polling = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=10)

