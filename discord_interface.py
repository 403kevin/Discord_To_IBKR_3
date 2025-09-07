# interfaces/discord_interface.py
import logging
import requests
import time
from threading import Thread
from datetime import datetime, timezone # NEW: The Mailman's watch

class DiscordInterface:
    """
    The bot's "Ears." This final, intelligent version checks the timestamp
    of every message to ensure the bot only acts on fresh, real-time signals.
    """
    def __init__(self, config, message_queue):
        self.config = config
        self.queue = message_queue
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.config.discord_user_token,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self._polling_thread = None
        self._is_polling = False
        # Initialize last_message_ids to "INIT" to handle the first poll specially.
        self.last_message_ids = {str(p["channel_id"]): "INIT" for p in self.config.profiles}

    def _poll_channel(self, profile):
        """Fetches and filters the latest messages from a single channel."""
        channel_id = str(profile["channel_id"])
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=20"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                messages = response.json()
                if not messages: return

                new_messages = []
                last_id = self.last_message_ids.get(channel_id)

                # On the very first poll, we just set the last seen message ID
                # to the newest one, and then we're done. This prevents processing
                # a backlog of old messages on startup.
                if last_id == "INIT":
                    self.last_message_ids[channel_id] = messages[0]['id']
                    logging.info(f"Initial poll for channel {channel_id}. Set last message ID to {messages[0]['id']}. Ignoring historical messages.")
                    return

                for msg in reversed(messages):
                    if msg['id'] > last_id:
                        # --- NEW: Check the "Postmark" (Timestamp) ---
                        msg_time_str = msg['timestamp']
                        # The timestamp format from Discord can have different precision, so we handle it.
                        msg_time = datetime.fromisoformat(msg_time_str.replace("Z", "+00:00"))
                        
                        age_seconds = (datetime.now(timezone.utc) - msg_time).total_seconds()

                        # Only process messages that are fresh.
                        if age_seconds <= self.config.signal_max_age_seconds:
                            new_messages.append(msg)
                        else:
                            logging.debug(f"Ignoring stale message (age: {age_seconds:.0f}s) from {channel_id}.")

                if new_messages:
                    self.last_message_ids[channel_id] = new_messages[-1]['id']
                    for msg in new_messages:
                        message_data = {"channel_id": msg["channel_id"], "content": msg["content"]}
                        self.queue.put(message_data)
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
        """Starts the polling thread."""
        if not self._is_polling:
            self._is_polling = True
            self._polling_thread = Thread(target=self._polling_loop, name="DiscordPollingThread", daemon=True)
            self._polling_thread.start()

    def stop_polling(self):
        """Stops the polling thread."""
        logging.info("Stopping Discord polling.")
        self._is_polling = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=10)

