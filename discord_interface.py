# interfaces/discord_interface.py
import logging
import requests
import time
from threading import Thread, Event # Event is our new "emergency cord"
from datetime import datetime, timezone
from collections import deque # The perfect tool for a short-term memory queue

class DiscordInterface:
    """
    The bot's "Ears." This definitive version includes the master shutdown
    listener and a short-term memory cache to prevent duplicate signals,
    based on the master GitHub file.
    """
    def __init__(self, config, message_queue, shutdown_event: Event):
        self.config = config
        self.queue = message_queue
        self.shutdown_event = shutdown_event # The shared emergency cord
        self.processed_ids = deque(maxlen=config.processed_message_cache_size)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": config.discord_user_token,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self._polling_thread = None
        self._is_polling = False
        self.last_message_ids = {str(p["channel_id"]): "INIT" for p in config.profiles}

    def _poll_channel(self, profile, is_shutdown_channel=False):
        """Fetches and filters messages from a single channel."""
        # Determine which channel to poll
        if is_shutdown_channel:
            if not self.config.master_shutdown_channel_id or self.config.master_shutdown_channel_id == "YOUR_PRIVATE_DISCORD_CHANNEL_ID":
                return # Don't poll if the shutdown channel isn't configured
            channel_id = self.config.master_shutdown_channel_id
        else:
            channel_id = str(profile["channel_id"])

        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=5"
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                logging.warning(f"Could not poll channel {channel_id}. Status: {response.status_code}")
                return

            messages = response.json()
            if not messages: return

            for msg in reversed(messages):
                # --- NEW: Master Shutdown Logic ---
                if is_shutdown_channel:
                    if self.config.master_shutdown_command.lower() in msg['content'].lower():
                        if msg['id'] not in self.processed_ids:
                            logging.critical("MASTER SHUTDOWN COMMAND DETECTED. Signaling termination.")
                            self.shutdown_event.set() # Pull the emergency cord
                            self.processed_ids.append(msg['id'])
                            return # Stop all polling

                # Regular signal processing
                else:
                    last_id = self.last_message_ids.get(channel_id)
                    if last_id == "INIT":
                        self.last_message_ids[channel_id] = messages[0]['id']
                        logging.info(f"Initial poll for channel {channel_id}. Set last message ID to {messages[0]['id']}. Ignoring historical messages.")
                        break # Set initial ID and move on to the next channel

                    if msg['id'] > last_id:
                        if msg['id'] in self.processed_ids: continue
                        
                        msg_time = datetime.fromisoformat(msg['timestamp'].replace("Z", "+00:00"))
                        age_seconds = (datetime.now(timezone.utc) - msg_time).total_seconds()
                        
                        if age_seconds <= self.config.signal_max_age_seconds:
                            self.queue.put({"channel_id": msg["channel_id"], "content": msg["content"]})
                            self.processed_ids.append(msg['id']) # Add to memory
            
            if messages and not is_shutdown_channel:
                # Always update to the newest message ID seen in the batch
                self.last_message_ids[channel_id] = messages[0]['id']

        except requests.RequestException as e:
            logging.error(f"Network error while polling Discord: {e}")

    def _polling_loop(self):
        logging.info("Discord polling thread started.")
        while self._is_polling and not self.shutdown_event.is_set():
            # Poll for signals
            for profile in self.config.profiles:
                if profile["enabled"]:
                    self._poll_channel(profile)
                    time.sleep(self.config.delay_between_channels)
            
            # Poll for shutdown command
            if self.config.master_shutdown_enabled:
                self._poll_channel(None, is_shutdown_channel=True)

            time.sleep(self.config.delay_after_full_cycle)
        logging.info("Discord polling thread stopped.")

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

