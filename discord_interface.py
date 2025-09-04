# interfaces/discord_interface.py
import requests
import logging
import time


class DiscordInterface:
    """
    A custom polling engine to scrape messages from specific Discord channels.
    It operates by making direct HTTP GET requests to Discord's API.
    """

    def __init__(self, config):
        self.token = config.discord_user_token
        self.headers = {"Authorization": self.token}
        self.last_message_ids = {}  # Stores the last seen message ID for each channel

    def poll_new_messages(self, channel_id: str) -> list:
        """
        Polls a single channel for new messages since the last poll.
        """
        if not self.token:
            logging.error("Discord token not set. Polling is disabled.")
            return []

        # Construct the API URL
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=10"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            messages = response.json()

            if not messages:
                return []

            last_id = self.last_message_ids.get(channel_id)
            newest_message_id = messages[0]['id']  # The first message is the newest

            # If we've never seen this channel, we only process the very last message
            if not last_id:
                self.last_message_ids[channel_id] = newest_message_id
                # Return only the content of the single most recent message
                return [messages[0]]

            new_messages = []
            # Iterate from oldest to newest in the fetched batch
            for msg in reversed(messages):
                if msg['id'] > last_id:
                    new_messages.append(msg)

            # Update the last seen message ID for the next poll
            self.last_message_ids[channel_id] = newest_message_id

            if new_messages:
                logging.info(f"Found {len(new_messages)} new message(s) in channel {channel_id}.")

            return new_messages

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch messages from Discord channel {channel_id}: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during Discord polling: {e}")
            return []

