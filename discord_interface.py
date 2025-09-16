import logging
import requests
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DiscordInterface:
    """
    A specialist module responsible for all interactions with the Discord API.
    This is a true "custom polling engine" that uses the `requests` library
    to "fly under the radar," as commanded by the README.md.

    This version is architecturally superior for our specific task as it gives
    us full control over the "disguise" and avoids the complexities of the
    full discord.py client library.
    """

    def __init__(self, config):
        """
        Initializes the requests session and the disguise.
        Args:
            config: The main configuration object.
        """
        self.config = config
        self.token = self.config.discord_user_token

        # This session object will hold our disguise and handle connections.
        self.session = requests.Session()

        # --- SURGICAL FIX: The "Chrome Browser" Disguise ---
        # This is the definitive implementation of the "discreet" protocol.
        self.session.headers.update({
            'Authorization': self.token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Content-Type': 'application/json'
        })
        # --- END SURGICAL FIX ---

        self.is_initialized = False

    async def initialize(self):
        """
        Verifies the token and disguise by making a test call to the Discord API.
        """
        if self.is_initialized:
            logger.warning("Discord interface is already initialized.")
            return

        try:
            # We will run the synchronous request in an async executor to avoid blocking.
            loop = asyncio.get_running_loop()

            # The test endpoint to verify authentication.
            url = "https://discord.com/api/v9/users/@me"

            # Use run_in_executor to call the blocking 'requests' method
            response = await loop.run_in_executor(None, lambda: self.session.get(url, timeout=10))

            # Check if the disguise worked.
            if response.status_code == 200:
                user_data = response.json()
                self.is_initialized = True
                logger.info(
                    f"Discord interface initialized successfully. Logged in as {user_data['username']}#{user_data['discriminator']}.")
            else:
                # If we get a 401, the token is genuinely bad.
                logger.critical(f"Discord login failed. Status code: {response.status_code}. Response: {response.text}")
                raise ConnectionRefusedError("Discord rejected the authorization token. Please generate a new one.")

        except requests.exceptions.RequestException as e:
            logger.critical(f"An error occurred during Discord initialization: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred during Discord initialization: {e}", exc_info=True)
            raise

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        """
        Asynchronously fetches the latest messages using the custom polling engine.
        """
        if not self.is_initialized:
            logger.error("Discord interface is not ready. Cannot fetch messages.")
            return []

        try:
            loop = asyncio.get_running_loop()
            url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"

            response = await loop.run_in_executor(None, lambda: self.session.get(url, timeout=10))

            if response.status_code == 200:
                messages_data = response.json()

                # We convert the raw API data into the same simple dictionary format
                # that the rest of our application expects.
                processed_messages = []
                for msg in messages_data:
                    # Parse timestamp safely
                    timestamp_str = msg.get("timestamp")
                    timestamp = datetime.fromisoformat(timestamp_str).astimezone(
                        timezone.utc) if timestamp_str else None

                    processed_messages.append({
                        "id": int(msg['id']),
                        "content": msg['content'],
                        "author": f"{msg['author']['username']}#{msg['author']['discriminator']}",
                        "timestamp": timestamp
                    })
                return processed_messages
            else:
                logger.error(
                    f"Failed to fetch messages from channel {channel_id}. Status: {response.status_code}, Response: {response.text}")
                return []

        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching messages: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Closes the requests session.
        """
        logger.info("Closing Discord interface session...")
        self.session.close()
        logger.info("Discord interface session closed.")

