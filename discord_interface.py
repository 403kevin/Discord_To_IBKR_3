import logging
import asyncio
import discord
from datetime import datetime

logger = logging.getLogger(__name__)

class DiscordInterface:
    """
    A specialist module responsible for all interactions with the Discord API.
    This class uses a direct HTTP polling method to be discreet and "fly under the radar,"
    adhering to the project's constitution. It does not use the official bot gateway.
    """

    def __init__(self, config):
        """
        Initializes the Discord interface.
        Args:
            config: The main configuration object.
        """
        self.config = config
        self.token = self.config.discord_user_token
        self.session = None  # To be initialized in an async context
        self.user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        self.user = None

    async def initialize(self):
        """
        Initializes the aiohttp session and tests the connection by fetching user info.
        This confirms the token is valid and the disguise is working.
        """
        import aiohttp  # Import here to keep dependency local to async context
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": self.token,
                "User-Agent": self.user_agent,
                "Content-Type": "application/json"
            }
        )
        try:
            async with self.session.get('https://discord.com/api/v9/users/@me') as response:
                if response.status == 200:
                    self.user = await response.json()
                    logger.info(f"Discord interface initialized successfully. Logged in as {self.user['username']}#{self.user['discriminator']}.")
                else:
                    logger.critical(f"Discord login failed. Status: {response.status}. Please check your DISCORD_AUTH_TOKEN.")
                    raise discord.errors.LoginFailure("Improper token or credentials passed.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during Discord initialization: {e}", exc_info=True)
            raise

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        """
        Asynchronously fetches the latest messages from a specific Discord channel using direct HTTP requests.
        """
        if not self.session:
            logger.error("Discord session not initialized. Cannot fetch messages.")
            return []

        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    messages = await response.json()
                    # Convert the raw API response into our standard, clean format.
                    processed_messages = []
                    for msg in messages:
                        # Parse the timestamp string into a datetime object
                        timestamp_obj = datetime.fromisoformat(msg['timestamp'])
                        processed_messages.append({
                            "id": int(msg['id']),
                            "content": msg['content'],
                            "author": f"{msg['author']['username']}#{msg['author']['discriminator']}",
                            "timestamp": timestamp_obj
                        })
                    return processed_messages
                else:
                    logger.error(f"Failed to fetch messages from channel {channel_id}. Status: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching messages from channel {channel_id}: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Gracefully closes the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Discord session closed.")

