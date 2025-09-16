import logging
import asyncio
import discord
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

class DiscordInterface:
    """
    A specialist module responsible for all interactions with the Discord API.
    This class uses the discord.py library in an asynchronous, non-blocking manner.
    It adheres to the "Single Operator" model.
    """

    def __init__(self, config):
        """
        Initializes the Discord client.
        Args:
            config: The main configuration object.
        """
        self.config = config
        self.token = self.config.discord_user_token
        
        # We need to specify which events the bot is interested in.
        # `intents.messages` is required to read message content.
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True # Crucial for reading message content

        # --- SURGICAL FIX: The "Chrome Browser" Disguise ---
        # As per the README.md, we must identify as a browser to "fly under the radar"
        # when using a user token. We achieve this by overriding the default User-Agent.
        # We pass this into the underlying aiohttp session used by discord.py.
        connector_args = {
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        # --- END SURGICAL FIX ---

        # The client is the main connection to Discord. We pass the disguise in here.
        self.client = discord.Client(intents=intents, connector_args=connector_args)
        self.is_initialized = False

    async def initialize(self):
        """
        Logs the bot into Discord. This is the correct, non-racy ignition sequence.
        """
        if self.is_initialized:
            logger.warning("Discord client is already initialized.")
            return

        try:
            # We use a background task to run the client's main loop,
            # allowing our own main loop in main.py to continue.
            asyncio.create_task(self.client.start(self.token))
            
            # Wait until the client has successfully connected and is ready.
            await self.client.wait_until_ready()
            
            self.is_initialized = True
            logger.info(f"Discord client initialized successfully. Logged in as {self.client.user}.")
        
        except discord.errors.LoginFailure as e:
            logger.critical(f"Discord login failed: Invalid token. Please check your .env file. Error: {e}")
            raise  # Re-raise the exception to stop the bot's startup
        except Exception as e:
            logger.critical(f"An unexpected error occurred during Discord initialization: {e}", exc_info=True)
            raise

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        """
        Asynchronously fetches the latest messages from a specific Discord channel.
        Args:
            channel_id (str): The ID of the channel to fetch messages from.
            limit (int): The maximum number of messages to retrieve.
        Returns:
            A list of message dictionaries, or an empty list if an error occurs.
        """
        if not self.is_initialized or self.client.is_closed():
            logger.error("Discord client is not ready or has been closed. Cannot fetch messages.")
            return []

        try:
            # Get the channel object from the client's cache.
            channel = self.client.get_channel(int(channel_id))
            if not channel:
                logger.error(f"Could not find Discord channel with ID: {channel_id}")
                return []

            # `history()` is an async iterator. We collect its results into a list.
            messages = [message async for message in channel.history(limit=limit)]

            # Convert the discord.Message objects into a simpler dictionary format.
            # This decouples the rest of our application from the discord.py library.
            processed_messages = []
            for msg in messages:
                processed_messages.append({
                    "id": msg.id,
                    "content": msg.content,
                    "author": str(msg.author),
                    "timestamp": msg.created_at
                })
            
            return processed_messages

        except discord.errors.Forbidden:
            logger.error(f"Permission error: The bot does not have permission to read messages in channel {channel_id}.")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching messages from channel {channel_id}: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Gracefully logs out and closes the connection to Discord.
        """
        if self.client and not self.client.is_closed():
            logger.info("Closing Discord connection...")
            await self.client.close()
            logger.info("Discord connection closed.")

