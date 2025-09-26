import asyncio
import logging
from datetime import datetime
from aiohttp import ClientSession, ClientError

class DiscordInterface:
    """
    The true, lightweight, custom polling engine. It makes direct, authenticated
    HTTP requests to Discord's API to fetch messages, adhering to the project's
    core "fly under the radar" philosophy.
    """
    def __init__(self, config):
        self.config = config
        self.token = config.discord_user_token
        self.session = None
        self._is_initialized = False
        self.headers = {
            "Authorization": self.token,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    async def initialize(self):
        """Initializes the aiohttp session for making API requests."""
        if not self.session or self.session.closed:
            self.session = ClientSession(headers=self.headers)
            self._is_initialized = True
            logging.info("Discord HTTP interface initialized successfully.")
            await self._verify_token()

    async def _verify_token(self):
        """Makes a simple API call to verify the auth token is valid."""
        if not self.is_initialized(): return
        try:
            async with self.session.get("https://discord.com/api/v9/users/@me") as response:
                if response.status == 200:
                    user_data = await response.json()
                    logging.info(f"Discord token verified. Authenticated as {user_data['username']}#{user_data['discriminator']}")
                else:
                    logging.error(f"Discord token is invalid. Status: {response.status}. Please check your .env file.")
                    self._is_initialized = False
        except ClientError as e:
            logging.error(f"An error occurred during Discord token verification: {e}")
            self._is_initialized = False

    def is_initialized(self):
        """Checks if the interface is active."""
        return self._is_initialized and self.session and not self.session.closed

    async def shutdown(self):
        """Gracefully closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._is_initialized = False
        logging.info("Discord HTTP interface has been shut down.")
        
    async def poll_for_new_messages(self, channel_id, last_processed_ids):
        """Polls a specific channel for new messages using a direct API call."""
        if not self.is_initialized():
            logging.error("Cannot poll for messages, Discord interface is not initialized.")
            return []
        
        # --- THE SURGICAL FIX: The Ever-Present Report Folder ---
        # Initialize the list at the top of the function. This guarantees it
        # always exists, even if the API call in the try block fails.
        new_messages = []
        
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=50"
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch Discord messages. Status: {response.status}, Response: {await response.text()}")
                    return [] # Return the empty list on failure
                
                messages = await response.json()
                # API returns newest first, so we reverse to process oldest unread first
                for msg in reversed(messages):
                    if msg['id'] not in last_processed_ids:
                        timestamp = datetime.fromisoformat(msg['timestamp'])
                        new_messages.append((msg['id'], msg['content'], timestamp))
                
        except asyncio.TimeoutError:
            logging.error(f"Request to Discord API timed out for channel {channel_id}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while polling Discord channel {channel_id}: {e}", exc_info=True)

        # The final check now safely operates on the 'new_messages' list, which is guaranteed to exist.
        if new_messages:
            logging.debug(f"Found {len(new_messages)} new message(s) in channel {channel_id}.")
        
        return new_messages

