import logging
import asyncio
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class DiscordInterface:
    """
    A specialist module responsible for all interactions with the Discord API.
    This is the "Master Intelligence" edition, capable of parsing both
    plain text messages and complex embeds, including all fields.
    """

    def __init__(self, config):
        self.config = config
        self.token = self.config.discord_user_token
        self.session = None
        self.user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        self.user = None

    async def initialize(self):
        """
        Initializes the aiohttp session and tests the connection.
        """
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
                    logger.info(
                        f"Discord interface initialized successfully. Logged in as {self.user['username']}#{self.user['discriminator']}.")
                else:
                    response_text = await response.text()
                    logger.critical(f"Discord login failed. Status: {response.status}, Response: {response_text}.")
                    raise ConnectionError("Improper token or credentials passed to Discord.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during Discord initialization: {e}", exc_info=True)
            raise

    async def get_latest_messages(self, channel_id: str, limit: int = 10) -> list:
        """
        Asynchronously fetches and processes the latest messages, handling both
        plain text and embeds by concatenating all readable text fields.
        """
        if not self.session:
            logger.error("Discord session not initialized. Cannot fetch messages.")
            return []

        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
        logger.debug(f"Fetching messages from channel ID: {channel_id}")
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    messages = await response.json()
                    logger.debug(f"Received {len(messages)} message(s) from Discord.")

                    processed_messages = []
                    for msg in messages:
                        # --- SURGICAL UPGRADE: The "Master Interrogator" ---
                        master_content = []

                        # Always start with the plain text content
                        if msg.get('content'):
                            master_content.append(msg['content'])

                        if msg.get('embeds'):
                            for embed in msg['embeds']:
                                if embed.get('title'):
                                    master_content.append(embed['title'])
                                if embed.get('description'):
                                    master_content.append(embed['description'])
                                # This is the critical upgrade: read all the fields
                                if embed.get('fields'):
                                    for field in embed['fields']:
                                        if field.get('name'):
                                            master_content.append(field['name'])
                                        if field.get('value'):
                                            master_content.append(field['value'])
                                if embed.get('footer') and embed['footer'].get('text'):
                                    master_content.append(embed['footer']['text'])

                        full_text = " ".join(master_content)
                        # --- END UPGRADE ---

                        if not full_text.strip():
                            continue

                        timestamp_obj = datetime.fromisoformat(msg['timestamp'])
                        processed_messages.append({
                            "id": int(msg['id']),
                            "content": full_text,  # Use the complete master text
                            "author": f"{msg['author']['username']}#{msg['author']['discriminator']}",
                            "timestamp": timestamp_obj
                        })
                    return processed_messages
                else:
                    response_text = await response.text()
                    logger.error(
                        f"Failed to fetch messages from channel {channel_id}. Status: {response.status}, Response: {response_text}")
                    return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching messages: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Gracefully closes the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Discord session closed.")