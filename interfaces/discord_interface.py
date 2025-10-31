import asyncio
import logging
from datetime import datetime
from aiohttp import ClientSession, ClientError

class DiscordInterface:
    """
    Enhanced Discord HTTP polling engine with embed support.
    Maintains the "ghost" philosophy while extracting both text and embed content.
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
        if not self.is_initialized(): 
            return
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
        """
        Polls a specific channel for new messages, including embedded content.
        Returns messages with both regular content and embed data.
        """
        if not self.is_initialized():
            logging.error("Cannot poll for messages, Discord interface is not initialized.")
            return []
        
        new_messages = []
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=50"
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch Discord messages. Status: {response.status}, Response: {await response.text()}")
                    return []
                
                messages = await response.json()
                for msg in reversed(messages):
                    if msg['id'] not in last_processed_ids:
                        timestamp = datetime.fromisoformat(msg['timestamp'])
                        
                        # Extract regular content
                        content = msg.get('content', '')
                        
                        # ENHANCED: Extract embed content if present
                        if msg.get('embeds'):
                            embed_text = self._extract_embed_content(msg['embeds'])
                            if embed_text:
                                # Combine regular content with embed content
                                if content and embed_text:
                                    # If there's both content and embed, combine them
                                    full_content = f"{content}\n{embed_text}"
                                elif embed_text:
                                    # If only embed, use that
                                    full_content = embed_text
                                else:
                                    # If only regular content
                                    full_content = content
                                
                                # Log when we find embeds (for debugging)
                                if embed_text:
                                    logging.debug(f"Found embed in message {msg['id']}: {embed_text[:100]}...")
                            else:
                                full_content = content
                        else:
                            full_content = content
                        
                        new_messages.append((msg['id'], full_content, timestamp))
                
        except asyncio.TimeoutError:
            logging.error(f"Request to Discord API timed out for channel {channel_id}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while polling Discord channel {channel_id}: {e}", exc_info=True)

        if new_messages:
            logging.debug(f"Found {len(new_messages)} new message(s) in channel {channel_id}.")
        
        return new_messages

    def _extract_embed_content(self, embeds):
        """
        Extracts relevant trading signal content from Discord embeds.
        Handles various embed formats used by trading signal providers.
        """
        extracted_parts = []
        
        for embed in embeds:
            # Extract title
            if embed.get('title'):
                extracted_parts.append(embed['title'])
            
            # Extract description
            if embed.get('description'):
                extracted_parts.append(embed['description'])
            
            # Extract fields (common in trading signal embeds)
            if embed.get('fields'):
                for field in embed['fields']:
                    field_name = field.get('name', '')
                    field_value = field.get('value', '')
                    
                    # Combine field name and value
                    if field_name and field_value:
                        extracted_parts.append(f"{field_name}: {field_value}")
                    elif field_value:
                        extracted_parts.append(field_value)
            
            # Extract footer (sometimes contains additional info)
            if embed.get('footer') and embed['footer'].get('text'):
                extracted_parts.append(embed['footer']['text'])
            
            # Extract author (sometimes contains trader name or signal type)
            if embed.get('author') and embed['author'].get('name'):
                extracted_parts.append(embed['author']['name'])
        
        # Join all parts with newlines
        combined_text = '\n'.join(extracted_parts)
        
        # Clean up common formatting issues
        combined_text = combined_text.replace('**', '')  # Remove bold markdown
        combined_text = combined_text.replace('__', '')  # Remove underline markdown
        combined_text = combined_text.replace('```', '') # Remove code blocks
        
        return combined_text.strip()
