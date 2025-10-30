"""
Discord Interface - HTTP Polling Engine with Embed Support
FIXED: Now parses Discord embeds in addition to regular text messages
"""
import asyncio
import logging
from datetime import datetime
from aiohttp import ClientSession, ClientError


class DiscordInterface:
    """
    The true, lightweight, custom polling engine. It makes direct, authenticated
    HTTP requests to Discord's API to fetch messages, adhering to the project's
    core "fly under the radar" philosophy.
    
    FIXED: Now properly extracts and parses Discord embeds for channels like NITRO.
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

    def _parse_embed_to_text(self, embed):
        """
        Convert Discord embed to parseable text format.
        
        Handles various embed formats used by different trading channels:
        - NITRO: Uses fields for Contract, Price, etc.
        - Other channels might use title, description, or fields differently
        
        Args:
            embed: Discord embed object (dict)
            
        Returns:
            str: Parsed text that signal_parser can understand, or None if empty
        """
        parts = []
        
        # Check for title (some channels put the signal in the title)
        if 'title' in embed and embed['title']:
            title = embed['title'].strip()
            # Add title if it contains trading-relevant keywords
            if any(keyword in title.upper() for keyword in ['ENTRY', 'EXIT', 'BUY', 'SELL', 'CALL', 'PUT', 'CONTRACT']):
                parts.append(title)
        
        # Check for description (some channels use this for the main signal)
        if 'description' in embed and embed['description']:
            parts.append(embed['description'].strip())
        
        # Parse fields (NITRO and others use these for structured data)
        if 'fields' in embed:
            for field in embed['fields']:
                field_name = field.get('name', '').strip()
                field_value = field.get('value', '').strip()
                
                if field_name and field_value:
                    # Format as "FieldName: Value" for parsing
                    # Special handling for common field names
                    if field_name.upper() in ['CONTRACT', 'TICKER', 'SYMBOL']:
                        # Put contract info first as it's most important
                        parts.insert(0, f"{field_value}")
                    elif field_name.upper() in ['PRICE', 'ENTRY', 'ENTRY PRICE', 'ASK', 'BID']:
                        # Add price info
                        parts.append(f"Price: {field_value}")
                    elif field_name.upper() in ['STRIKE', 'EXPIRY', 'EXPIRATION', 'EXP']:
                        # Add strike/expiry info
                        parts.append(f"{field_name}: {field_value}")
                    else:
                        # Add as-is for other fields
                        parts.append(f"{field_name}: {field_value}")
        
        # Check footer (some channels put additional info here)
        if 'footer' in embed and 'text' in embed['footer']:
            footer_text = embed['footer']['text'].strip()
            if footer_text:
                parts.append(footer_text)
        
        # Join all parts with spaces
        combined_text = ' '.join(parts) if parts else None
        
        # Log what we extracted for debugging
        if combined_text:
            logging.debug(f"Extracted from embed: '{combined_text}'")
        
        return combined_text

    async def poll_for_new_messages(self, channel_id, last_processed_ids):
        """
        Polls a specific channel for new messages using a direct API call.
        FIXED: Now properly handles both text messages and embeds.
        
        Args:
            channel_id: Discord channel ID to poll
            last_processed_ids: List of already processed message IDs
            
        Returns:
            List of tuples: [(msg_id, content, timestamp), ...]
            Content can be from regular text OR parsed from embeds
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
                    # Skip if already processed
                    if msg['id'] in last_processed_ids:
                        continue
                    
                    timestamp = datetime.fromisoformat(msg['timestamp'])
                    msg_id = msg['id']
                    
                    # Try to get content from various sources
                    content_found = False
                    
                    # 1. Check for regular text content
                    if msg.get('content') and msg['content'].strip():
                        text_content = msg['content'].strip()
                        
                        # Check if it's just a role mention (like NITRO does)
                        if text_content.startswith('<@&') and text_content.endswith('>'):
                            logging.debug(f"Message {msg_id} is just a role mention, checking for embeds...")
                        else:
                            # Regular text message with actual content
                            new_messages.append((msg_id, text_content, timestamp))
                            content_found = True
                    
                    # 2. Check for embeds (NITRO and other channels use these)
                    if not content_found and 'embeds' in msg and msg['embeds']:
                        logging.debug(f"Message {msg_id} has {len(msg['embeds'])} embed(s)")
                        
                        # Process each embed (usually just one)
                        for embed in msg['embeds']:
                            embed_content = self._parse_embed_to_text(embed)
                            if embed_content:
                                # Add "ENTRY" prefix if not present (many embeds assume entry)
                                if 'ENTRY' not in embed_content.upper() and 'EXIT' not in embed_content.upper():
                                    # Check if embed title or any field suggests it's an entry
                                    if any(word in str(embed).upper() for word in ['ENTRY', 'OPENING', 'BUY']):
                                        embed_content = f"ENTRY {embed_content}"
                                
                                new_messages.append((msg_id, embed_content, timestamp))
                                content_found = True
                                logging.info(f"📎 Parsed embed from message {msg_id}: {embed_content[:100]}...")
                                break  # Usually only need first embed
                    
                    # 3. Log if we found a message with no parseable content
                    if not content_found:
                        # Check if message has attachments or other content types
                        if msg.get('attachments'):
                            logging.debug(f"Message {msg_id} has attachments but no text/embed content")
                        elif msg.get('components'):
                            logging.debug(f"Message {msg_id} has components but no text/embed content")
                        else:
                            # Empty message or just a role ping with no embed
                            author = msg.get('author', {}).get('username', 'Unknown')
                            logging.debug(f"Message {msg_id} from {author} has no parseable content")
                
        except asyncio.TimeoutError:
            logging.error(f"Request to Discord API timed out for channel {channel_id}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while polling Discord channel {channel_id}: {e}", exc_info=True)

        # Log summary (debug level to reduce noise)
        if new_messages:
            logging.debug(f"Found {len(new_messages)} new message(s) in channel {channel_id}.")
            # Log first message for debugging
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                first_msg = new_messages[0]
                logging.debug(f"  First message: ID={first_msg[0]}, Content='{first_msg[1][:50]}...'")
        
        return new_messages

    async def test_embed_parsing(self, channel_id):
        """
        Test function to debug embed parsing for a specific channel.
        Useful for checking what NITRO and other embed-based channels are sending.
        """
        if not self.is_initialized():
            await self.initialize()
        
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=10"
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch messages: {response.status}")
                    return
                
                messages = await response.json()
                
                logging.info(f"\n{'='*60}")
                logging.info(f"EMBED PARSING TEST for channel {channel_id}")
                logging.info(f"{'='*60}")
                
                for msg in messages[:5]:  # Just check last 5 messages
                    msg_id = msg['id']
                    author = msg.get('author', {}).get('username', 'Unknown')
                    content = msg.get('content', '')
                    
                    logging.info(f"\nMessage {msg_id} from {author}:")
                    logging.info(f"  Text content: '{content}'")
                    
                    if 'embeds' in msg and msg['embeds']:
                        for i, embed in enumerate(msg['embeds']):
                            logging.info(f"  Embed {i+1}:")
                            if 'title' in embed:
                                logging.info(f"    Title: {embed['title']}")
                            if 'description' in embed:
                                logging.info(f"    Description: {embed['description']}")
                            if 'fields' in embed:
                                for field in embed['fields']:
                                    logging.info(f"    Field: {field.get('name')} = {field.get('value')}")
                            
                            # Show parsed result
                            parsed = self._parse_embed_to_text(embed)
                            logging.info(f"    PARSED TO: '{parsed}'")
                    else:
                        logging.info(f"  No embeds")
                
                logging.info(f"{'='*60}\n")
                
        except Exception as e:
            logging.error(f"Error in embed test: {e}", exc_info=True)
