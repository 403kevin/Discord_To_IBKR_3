"""
Discord HTTP Interface - FIXED with Option 3 Clean Logging
"""
import requests
import time
import logging
from typing import List, Dict, Optional
import os


class DiscordInterface:
    """
    Discord HTTP interface for polling messages from channels.
    Uses HTTP API instead of WebSocket for simpler implementation.
    """
    
    def __init__(self, token: str, channel_ids: List[str]):
        """
        Initialize Discord HTTP interface.
        
        Args:
            token: Discord bot token or user token
            channel_ids: List of channel IDs to monitor
        """
        self.token = token
        self.channel_ids = channel_ids
        self.base_url = "https://discord.com/api/v10"
        self.headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }
        self.processed_message_ids = set()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests per channel
        
        logging.info("Discord HTTP interface initialized successfully.")
    
    def verify_token(self) -> bool:
        """
        Verify that the Discord token is valid.
        
        Returns:
            bool: True if token is valid, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/users/@me")
            if response.status_code == 200:
                user_data = response.json()
                username = user_data.get('username', 'Unknown')
                discriminator = user_data.get('discriminator', '0')
                logging.info(f"Discord token verified. Authenticated as {username}#{discriminator}")
                return True
            else:
                logging.error(f"Discord token verification failed: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error verifying Discord token: {e}")
            return False
    
    def fetch_messages(self, channel_id: str, limit: int = 50, before: Optional[str] = None) -> List[Dict]:
        """
        Fetch messages from a Discord channel.
        
        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to fetch (1-100)
            before: Message ID to fetch messages before (for pagination)
            
        Returns:
            List of message dictionaries
        """
        # Rate limiting
        now = time.time()
        last_request = self.last_request_time.get(channel_id, 0)
        time_since_last = now - last_request
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # Build URL
        url = f"{self.base_url}/channels/{channel_id}/messages"
        params = {"limit": min(limit, 100)}
        if before:
            params["before"] = before
        
        try:
            response = self.session.get(url, params=params)
            self.last_request_time[channel_id] = time.time()
            
            if response.status_code == 200:
                messages = response.json()
                return messages
            elif response.status_code == 429:
                # Rate limited
                retry_after = response.json().get('retry_after', 5)
                logging.warning(f"Rate limited on channel {channel_id}. Retry after {retry_after}s")
                time.sleep(retry_after)
                return []
            else:
                logging.error(f"Failed to fetch messages from {channel_id}: {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"Error fetching messages from {channel_id}: {e}")
            return []
    
    def _parse_embed_to_text(self, embed: Dict, log_parsing: bool = True) -> str:
        """
        Convert Discord embed to parseable text format.
        
        Args:
            embed: Discord embed dict
            log_parsing: If True, log the parsed text (default True for live trading)
            
        Returns:
            Parsed text string from embed
        """
        text = ""
        
        # Add title if present
        if embed.get('title'):
            text += embed['title'] + " "
        
        # Add description if present
        if embed.get('description'):
            text += embed['description'] + " "
        
        # Parse fields (most important for trading signals)
        if embed.get('fields'):
            for field in embed['fields']:
                field_name = field.get('name', '')
                field_value = field.get('value', '')
                text += f"{field_name} {field_value} "
        
        # Add footer text if present
        if embed.get('footer') and embed.get('footer', {}).get('text'):
            text += embed['footer']['text']
        
        # Only log if requested (True for live trading, False for warmup)
        if log_parsing and text:
            logging.info(f"📎 Parsed embed: {text[:100]}...")
        
        return text.strip()
    
    def get_new_messages(self, channel_id: str) -> List[Dict]:
        """
        Get new messages from a channel that haven't been processed yet.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            List of new message dictionaries with combined text and embed content
        """
        messages = self.fetch_messages(channel_id, limit=10)
        new_messages = []
        
        for msg in messages:
            msg_id = msg['id']
            
            # Skip if already processed
            if msg_id in self.processed_message_ids:
                continue
            
            # Mark as processed
            self.processed_message_ids.add(msg_id)
            
            # Combine text content with embed content
            text_content = msg.get('content', '')
            
            # Parse embeds if present (log during live trading)
            if msg.get('embeds'):
                embed_text = self._parse_embed_to_text(msg['embeds'][0], log_parsing=True)
                # Combine text and embed content
                combined_text = text_content + ' ' + embed_text if text_content else embed_text
                msg['content'] = combined_text.strip()
            
            new_messages.append(msg)
        
        return new_messages
    
    def warmup_message_cache(self):
        """
        Load recent messages and mark them as processed to avoid acting on old signals.
        This should be called once at startup.
        """
        logging.info("🔥 Warming up message cache (marking existing messages as processed)...")
        
        embed_count = 0  # Track embeds during warmup
        
        for channel_id in self.channel_ids:
            messages = self.fetch_messages(channel_id, limit=50)
            for msg in messages:
                msg_id = msg['id']
                self.processed_message_ids.add(msg_id)
                
                # Parse embeds if present (silently during warmup)
                if msg.get('embeds'):
                    embed_count += 1
                    embed_text = self._parse_embed_to_text(msg['embeds'][0], log_parsing=False)
        
        # Single clean summary line
        logging.info(f"✅ Warmup complete: Marked {len(self.processed_message_ids)} messages ({embed_count} with embeds) as processed")
        logging.info("🎯 Bot is now live and monitoring for NEW signals only")
    
    def poll_all_channels(self) -> List[tuple]:
        """
        Poll all configured channels for new messages.
        
        Returns:
            List of tuples: (channel_id, message_dict)
        """
        all_new_messages = []
        
        for channel_id in self.channel_ids:
            new_messages = self.get_new_messages(channel_id)
            for msg in new_messages:
                all_new_messages.append((channel_id, msg))
        
        return all_new_messages
    
    def shutdown(self):
        """
        Clean shutdown of Discord interface.
        """
        logging.info("Discord interface shutting down...")
        self.session.close()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example: Read token from environment variable
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logging.error("DISCORD_TOKEN environment variable not set")
        exit(1)
    
    # Example channel IDs (replace with your actual channel IDs)
    channel_ids = [
        "1234567890123456789",  # Replace with real channel ID
    ]
    
    # Initialize interface
    discord = DiscordInterface(token, channel_ids)
    
    # Verify token
    if not discord.verify_token():
        logging.error("Failed to verify Discord token")
        exit(1)
    
    # Warmup (mark existing messages as processed)
    discord.warmup_message_cache()
    
    # Poll for new messages
    logging.info("Starting to poll for new messages...")
    try:
        while True:
            new_messages = discord.poll_all_channels()
            
            if new_messages:
                for channel_id, msg in new_messages:
                    content = msg.get('content', '')
                    author = msg.get('author', {}).get('username', 'Unknown')
                    logging.info(f"New message in {channel_id} from {author}: {content[:100]}")
            
            time.sleep(5)  # Poll every 5 seconds
            
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        discord.shutdown()
