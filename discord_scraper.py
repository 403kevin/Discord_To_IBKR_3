# discord_scraper.py

import discord
import asyncio
import logging
from typing import Callable

class DiscordScraper(discord.Client):
    """
    A Discord client (self-bot) that scrapes messages from specified channels.
    WARNING: Using a self-bot is against Discord's Terms of Service and can result in account termination.
    Use at your own risk.
    """
    def __init__(self, target_channel_ids: list[int], on_message_callback: Callable, **options):
        super().__init__(**options)
        self.target_channel_ids = set(target_channel_ids)
        self.on_message_callback = on_message_callback
        self.ready = False
        logging.info("Discord Scraper initialized.")

    async def on_ready(self):
        """
        Called when the bot has successfully connected to Discord.
        """
        logging.info(f'Logged in as {self.user.name} ({self.user.id})')a
        self.ready = True
        # You can optionally fetch and print channel names for verification
        for channel_id in self.target_channel_ids:
            channel = self.get_channel(channel_id)
            if channel:
                logging.info(f"Successfully monitoring channel: #{channel.name} in '{channel.guild.name}'")
            else:
                logging.warning(f"Could not find channel with ID: {channel_id}. Check permissions and ID.")


    async def on_message(self, message: discord.Message):
        """
        Called every time a message is sent in any channel the user has access to.
        """
        # Don't process our own messages or messages from other bots
        if message.author == self.user or message.author.bot:
            return

        # Check if the message is from one of our target channels
        if message.channel.id in self.target_channel_ids:
            channel_name = message.channel.name
            logging.info(f"New message detected in monitored channel '{channel_name}': '{message.content}'")
            
            # Pass the message content and channel name to the main application logic
            try:
                # The callback should handle parsing and deciding whether to trade
                self.on_message_callback(message.content, channel_name)
            except Exception as e:
                logging.error(f"Error processing message from Discord: {e}")

def run_scraper(token: str, channel_ids: list[int], callback: Callable):
    """
    Function to run the Discord scraper in a separate thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Define intents to receive message content
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True # Required to read message content

    client = DiscordScraper(target_channel_ids=channel_ids, on_message_callback=callback, intents=intents)
    
    try:
        # The `bot=False` argument is what designates this as a user account login (self-bot)
        client.run(token, bot=False)
    except Exception as e:
        logging.critical(f"A critical error occurred in the Discord client: {e}")
        logging.critical("This may be due to an invalid token or Discord API changes.")

