import asyncio
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

class DiscordInterface:
    """
    Manages all communication with Discord using Playwright to "fly under the radar".
    This version is hardened for robust startup and shutdown sequences.
    """
    def __init__(self, config):
        self.config = config
        self.playwright = None
        self.browser = None
        self.page = None
        self._is_initialized = False

    async def initialize(self):
        """Initializes Playwright, launches a browser, and logs into Discord."""
        try:
            self.playwright = await async_playwright().start()
            # Using chromium, but could be firefox or webkit
            self.browser = await self.playwright.chromium.launch(headless=True) 
            self.page = await self.browser.new_page()
            
            logging.info("Navigating to Discord login page...")
            await self.page.goto("https://discord.com/login")
            
            # Using more specific selectors for robustness
            await self.page.fill('input[name="email"]', self.config.discord_user_email)
            await self.page.fill('input[name="password"]', self.config.discord_user_password)
            await self.page.click('button[type="submit"]')
            
            # Wait for the main app interface to load, indicating a successful login
            await self.page.wait_for_selector('div[class*="guilds-"]', timeout=30000)
            
            self._is_initialized = True
            logging.info(f"Discord interface initialized successfully. Logged in as {self.config.discord_user_email}.")
            return True
        except PlaywrightTimeoutError:
            logging.error("Timeout occurred during Discord login. Possible reasons: incorrect credentials, a required captcha, or 2FA.")
            await self.shutdown()
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during Discord initialization: {e}", exc_info=True)
            await self.shutdown()
            return False

    def is_initialized(self):
        """
        SURGICAL FIX: New method for the main orchestrator to safely check
        if the interface is active before attempting shutdown.
        """
        return self._is_initialized and self.page and not self.page.is_closed()

    async def shutdown(self):
        """
        SURGICAL FIX: New method to gracefully close the browser and Playwright,
        preventing ghost processes.
        """
        if self.browser and not self.browser.is_closed():
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._is_initialized = False
        logging.info("Discord interface has been shut down.")
        
    async def poll_for_new_messages(self, channel_id, last_processed_ids):
        """
        Polls a specific channel for new messages that have not yet been processed.
        NOTE: This is the most fragile part of the bot. Discord UI changes can
        break these selectors.
        """
        if not self.is_initialized():
            logging.error("Cannot poll for messages, Discord interface is not initialized.")
            return []
        
        try:
            channel_url = f"https://discord.com/channels/@me/{channel_id}" # Assuming DMs for now, adjust if server channel
            if self.page.url != channel_url:
                await self.page.goto(channel_url, wait_until="networkidle")

            # This selector targets individual messages in a channel. It needs to be verified.
            messages = await self.page.query_selector_all('li[class*="messageListItem-"]')
            
            new_messages = []
            # Iterate in reverse to get oldest messages first
            for message_element in reversed(messages):
                message_id = await message_element.get_attribute('id')
                
                # Extract the snowflake ID part
                snowflake_id = message_id.split('-')[-1]

                if snowflake_id not in last_processed_ids:
                    content_element = await message_element.query_selector('div[class*="contents-"]')
                    if content_element:
                        content = await content_element.inner_text()
                        # This is a basic timestamp from the client; not ideal but a starting point
                        timestamp = datetime.now() 
                        new_messages.append((snowflake_id, content, timestamp))

            return new_messages

        except Exception as e:
            logging.error(f"Error polling Discord channel {channel_id}: {e}", exc_info=True)
            # A critical failure here might mean we need to re-initialize
            self._is_initialized = False 
            return []

