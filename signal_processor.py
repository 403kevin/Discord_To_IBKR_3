import logging
import asyncio
from datetime import datetime, timezone, timedelta
from collections import deque
from ib_insync import Option, MarketOrder, Order
from services.signal_parser import SignalParser

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    The "brain" of the bot. This class is responsible for taking raw messages,
    processing them into trades, and managing the lifecycle of those trades.
    It is the central orchestrator that uses the other specialist modules.
    """

    def __init__(self, config, ib_interface, discord_interface, sentiment_analyzer):
        """
        Initializes the SignalProcessor.
        """
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.sentiment_analyzer = sentiment_analyzer
        
        # --- SURGICAL ADDITION: Cooldown Module State ---
        # This dictionary will store the state for the consecutive loss monitor.
        # Key: channel_id, Value: {'consecutive_losses': int, 'on_cooldown': bool, 'end_time': datetime}
        self._channel_states = {}
        # --- END SURGICAL ADDITION ---

        # In-memory storage for active trades and processed message IDs.
        self.active_trades = {}
        self.processed_message_ids = deque(maxlen=self.config.processed_message_cache_size)

    # --- SURGICAL ADDITION: Cooldown Module Methods ---
    def get_cooldown_status(self, channel_id):
        """
        Checks the cooldown status for a given channel.
        """
        return self._channel_states.setdefault(channel_id, {
            'consecutive_losses': 0,
            'on_cooldown': False,
            'end_time': None
        })

    def reset_consecutive_losses(self, channel_id):
        """
        Resets the loss counter and cooldown status for a channel.
        """
        state = self.get_cooldown_status(channel_id)
        state['consecutive_losses'] = 0
        state['on_cooldown'] = False
        state['end_time'] = None
        logger.info(f"Consecutive loss counter for channel {channel_id} has been reset.")
    # --- END SURGICAL ADDITION ---

    async def process_signal(self, message: dict, profile: dict):
        """
        The main entry point for processing a single Discord message.
        """
        msg_id = message['id']
        if msg_id in self.processed_message_ids:
            return # Skip already processed messages
        self.processed_message_ids.append(msg_id)

        # 1. Parse the message using our specialist parser.
        parser = SignalParser(self.config)
        parsed_signal = parser.parse_signal_message(message['content'], profile)
        
        if not parsed_signal:
            return # The message was not a valid trade signal.

        # ... (rest of the processing logic will go here)
        logger.info(f"Successfully processed signal: {parsed_signal}")
        # In a full implementation, this is where you would create the contract,
        # perform sentiment analysis, check margin, and place the trade.

    async def monitor_active_trades(self):
        """
        This method will be responsible for managing all ongoing trades,
        checking for exit conditions (stop loss, take profit, timeout),
        and updating the internal state. In a real implementation, this
        would be a complex piece of logic.
        """
        # Placeholder for the trade monitoring logic.
        # It would iterate through self.active_trades.
        # When a trade closes with a loss, it would increment the counter:
        #
        # state = self.get_cooldown_status(channel_id)
        # state['consecutive_losses'] += 1
        # if state['consecutive_losses'] >= max_losses:
        #     state['on_cooldown'] = True
        #     state['end_time'] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)
        #     logger.warning(f"Channel {channel_id} placed on cooldown.")
        #
        # When a trade closes with a profit, it would call:
        # self.reset_consecutive_losses(channel_id)
        
        await asyncio.sleep(0.1) # Non-blocking sleep

