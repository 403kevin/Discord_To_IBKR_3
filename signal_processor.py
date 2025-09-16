import logging
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from collections import deque  # <-- SURGICAL FIX: The missing tool is now imported.
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
        
        self._channel_states = {}
        self.active_trades = {}
        self.processed_message_ids = deque(maxlen=self.config.processed_message_cache_size)

    def get_cooldown_status(self, channel_id):
        """Checks the cooldown status for a given channel."""
        return self._channel_states.setdefault(str(channel_id), {
            'consecutive_losses': 0,
            'on_cooldown': False,
            'end_time': None
        })

    def reset_consecutive_losses(self, channel_id):
        """Resets the loss counter and cooldown status for a channel."""
        state = self.get_cooldown_status(str(channel_id))
        state['consecutive_losses'] = 0
        state['on_cooldown'] = False
        state['end_time'] = None
        logger.info(f"Consecutive loss counter for channel {channel_id} has been reset.")

    async def process_signal(self, message: dict, profile: dict):
        """
        The main entry point for processing a single Discord message.
        This is the complete, battle-hardened ignition system.
        """
        msg_id = message['id']
        if msg_id in self.processed_message_ids:
            return # Skip already processed messages
        
        # --- SURGICAL FIX: Put the Bouncer at the Door ---
        # This is the new gatekeeper logic that enforces the max age rule.
        message_timestamp = message['timestamp']
        current_time = datetime.now(timezone.utc)
        message_age = (current_time - message_timestamp).total_seconds()
        
        if message_age > self.config.signal_max_age_seconds:
            # This signal is too old, reject it immediately.
            return 
        # --- END SURGICAL FIX ---

        # If the message is fresh, we can now add it to our memory and process it.
        self.processed_message_ids.append(msg_id)

        parser = SignalParser(self.config)
        parsed_signal = parser.parse_signal_message(message['content'], profile)
        
        if not parsed_signal:
            return

        logger.info(f"Successfully parsed signal: {parsed_signal}")

        # --- SENTIMENT ANALYSIS GATE ---
        if self.config.sentiment_filter['enabled']:
            sentiment_score = await self.sentiment_analyzer.analyze_sentiment(parsed_signal['ticker'])
            if sentiment_score is None or sentiment_score < self.config.sentiment_filter['sentiment_threshold']:
                logger.warning(f"Trade for {parsed_signal['ticker']} halted due to low sentiment score: {sentiment_score}")
                return

        # --- BUILD THE CONTRACT ---
        try:
            contract = Option(
                symbol=parsed_signal['ticker'],
                lastTradeDateOrContractMonth=parsed_signal['expiry'],
                strike=parsed_signal['strike'],
                right=parsed_signal['option_type'],
                exchange='SMART',
                currency='USD'
            )
            await self.ib_interface.ib.qualifyContractsAsync(contract)
        except Exception as e:
            logger.error(f"Contract qualification failed for {parsed_signal}: {e}")
            return

        # --- CALCULATE TRADE SIZE ---
        # NOTE: This is a simplified sizing model. A real-world bot would need
        # to fetch the live price to calculate the exact number of contracts.
        # For now, we will assume a simple quantity for demonstration.
        quantity = 1 # Placeholder quantity

        # --- BUILD THE ORDER ---
        order = MarketOrder(
            action=parsed_signal['action'].upper(), # Ensure action is uppercase
            totalQuantity=quantity
        )

        # --- PLACE THE TRADE ---
        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade,
                    "entry_time": datetime.now(timezone.utc),
                    "profile": profile
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']}.")
            else:
                logger.warning(f"Trade for {parsed_signal['ticker']} was not placed (likely due to an existing open order).")

        except Exception as e:
            logger.error(f"Failed to place trade for {parsed_signal['ticker']}: {e}")


    async def monitor_active_trades(self):
        """
        This method will be responsible for managing all ongoing trades.
        """
        if not self.active_trades:
            return

        # Create a copy of the keys to iterate over, as the dictionary may change size.
        trade_ids = list(self.active_trades.keys())
        
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info:
                continue
            
            trade = trade_info['trade_obj']
            
            # Check the status of the order
            if trade.orderStatus.status == 'Filled':
                logger.info(f"Trade {trade_id} ({trade.contract.localSymbol}) has been filled.")
                # Once filled, a real bot would attach a stop loss, native trail, etc.
                # It would then be managed by a different part of the monitoring logic.
                # For now, we will just remove it from active *entry* monitoring.
                del self.active_trades[trade_id]
            
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                logger.warning(f"Trade {trade_id} ({trade.contract.localSymbol}) is no longer active. Status: {trade.orderStatus.status}")
                del self.active_trades[trade_id]

        await asyncio.sleep(0.1) # Non-blocking sleep

