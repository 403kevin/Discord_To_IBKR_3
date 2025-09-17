import logging
import asyncio
import uuid
from datetime import datetime, timezone
from collections import deque
from ib_insync import Option, MarketOrder
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
        This is the complete, battle-hardened ignition system with diagnostics and perfect memory.
        """
        msg_id = message['id']
        
        # --- Gatekeeper #1: Short-Term Memory ---
        if msg_id in self.processed_message_ids:
            return # Silently ignore duplicates, this is expected behavior.
        
        # --- SURGICAL FIX: The Perfect Memory ---
        # The bot will now remember EVERY message it sees, not just processed ones.
        # This is the key to stopping the log spam.
        self.processed_message_ids.append(msg_id)
        
        # --- Gatekeeper #2: The Bouncer (Age Check) ---
        message_timestamp = message['timestamp']
        current_time = datetime.now(timezone.utc)
        message_age = (current_time - message_timestamp).total_seconds()
        
        if message_age > self.config.signal_max_age_seconds:
            logger.info(f"Signal REJECTED (ID: {msg_id}): Stale message (Age: {message_age:.0f}s > {self.config.signal_max_age_seconds}s).")
            return 
        
        # --- Gatekeeper #3: The Translator ---
        parser = SignalParser(self.config)
        parsed_signal = parser.parse_signal_message(message['content'], profile)
        
        if not parsed_signal:
            logger.info(f"Signal IGNORED (ID: {msg_id}): Message content did not parse into a valid trade signal.")
            return

        # --- If we reach here, the signal is valid and fresh. ---
        logger.info(f"Signal ACCEPTED (ID: {msg_id}): Successfully parsed signal: {parsed_signal}")

        # --- Gatekeeper #4: Sentiment Analysis ---
        if self.config.sentiment_filter['enabled']:
            sentiment_score = await self.sentiment_analyzer.analyze_sentiment(parsed_signal['ticker'])
            if sentiment_score is None or sentiment_score < self.config.sentiment_filter['sentiment_threshold']:
                logger.warning(f"Trade for {parsed_signal['ticker']} HALTED (ID: {msg_id}): Low sentiment score: {sentiment_score}")
                return

        # --- Build the Contract ---
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
            logger.error(f"Contract qualification FAILED for {parsed_signal} (ID: {msg_id}): {e}")
            return

        # --- Calculate Trade Size (Placeholder) ---
        quantity = 1

        # --- Build the Order ---
        order = MarketOrder(
            action=parsed_signal['action'].upper(),
            totalQuantity=quantity
        )

        # --- Place the Trade ---
        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade,
                    "entry_time": datetime.now(timezone.utc),
                    "profile": profile
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']} (Original Msg ID: {msg_id}).")
            else:
                logger.warning(f"Trade for {parsed_signal['ticker']} NOT PLACED (ID: {msg_id}): Likely due to an existing open order.")
        except Exception as e:
            logger.error(f"Failed to place trade for {parsed_signal['ticker']} (ID: {msg_id}): {e}")


    async def monitor_active_trades(self):
        """
        Manages all ongoing trades.
        """
        if not self.active_trades:
            return

        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info: continue
            
            trade = trade_info['trade_obj']
            
            if trade.orderStatus.status == 'Filled':
                logger.info(f"Trade {trade_id} ({trade.contract.localSymbol}) has been filled.")
                del self.active_trades[trade_id]
            
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                logger.warning(f"Trade {trade_id} ({trade.contract.localSymbol}) is no longer active. Status: {trade.orderStatus.status}")
                del self.active_trades[trade_id]

        await asyncio.sleep(0.1)

