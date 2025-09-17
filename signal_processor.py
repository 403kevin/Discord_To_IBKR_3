import logging
import asyncio
import uuid
from datetime import datetime, timezone
from collections import deque
from ib_insync import Option, MarketOrder, Order
from services.signal_parser import SignalParser

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    The "brain" of the bot. This is the feature-complete version responsible
    for processing signals and managing the lifecycle of trades with detailed logging.
    """

    def __init__(self, config, ib_interface, discord_interface, sentiment_analyzer, telegram_interface):
        """
        Initializes the SignalProcessor with all its specialist tools.
        """
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.sentiment_analyzer = sentiment_analyzer
        self.telegram_interface = telegram_interface
        
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
        """
        msg_id = message['id']
        
        if msg_id in self.processed_message_ids:
            return
        
        self.processed_message_ids.append(msg_id)
        
        message_timestamp = message['timestamp']
        current_time = datetime.now(timezone.utc)
        message_age = (current_time - message_timestamp).total_seconds()
        
        if message_age > self.config.signal_max_age_seconds:
            logger.debug(f"Signal REJECTED (ID: {msg_id}): Stale message.")
            return 
        
        parser = SignalParser(self.config)
        parsed_signal = parser.parse_signal_message(message['content'], profile)
        
        if not parsed_signal:
            logger.debug(f"Signal IGNORED (ID: {msg_id}): Message content did not parse.")
            return

        logger.info(f"Signal ACCEPTED (ID: {msg_id}): Successfully parsed signal: {parsed_signal}")
        await self.telegram_interface.send_message(f"✅ **Signal Accepted**\n`{message['content']}`")

        # --- Build and Qualify the Contract ---
        try:
            contract = Option(
                symbol=parsed_signal['ticker'],
                lastTradeDateOrContractMonth=parsed_signal['expiry'],
                strike=parsed_signal['strike'],
                right=parsed_signal['option_type'],
                exchange='SMART',
                currency='USD'
            )
            # The scout: ask IBKR if this contract is valid.
            qualified_contracts = await self.ib_interface.ib.qualifyContractsAsync(contract)
            
            # --- SURGICAL FIX: The "Target Confirmation" Protocol ---
            # If the broker returns an empty list, the contract is a ghost. Abort mission.
            if not qualified_contracts:
                logger.error(f"Contract qualification FAILED for {parsed_signal} (ID: {msg_id}): No security definition found.")
                await self.telegram_interface.send_message(f"❌ **Trade Failed**\n`{parsed_signal['ticker']}`\nReason: Contract does not exist (check expiry/strike).")
                return

        except Exception as e:
            logger.error(f"Contract qualification FAILED for {parsed_signal} (ID: {msg_id}): {e}", exc_info=True)
            await self.telegram_interface.send_message(f"❌ **Trade Failed**\n`{parsed_signal['ticker']}`\nReason: `{e}`")
            return
            
        # --- If we reach here, the contract is confirmed to be valid. ---
        quantity = 1 # Placeholder for future dynamic sizing logic
        order = MarketOrder(action=parsed_signal['action'].upper(), totalQuantity=quantity)

        # --- Place the Trade ---
        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade,
                    "entry_time": datetime.now(timezone.utc),
                    "profile": profile,
                    "fill_processed": False,
                    "entry_price": None
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']}.")
                await self.telegram_interface.send_message(f"🚀 **Trade Placed**\n`{trade.order.action} {trade.order.totalQuantity} {trade.contract.localSymbol}`")
            else:
                logger.warning(f"Trade for {parsed_signal['ticker']} NOT PLACED: Likely due to an existing open order.")
                await self.telegram_interface.send_message(f"⚠️ **Trade Not Placed**\n`{parsed_signal['ticker']}`\nReason: Conflict with an existing open order.")
        except Exception as e:
            logger.error(f"Trade execution FAILED for {parsed_signal} (ID: {msg_id}): {e}", exc_info=True)
            await self.telegram_interface.send_message(f"❌ **Trade Failed**\n`{parsed_signal['ticker']}`\nReason: `{e}`")


    async def monitor_active_trades(self):
        """
        The "battle log." This function monitors all ongoing trades.
        """
        if not self.active_trades:
            return

        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info: continue
            
            trade = trade_info['trade_obj']
            contract = trade.contract
            
            if not trade_info['fill_processed']:
                if trade.orderStatus.status == 'Filled':
                    trade_info['fill_processed'] = True
                    trade_info['entry_price'] = trade.orderStatus.avgFillPrice
                    logger.info(f"Trade {trade_id} ({contract.localSymbol}) has been filled at ${trade_info['entry_price']:.2f}.")
                    await self.telegram_interface.send_message(f"✅ **Trade Filled**\n`{contract.localSymbol}`\n{trade.order.action} {trade.order.totalQuantity} @ ${trade_info['entry_price']:.2f}")
                elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                    logger.warning(f"Trade {trade_id} ({contract.localSymbol}) is no longer active. Status: {trade.orderStatus.status}")
                    del self.active_trades[trade_id]
                continue

            try:
                ticker = self.ib_interface.ib.reqMktData(contract, '', True, False)
                await asyncio.sleep(1)
                
                last_price = ticker.last
                if not last_price or last_price < 0:
                    logger.debug(f"No valid market price for {contract.localSymbol}. Waiting...")
                    continue

                entry_price = trade_info['entry_price']
                pnl_percent = ((last_price - entry_price) / entry_price) * 100 if entry_price else 0

                logger.info(
                    f"MONITORING {contract.localSymbol}: "
                    f"Entry=${entry_price:.2f}, "
                    f"Last=${last_price:.2f}, "
                    f"PnL={pnl_percent:.2f}%"
                )
                
                self.ib_interface.ib.cancelMktData(contract)

            except Exception as e:
                logger.error(f"Error monitoring trade {trade_id} ({contract.localSymbol}): {e}", exc_info=True)
                self.ib_interface.ib.cancelMktData(contract)

