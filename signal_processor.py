import logging
import asyncio
import uuid
from datetime import datetime, timezone
from collections import deque
from ib_insync import Option, MarketOrder, Order, TrailOrder # <-- SURGICAL ADDITION
from services.signal_parser import SignalParser

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    The "brain" of the bot. This is the feature-complete "Professional Trader"
    version with the restored Native Trail safety reflex.
    """

    def __init__(self, config, ib_interface, discord_interface, sentiment_analyzer, telegram_interface):
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.sentiment_analyzer = sentiment_analyzer
        self.telegram_interface = telegram_interface
        
        self._channel_states = {}
        self.active_trades = {}
        self.processed_message_ids = deque(maxlen=self.config.processed_message_cache_size)

    # ... (get_cooldown_status and reset_consecutive_losses remain the same) ...

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
        # ... (This entire function remains the same as the last version) ...
        msg_id = message['id']
        if msg_id in self.processed_message_ids: return
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

        logger.info(f"Signal ACCEPTED (ID: {msg_id}): Parsed: {parsed_signal}")
        contract_str = (f"{parsed_signal['ticker']} "
                        f"{parsed_signal['expiry'][4:6]}/{parsed_signal['expiry'][6:8]}/{parsed_signal['expiry'][2:4]} "
                        f"{int(parsed_signal['strike'])}{parsed_signal['option_type']}")
        
        sentiment_score = 'N/A'
        # ... (Sentiment and Veto logic is unchanged) ...

        try:
            contract = Option(
                symbol=parsed_signal['ticker'],
                lastTradeDateOrContractMonth=parsed_signal['expiry'],
                strike=parsed_signal['strike'],
                right=parsed_signal['option_type'],
                exchange='SMART',
                currency='USD'
            )
            qualified_contracts = await self.ib_interface.ib.qualifyContractsAsync(contract)
            if not qualified_contracts:
                reason = "Contract does not exist (check expiry/strike)."
                # ... (Veto logic is unchanged) ...
                return
        except Exception as e:
            # ... (Veto logic is unchanged) ...
            return
            
        quantity = 1
        order = MarketOrder(action=parsed_signal['action'].upper(), totalQuantity=quantity)

        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade, "profile": profile, "fill_processed": False,
                    "entry_price": None, "sentiment_score": sentiment_score,
                    "high_water_mark": 0, "native_trail_attached": False # New flag
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']}.")
            else:
                logger.warning(f"Trade for {contract_str} NOT PLACED: Likely due to a conflict.")
        except Exception as e:
            logger.error(f"Trade execution FAILED for {contract_str}: {e}", exc_info=True)


    async def monitor_active_trades(self):
        """
        The "battle log." This is the upgraded version with the Native Trail safety reflex.
        """
        if not self.active_trades:
            return

        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info: continue
            
            trade = trade_info['trade_obj']
            contract = trade.contract
            profile = trade_info['profile']
            
            # --- MONITOR UNTIL FILLED ---
            if not trade_info['fill_processed']:
                if trade.orderStatus.status == 'Filled':
                    trade_info['fill_processed'] = True
                    entry_price = trade.orderStatus.avgFillPrice
                    trade_info['entry_price'] = entry_price
                    trade_info['high_water_mark'] = entry_price
                    
                    # ... (Telegram Entry Confirmation logic is unchanged) ...
                    
                    # --- SURGICAL UPGRADE: The "Safety Reflex" ---
                    if profile['safety_net']['enabled'] and not trade_info['native_trail_attached']:
                        # --- BATTLE-HARDENED PAUSE ---
                        # Wait a moment for the broker's systems to fully register the fill.
                        await asyncio.sleep(1) 
                        
                        try:
                            trail_percent = profile['safety_net']['native_trail_percent']
                            opposite_action = 'SELL' if trade.order.action == 'BUY' else 'BUY'
                            
                            trail_order = TrailOrder(
                                action=opposite_action,
                                totalQuantity=trade.order.totalQuantity,
                                trailingPercent=trail_percent,
                                tif='GTC' # Good Till Canceled
                            )
                            
                            trail_trade = await self.ib_interface.place_order(contract, trail_order)
                            if trail_trade:
                                trade_info['native_trail_attached'] = True
                                logger.info(f"Successfully attached {trail_percent}% native trail for {contract.localSymbol}.")
                                await self.telegram_interface.send_message(f"🛡️ **Safety Net Attached**\n`{contract.localSymbol}`\n{trail_percent}% Native Trail is active.")
                        except Exception as e:
                            logger.error(f"Failed to attach native trail for {contract.localSymbol}: {e}", exc_info=True)
                            await self.telegram_interface.send_message(f"🚨 **Native Trail FAILED**\n`{contract.localSymbol}`\nReason: `{e}`")

                elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                    # ... (Cancellation logic is unchanged) ...
                continue

            # --- MONITOR FILLED POSITIONS (The Battle Log) ---
            try:
                # ... (The "battle log" PnL monitoring logic is unchanged) ...
            except Exception as e:
                # ... (Error handling for monitoring is unchanged) ...

