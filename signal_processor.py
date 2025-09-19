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
    The "brain" of the bot. This is the "Professional Trader" edition with
    the critical Price Check and Capital Management Gatekeepers.
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

    def get_cooldown_status(self, channel_id):
        return self._channel_states.setdefault(str(channel_id), {
            'consecutive_losses': 0, 'on_cooldown': False, 'end_time': None
        })

    def reset_consecutive_losses(self, channel_id):
        state = self.get_cooldown_status(str(channel_id))
        state['consecutive_losses'] = 0
        state['on_cooldown'] = False
        state['end_time'] = None
        logger.info(f"Consecutive loss counter for channel {channel_id} has been reset.")

    async def process_signal(self, message: dict, profile: dict):
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
        if self.config.sentiment_filter['enabled']:
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(message['content'])
            if sentiment_score is None or sentiment_score < self.config.sentiment_filter['sentiment_threshold']:
                reason = f"Sentiment score {sentiment_score:.2f} below threshold {self.config.sentiment_filter['sentiment_threshold']}"
                logger.warning(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

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
                logger.error(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return
        except Exception as e:
            reason = str(e)
            logger.error(f"Trade for {contract_str} VETOED during qualification: {reason}", exc_info=True)
            veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                        f"Source Channel: `{profile['channel_name']}`\n"
                        f"Contract details: `{contract_str}`\n"
                        f"Reason: `Qualification Error: {reason}`")
            await self.telegram_interface.send_message(veto_msg)
            return

        try:
            ticker = self.ib_interface.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(1.5)
            
            live_price = ticker.last if ticker.last and not ticker.last != ticker.last else ticker.close if ticker.close else 0
            self.ib_interface.ib.cancelMktData(contract)

            if not live_price > 0:
                reason = "Could not fetch a valid live price."
                logger.error(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

            min_price = profile['trading']['min_price_per_contract']
            max_price = profile['trading']['max_price_per_contract']

            if not (min_price <= live_price <= max_price):
                reason = f"Live price ${live_price:.2f} is outside range (${min_price:.2f} - ${max_price:.2f})."
                logger.warning(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

            cost_per_contract = live_price * 100
            funds_allocation = profile['trading']['funds_allocation']
            
            if cost_per_contract <= 0:
                reason = "Contract price is zero or negative."
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

            quantity = int(funds_allocation / cost_per_contract)

            if quantity == 0:
                reason = f"Insufficient funds. 1 contract costs ${cost_per_contract:.2f}, allocation is ${funds_allocation}."
                logger.warning(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return
            
            logger.info(f"Sizing trade for {contract_str}: Allocation=${funds_allocation}, Price=${live_price:.2f}, Calculated Quantity={quantity}")
            
        except Exception as e:
            reason = f"Price check failed: {e}"
            logger.error(f"Trade for {contract_str} VETOED during price check: {reason}", exc_info=True)
            veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                        f"Source Channel: `{profile['channel_name']}`\n"
                        f"Contract details: `{contract_str}`\n"
                        f"Reason: `{reason}`")
            await self.telegram_interface.send_message(veto_msg)
            self.ib_interface.ib.cancelMktData(contract)
            return
            
        order = MarketOrder(
            action=parsed_signal['action'].upper(), 
            totalQuantity=quantity
        )
        
        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade, "profile": profile, "fill_processed": False,
                    "entry_price": None, "sentiment_score": sentiment_score,
                    "high_water_mark": 0, "native_trail_attached": False
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']}.")
            else:
                logger.warning(f"Trade for {contract_str} NOT PLACED: Likely due to a conflict.")
        except Exception as e:
            logger.error(f"Trade execution FAILED for {contract_str}: {e}", exc_info=True)


    async def monitor_active_trades(self):
        # ... (This function remains unchanged from the last correct version) ...
        pass

