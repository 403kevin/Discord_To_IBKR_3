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
    The "brain" of the bot. This is the "Professional Trader" edition with the
    critical Price Check Gatekeeper.
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

    # ... (get_cooldown_status and reset_consecutive_losses are unchanged) ...

    async def process_signal(self, message: dict, profile: dict):
        msg_id = message['id']
        if msg_id in self.processed_message_ids: return
        self.processed_message_ids.append(msg_id)
        
        # ... (Age check and parsing logic are unchanged) ...
        
        parser = SignalParser(self.config)
        parsed_signal = parser.parse_signal_message(message['content'], profile)
        if not parsed_signal:
            logger.debug(f"Signal IGNORED (ID: {msg_id}): Message content did not parse.")
            return

        logger.info(f"Signal ACCEPTED (ID: {msg_id}): Parsed: {parsed_signal}")
        
        contract_str = (f"{parsed_signal['ticker']} "
                        f"{parsed_signal['expiry'][4:6]}/{parsed_signal['expiry'][6:8]}/{parsed_signal['expiry'][2:4]} "
                        f"{int(parsed_signal['strike'])}{parsed_signal['option_type']}")

        # ... (Sentiment analysis logic is unchanged) ...
        
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
                # ... (Veto logic for non-existent contract is unchanged) ...
                return
        except Exception as e:
            # ... (Veto logic for qualification error is unchanged) ...
            return

        # --- SURGICAL UPGRADE: The Price Check Gatekeeper ---
        try:
            # 1. Fetch the live market data.
            ticker = self.ib_interface.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(1.5) # Wait for the price to arrive
            
            # Use the most reliable price available (last, then close)
            live_price = ticker.last if ticker.last and not ticker.last != ticker.last else ticker.close if ticker.close else 0
            self.ib_interface.ib.cancelMktData(contract) # Clean up the data stream

            if not live_price > 0:
                reason = "Could not fetch a valid live price for the contract."
                logger.error(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

            # 2. Enforce the constitution's price rules.
            min_price = profile['trading']['min_price_per_contract']
            max_price = profile['trading']['max_price_per_contract']

            if not (min_price <= live_price <= max_price):
                reason = f"Live price ${live_price:.2f} is outside the allowed range (${min_price:.2f} - ${max_price:.2f})."
                logger.warning(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

        except Exception as e:
            reason = f"Price check failed: {e}"
            logger.error(f"Trade for {contract_str} VETOED during price check: {reason}", exc_info=True)
            veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                        f"Source Channel: `{profile['channel_name']}`\n"
                        f"Contract details: `{contract_str}`\n"
                        f"Reason: `{reason}`")
            await self.telegram_interface.send_message(veto_msg)
            self.ib_interface.ib.cancelMktData(contract) # Ensure cleanup on error
            return
        # --- END UPGRADE ---
            
        quantity = 1
        order = MarketOrder(action=parsed_signal['action'].upper(), totalQuantity=quantity)

        # ... (Place order and monitor_active_trades logic are unchanged) ...