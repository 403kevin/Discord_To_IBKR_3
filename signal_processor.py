import logging
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from interfaces.telegram_interface import TelegramInterface
from services.message_parsers import SignalParser
from services.sentiment_analysis import SentimentAnalyzer
import asyncio
from ib_insync import Option, Trade, Order
from collections import deque

class SignalProcessor:
    """
    The central processing unit of the bot. It orchestrates the flow of data
    from Discord, through parsing and analysis, to the IBKR interface.
    This is the "Single Operator" that ensures sequential, safe operations.
    """

    def __init__(self, config: Config, ib_interface: IBInterface,
                 discord_interface: DiscordInterface, telegram_interface: TelegramInterface,
                 sentiment_analyzer: SentimentAnalyzer):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.telegram_interface = telegram_interface
        self.sentiment_analyzer = sentiment_analyzer
        self.parser = SignalParser(config)
        self.processed_message_ids = deque(maxlen=self.config.processed_message_cache_size)
        self.active_trades = {} # Stores trade objects by conId

    async def process_signal(self, message):
        """Processes a single Discord message."""
        message_id = message['id']
        if message_id in self.processed_message_ids:
            self.logger.debug(f"Skipping already processed message ID: {message_id}")
            return
        self.processed_message_ids.append(message_id)

        # Match message channel to a profile
        profile = next((p for p in self.config.profiles if p['channel_id'] == message['channel_id']), None)
        if not profile or not profile.get('enabled', False):
            return

        # Parse the message
        parsed_signal = self.parser.parse_signal_message(message['content'], profile)
        if not parsed_signal:
            return

        self.logger.info(f"Successfully parsed signal from {profile['channel_name']}: {parsed_signal}")

        # --- Sentiment Analysis Gate ---
        if self.config.sentiment_filter['enabled']:
            is_positive = await self.sentiment_analyzer.analyze_sentiment_for_ticker(parsed_signal['ticker'])
            if not is_positive:
                self.logger.warning(f"Trade for {parsed_signal['ticker']} halted due to negative sentiment.")
                await self.telegram_interface.send_message(
                    f"Sentiment Alert: Trade for {parsed_signal['ticker']} halted due to negative news."
                )
                return

        # --- Pre-Market Trading Logic ---
        # (This section would contain logic to adjust trade quantity based on pre-market rules)
        # For now, we assume regular hours.

        # --- Contract Creation ---
        contract = Option(
            parsed_signal['ticker'],
            parsed_signal['expiry'],
            parsed_signal['strike'],
            parsed_signal['option_type'],
            'SMART'
        )
        
        # Qualify contract to get conId
        qualified_contracts = await self.ib_interface.ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            self.logger.error(f"Could not qualify contract for signal: {parsed_signal}")
            return
        qualified_contract = qualified_contracts[0]

        # --- Order Sizing ---
        # (Simplified logic, a real implementation would be more complex)
        trade_value = profile['trading']['funds_allocation']
        # Fetch ticker price to estimate quantity
        ticker_data = self.ib_interface.ib.reqMktData(qualified_contract, '', False, False)
        await asyncio.sleep(2) # Allow time for data to arrive
        
        last_price = ticker_data.last
        if not last_price or last_price <= 0:
            self.logger.error(f"Could not get a valid market price for {qualified_contract.localSymbol}. Aborting trade.")
            self.ib_interface.ib.cancelMktData(qualified_contract)
            return
        
        self.ib_interface.ib.cancelMktData(qualified_contract) # Clean up
        
        quantity = int(trade_value / (last_price * 100)) # 100 shares per contract
        if quantity == 0:
            self.logger.warning(f"Calculated quantity is 0 for {qualified_contract.localSymbol}. Min contract price likely too high. Aborting.")
            return

        # --- Order Creation & Placement ---
        order = Order(
            action=parsed_signal['action'],
            orderType=profile['trading']['entry_order_type'],
            totalQuantity=quantity,
            tif=profile['trading']['time_in_force']
        )

        trade = self.ib_interface.place_order(qualified_contract, order)
        if trade:
            self.logger.info(f"Trade placed for {qualified_contract.localSymbol}. OrderId: {trade.order.orderId}")
            self.active_trades[qualified_contract.conId] = trade
            await self.telegram_interface.send_message(
                f"Trade Alert: Placed order for {quantity} contracts of {qualified_contract.localSymbol}."
            )

    async def monitor_active_trades(self):
        """
        The core of the "Single Operator" model. This loop runs sequentially
        and handles all post-trade management.
        """
        if not self.active_trades:
            return

        self.logger.debug(f"Monitoring {len(self.active_trades)} active trade(s).")
        
        # Create a copy of the keys to iterate over, as the dictionary may change
        con_ids_to_check = list(self.active_trades.keys())

        for conId in con_ids_to_check:
            trade = self.active_trades.get(conId)
            if not trade or not trade.isDone():
                continue # Skip trades not yet filled or already closed

            # --- Safety Net: Attach Native Trailing Stop ---
            # (This is a simplified placeholder. A full implementation is complex.)
            # We would check if a native trail has been attached already.
            # If not, create and place the trailing stop order here.
            
            # --- Dynamic Exit Logic ---
            # Example: Check for timeout exit
            # We would compare the trade's open time to the current time.
            # If it exceeds the profile's timeout, we would place a closing order.

            # Example: PSAR or RSI based exits
            # We would fetch historical data, calculate indicators, and decide to close.
            
            # If an exit condition is met:
            #   close_order = Order(...)
            #   self.ib_interface.place_order(trade.contract, close_order)
            #   del self.active_trades[conId] # Remove from active monitoring

    def get_active_trades(self):
        return self.active_trades
