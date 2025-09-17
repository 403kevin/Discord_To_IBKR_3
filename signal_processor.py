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
    The "brain" of the bot. This is the feature-complete "Professional Trader"
    version responsible for processing signals and managing the lifecycle of trades
    with a detailed battle log and a professional Telegram notification system.
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

        logger.info(f"Signal ACCEPTED (ID: {msg_id}): Parsed: {parsed_signal}")
        
        # --- Build Contract for Veto Message ---
        contract_str = (f"{parsed_signal['ticker']} "
                        f"{parsed_signal['expiry'][4:6]}/{parsed_signal['expiry'][6:8]}/{parsed_signal['expiry'][2:4]} "
                        f"{int(parsed_signal['strike'])}{parsed_signal['option_type']}")

        # --- Gatekeeper: Sentiment Analysis ---
        sentiment_score = 'N/A'
        if self.config.sentiment_filter['enabled']:
            sentiment_score = await self.sentiment_analyzer.analyze_sentiment(parsed_signal['ticker'])
            if sentiment_score is None or sentiment_score < self.config.sentiment_filter['sentiment_threshold']:
                reason = f"Sentiment score {sentiment_score} below threshold {self.config.sentiment_filter['sentiment_threshold']}"
                logger.warning(f"Trade for {contract_str} VETOED: {reason}")
                veto_msg = (f"❌ **Trade Vetoed** ❌\n"
                            f"Source Channel: `{profile['channel_name']}`\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Reason: `{reason}`")
                await self.telegram_interface.send_message(veto_msg)
                return

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
            
        quantity = 1 # Placeholder for future dynamic sizing logic
        order = MarketOrder(action=parsed_signal['action'].upper(), totalQuantity=quantity)

        # --- Place the Trade ---
        try:
            trade = await self.ib_interface.place_order(contract, order)
            if trade:
                trade_id = str(uuid.uuid4())
                self.active_trades[trade_id] = {
                    "trade_obj": trade,
                    "profile": profile,
                    "fill_processed": False,
                    "entry_price": None,
                    "sentiment_score": sentiment_score,
                    "high_water_mark": 0 # For trailing stop logic
                }
                logger.info(f"Successfully placed trade {trade_id} for {parsed_signal['ticker']}.")
            else:
                logger.warning(f"Trade for {contract_str} NOT PLACED: Likely due to a conflict with an existing open order.")
        except Exception as e:
            logger.error(f"Trade execution FAILED for {contract_str}: {e}", exc_info=True)


    async def monitor_active_trades(self):
        """
        The "battle log." This function monitors all ongoing trades, provides detailed
        logging on their status, and manages their lifecycle.
        """
        if not self.active_trades:
            return

        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info: continue
            
            trade = trade_info['trade_obj']
            contract = trade.contract
            
            # --- MONITOR UNTIL FILLED ---
            if not trade_info['fill_processed']:
                if trade.orderStatus.status == 'Filled':
                    trade_info['fill_processed'] = True
                    entry_price = trade.orderStatus.avgFillPrice
                    trade_info['entry_price'] = entry_price
                    trade_info['high_water_mark'] = entry_price # Initialize high-water mark
                    
                    profile = trade_info['profile']
                    exit_strategy = profile['exit_strategy']
                    momentum_exit = "NONE"
                    if exit_strategy.get('momentum_exits', {}).get('psar_enabled'): momentum_exit = "PSAR"
                    elif exit_strategy.get('momentum_exits', {}).get('rsi_hook_enabled'): momentum_exit = "RSI"
                    
                    contract_str = f"{contract.symbol} {contract.lastTradeDateOrContractMonth[4:6]}/{contract.lastTradeDateOrContractMonth[6:8]}/{contract.lastTradeDateOrContractMonth[2:4]} {int(contract.strike)}{contract.right}"

                    entry_msg = (f"✅ **Trade Entry Confirmed** ✅\n"
                                 f"Source Channel: `{profile['channel_name']}`\n"
                                 f"Contract details: `{contract_str}`\n"
                                 f"Quantity: `{int(trade.order.totalQuantity)}`\n"
                                 f"Entry Price: `${entry_price:.2f}`\n"
                                 f"Vader Sentiment Score: `{trade_info['sentiment_score']}`\n"
                                 f"Trail method: `{exit_strategy['trail_method']}`\n"
                                 f"Momentum Exit: `{momentum_exit}`")
                    await self.telegram_interface.send_message(entry_msg)
                    logger.info(f"Trade {trade_id} ({contract.localSymbol}) has been filled at ${entry_price:.2f}.")
                    
                elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                    logger.warning(f"Trade {trade_id} ({contract.localSymbol}) is no longer active. Status: {trade.orderStatus.status}")
                    del self.active_trades[trade_id]
                continue

            # --- MONITOR FILLED POSITIONS (The Battle Log) ---
            try:
                ticker = self.ib_interface.ib.reqMktData(contract, '', False, False)
                await asyncio.sleep(1.5) # Give a moment for data to arrive
                
                last_price = ticker.last if ticker.last and not ticker.last != ticker.last else ticker.close if ticker.close else 0
                
                if not last_price > 0:
                    logger.debug(f"No valid market price for {contract.localSymbol}. Waiting...")
                    self.ib_interface.ib.cancelMktData(contract)
                    continue

                # Update High-Water Mark
                trade_info['high_water_mark'] = max(trade_info['high_water_mark'], last_price)
                high_water_mark = trade_info['high_water_mark']
                
                # Freestyle Logic: Calculate a simple percentage-based trailing stop
                pullback_percent = trade_info['profile']['exit_strategy']['trail_settings']['pullback_percent']
                trailing_stop_price = high_water_mark * (1 - (pullback_percent / 100))
                
                entry_price = trade_info['entry_price']
                pnl_percent = ((last_price - entry_price) / entry_price) * 100 if entry_price else 0

                # This is the detailed logging you requested.
                logger.info(
                    f"MONITORING {contract.localSymbol}: "
                    f"Entry=${entry_price:.2f}, "
                    f"Last=${last_price:.2f}, "
                    f"PnL={pnl_percent:.2f}%, "
                    f"High-Water Mark=${high_water_mark:.2f}, "
                    f"Trail Stop=${trailing_stop_price:.2f}"
                )
                
                self.ib_interface.ib.cancelMktData(contract)

            except Exception as e:
                logger.error(f"Error monitoring trade {trade_id} ({contract.localSymbol}): {e}")
                self.ib_interface.ib.cancelMktData(contract)

