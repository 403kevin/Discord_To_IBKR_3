import asyncio
import logging
from datetime import datetime, timedelta
from collections import deque
import json
import os

class SignalProcessor:
    """
    The orchestrator - manages the full lifecycle of signal processing.
    ENHANCED VERSION: Added comprehensive logging and per-channel buzzwords
    """
    def __init__(self, config, discord_interface, ib_interface, telegram_interface,
                 signal_parser, sentiment_analyzer, state_manager):
        self.config = config
        self.discord_interface = discord_interface
        self.ib_interface = ib_interface
        self.telegram_interface = telegram_interface
        self.signal_parser = signal_parser
        self.sentiment_analyzer = sentiment_analyzer
        self.state_manager = state_manager
        self.open_positions = {}
        self._shutdown_event = asyncio.Event()
        self._processed_messages = {}
        self._last_trade_time = None
        self._startup_time = datetime.now()
        
        # Initialize message tracking for each channel
        for profile in self.config.channel_profiles:
            if profile['enabled']:
                channel_id = profile['channel_id']
                self._processed_messages[channel_id] = deque(maxlen=1000)
                
        logging.info(f"SignalProcessor initialized with {len(self._processed_messages)} active channels")

    async def start(self):
        """Main entry point - starts all processing tasks."""
        logging.info("Starting Signal Processor...")
        
        # Load existing state
        self.open_positions = await self.state_manager.load_state()
        logging.info(f"Loaded {len(self.open_positions)} open positions from state")
        
        # Perform initial reconciliation
        await self._initial_reconciliation()
        
        # Start all async tasks
        tasks = [
            self._poll_discord_for_signals(),
            self._monitor_open_positions(),
            self._reconciliation_loop()
        ]
        
        await asyncio.gather(*tasks)

    async def _poll_discord_for_signals(self):
        """Continuously polls Discord channels for new messages."""
        while not self._shutdown_event.is_set():
            for profile in self.config.channel_profiles:
                if not profile['enabled']:
                    continue
                    
                channel_id = profile['channel_id']
                channel_name = profile.get('channel_name', channel_id)
                
                try:
                    raw_messages = await self.discord_interface.poll_for_new_messages(
                        channel_id, self._processed_messages[channel_id]
                    )
                    
                    if raw_messages:
                        await self._process_new_signals(raw_messages, profile)
                        
                except Exception as e:
                    logging.error(f"Error polling channel {channel_name}: {e}")
                    
            await asyncio.sleep(self.config.discord_poll_interval_seconds)

    async def _process_new_signals(self, raw_messages, profile):
        """Processes raw Discord messages into trade signals with enhanced logging."""
        channel_name = profile.get('channel_name', profile['channel_id'])
        
        for msg in raw_messages:
            try:
                msg_id = msg.get('id')
                msg_content = msg.get('content', '')
                msg_author = msg.get('author', {}).get('username', 'Unknown')
                msg_timestamp = msg.get('timestamp')
                
                # Check if message is too old
                if msg_timestamp:
                    msg_time = datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00'))
                    if msg_time < self._startup_time:
                        logging.info(f"Ignoring stale message {msg_id} (timestamp before bot start)")
                        self._processed_messages[profile['channel_id']].append(msg_id)
                        continue
                
                logging.info(f"Processing message {msg_id} from '{msg_author}'")
                logging.info(f"Raw content: '{msg_content[:200]}{'...' if len(msg_content) > 200 else ''}'")
                
                # Skip if already processed
                if msg_id in self._processed_messages[profile['channel_id']]:
                    logging.debug(f"Message {msg_id} already processed, skipping")
                    continue
                
                # Mark as processed immediately
                self._processed_messages[profile['channel_id']].append(msg_id)
                
                # Check for channel-specific ignore keywords
                channel_ignore = profile.get('buzzwords_ignore', [])
                global_ignore = self.config.buzzwords_ignore
                all_ignore_words = set(channel_ignore + global_ignore)
                
                if self._contains_keywords(msg_content, all_ignore_words):
                    ignored_word = self._get_matched_keyword(msg_content, all_ignore_words)
                    logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Ignore keyword '{ignored_word}' | Message: '{msg_content[:100]}'")
                    continue
                
                # Parse the signal with channel-specific buzzwords
                parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
                
                if not parsed_signal:
                    logging.info(f"⚠️ NOT PARSED - Channel: {channel_name} | Reason: Invalid format | Message: '{msg_content[:100]}'")
                    continue
                
                logging.info(f"✓ PARSED - Ticker: {parsed_signal['ticker']} | Strike: {parsed_signal['strike']} | Type: {parsed_signal['contract_type']} | Expiry: {parsed_signal['expiry_date']}")
                
                # Check sentiment if enabled
                if self.config.sentiment_filter_enabled:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(msg_content)
                    sentiment_score = sentiment['compound']
                    
                    if sentiment_score < profile.get('sentiment_threshold', 0.0):
                        logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Low sentiment ({sentiment_score:.2f}) | Signal: {parsed_signal['ticker']}")
                        
                        # Send Telegram notification for sentiment veto
                        veto_msg = (
                            f"🚫 *Trade Vetoed by Sentiment*\n\n"
                            f"Channel: {channel_name}\n"
                            f"Signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type']}\n"
                            f"Sentiment Score: {sentiment_score:.2f}\n"
                            f"Threshold: {profile.get('sentiment_threshold', 0.0):.2f}\n"
                            f"Message: _{msg_content[:100]}_"
                        )
                        await self.telegram_interface.send_message(veto_msg)
                        continue
                    
                    parsed_signal['sentiment'] = sentiment
                else:
                    parsed_signal['sentiment'] = None
                
                # Check global cooldown
                if self._last_trade_time:
                    time_since_last = (datetime.now() - self._last_trade_time).total_seconds()
                    if time_since_last < self.config.cooldown_after_trade_seconds:
                        logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Cooldown ({time_since_last:.0f}s < {self.config.cooldown_after_trade_seconds}s) | Signal: {parsed_signal['ticker']}")
                        continue
                
                # Execute the trade
                logging.info(f"🎯 EXECUTING - Channel: {channel_name} | Signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type']}")
                await self._execute_trade(parsed_signal, profile)
                
                # Save state
                await self.state_manager.save_state(self.open_positions)
                
            except Exception as e:
                logging.error(f"Error processing message {msg_id}: {e}", exc_info=True)

    def _contains_keywords(self, text, keywords):
        """Check if text contains any of the keywords."""
        text_upper = text.upper()
        return any(keyword.upper() in text_upper for keyword in keywords)
    
    def _get_matched_keyword(self, text, keywords):
        """Get the first matched keyword."""
        text_upper = text.upper()
        for keyword in keywords:
            if keyword.upper() in text_upper:
                return keyword
        return None

    async def _execute_trade(self, signal, profile):
        """Executes a parsed trade signal."""
        try:
            # Create contract
            contract_details = await self.ib_interface.create_option_contract(
                signal['ticker'],
                signal['expiry_date'],
                signal['strike'],
                signal['contract_type']
            )
            
            if not contract_details:
                logging.error(f"Failed to create contract for {signal}")
                return
            
            contract, market_price = contract_details
            
            # Calculate position size
            allocation = profile['trading']['funds_allocation']
            quantity = self._calculate_position_size(allocation, market_price)
            
            if quantity <= 0:
                logging.info(f"Position size is 0 for {signal['ticker']} at ${market_price}")
                return
            
            # Place the order
            trade = await self.ib_interface.place_order(
                contract, quantity, profile['trading']['entry_order_type']
            )
            
            if trade:
                # Store position info
                position_info = {
                    'contract': contract,
                    'quantity': quantity,
                    'entry_price': market_price,
                    'entry_time': datetime.now(),
                    'profile': profile,
                    'signal': signal,
                    'trade': trade
                }
                
                self.open_positions[contract.conId] = position_info
                self._last_trade_time = datetime.now()
                
                # Send notifications
                await self._send_trade_notification(position_info, 'ENTRY')
                
                logging.info(f"✅ TRADE OPENED - {signal['ticker']} {quantity}x @ ${market_price}")
                
        except Exception as e:
            logging.error(f"Error executing trade: {e}", exc_info=True)

    def _calculate_position_size(self, allocation, price):
        """Calculate the number of contracts based on allocation and price."""
        if price <= 0:
            return 0
        contract_cost = price * 100
        return int(allocation / contract_cost)

    async def _monitor_open_positions(self):
        """Monitor positions and execute exit strategies."""
        while not self._shutdown_event.is_set():
            for conId, position in list(self.open_positions.items()):
                try:
                    # Get current price
                    current_data = await self.ib_interface.get_market_data(position['contract'])
                    
                    if current_data and current_data.last > 0:
                        # Evaluate exit conditions
                        exit_reason = await self._evaluate_exit_conditions(
                            position, current_data.last
                        )
                        
                        if exit_reason:
                            await self._execute_close_trade(conId, exit_reason)
                            
                except Exception as e:
                    logging.error(f"Error monitoring position {conId}: {e}")
                    
            await asyncio.sleep(5)

    async def _evaluate_exit_conditions(self, position, current_price):
        """Evaluate all exit strategies for a position."""
        profile = position['profile']
        exit_config = profile['exit_strategy']
        entry_price = position['entry_price']
        
        # Check breakeven
        if exit_config.get('breakeven_trigger_percent'):
            trigger = exit_config['breakeven_trigger_percent'] / 100
            if current_price >= entry_price * (1 + trigger):
                if current_price <= entry_price:
                    return "Breakeven Stop"
        
        # Add other exit logic here (ATR, RSI, PSAR, etc.)
        
        return None

    async def _execute_close_trade(self, conId, reason):
        """Close a position."""
        try:
            position = self.open_positions[conId]
            
            # Cancel any open orders first
            await self.ib_interface.cancel_all_orders_for_contract(conId)
            
            # Place sell order
            trade = await self.ib_interface.place_order(
                position['contract'],
                position['quantity'],
                'MKT',
                action='SELL'
            )
            
            if trade:
                # Remove from tracking
                del self.open_positions[conId]
                
                # Send notification
                await self._send_trade_notification(position, 'EXIT', reason)
                
                logging.info(f"✅ TRADE CLOSED - {position['signal']['ticker']} | Reason: {reason}")
                
        except Exception as e:
            logging.error(f"Error closing position {conId}: {e}")

    async def _send_trade_notification(self, position_info, status, exit_reason=None):
        """Send formatted Telegram notification."""
        try:
            signal = position_info['signal']
            profile = position_info['profile']
            
            if status == 'ENTRY':
                message = (
                    f"✅ *Trade Entry Confirmed* ✅\n\n"
                    f"*Source Channel:* {profile.get('channel_name', 'Unknown')}\n"
                    f"*Contract:* {signal['ticker']} {signal['expiry_date']} {signal['strike']}{signal['contract_type']}\n"
                    f"*Quantity:* {position_info['quantity']}\n"
                    f"*Entry Price:* ${position_info['entry_price']:.2f}\n"
                )
            else:  # EXIT
                message = (
                    f"🔴 *SELL Order Executed*\n\n"
                    f"*Contract:* {signal['ticker']} {signal['expiry_date']} {signal['strike']}{signal['contract_type']}\n"
                    f"*Exit Reason:* {exit_reason}\n"
                )
            
            await self.telegram_interface.send_message(message)
            
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {e}")

    async def _reconciliation_loop(self):
        """Periodic reconciliation with broker."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)  # Run every 60 seconds
            
            try:
                logging.info("--- Starting periodic position reconciliation ---")
                
                # Get broker positions
                ib_positions = await self.ib_interface.get_open_positions()
                
                # Skip zero positions
                active_ib_positions = [p for p in ib_positions if p.position != 0]
                
                logging.info(f"Reconciliation: Found {len(active_ib_positions)} active positions at broker")
                
                # Check for untracked positions
                for ib_pos in active_ib_positions:
                    if ib_pos.contract.conId not in self.open_positions:
                        logging.warning(f"Found untracked position: {ib_pos.contract.localSymbol}")
                        # Optionally adopt or close it
                
                logging.info("--- Position reconciliation complete ---")
                
            except Exception as e:
                logging.error(f"Error during reconciliation: {e}")

    async def _initial_reconciliation(self):
        """Initial reconciliation on startup."""
        logging.info("Performing initial state reconciliation with broker...")
        
        try:
            ib_positions = await self.ib_interface.get_open_positions()
            active_positions = [p for p in ib_positions if p.position != 0]
            
            logging.info(f"Reconciliation: Found {len(active_positions)} active positions at broker")
            
            # Reconcile with saved state
            for ib_pos in active_positions:
                if ib_pos.contract.conId not in self.open_positions:
                    logging.warning(f"Untracked position found: {ib_pos.contract.localSymbol}")
            
            logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")
            
        except Exception as e:
            logging.error(f"Error during initial reconciliation: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        logging.info("Shutting down SignalProcessor...")
        self._shutdown_event.set()
        await self.state_manager.save_state(self.open_positions)