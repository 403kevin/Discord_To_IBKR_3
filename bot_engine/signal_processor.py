"""
Signal Processor - The Core Trading Engine
COMPLETE VERSION with ghost position fix applied
"""
import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional

from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from interfaces.telegram_interface import TelegramInterface
from services.signal_parser import SignalParser
from services.state_manager import StateManager
from services.exit_strategy_manager import ExitStrategyManager
from services.sentiment_analyzer import SentimentAnalyzer


class SignalProcessor:
    """
    The central brain of the trading bot. Coordinates Discord polling,
    signal parsing, IBKR execution, exit strategies, and state management.
    """

    def __init__(self, config):
        self.config = config
        
        # Core components
        self.ib_interface = IBInterface(config)
        self.discord_interface = DiscordInterface(config)
        self.telegram_interface = TelegramInterface(config)
        self.signal_parser = SignalParser(config)
        self.state_manager = StateManager(config)
        self.exit_manager = ExitStrategyManager(config, self.ib_interface, self.telegram_interface)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        
        # Trading state
        self.open_positions = {}  # {conId: position_data}
        self._processed_messages = defaultdict(lambda: deque(maxlen=config.processed_message_cache_size))
        self._shutdown_event = asyncio.Event()
        self._bot_start_time = datetime.now(pytz.UTC)
        self._last_trade_time = None
        
        logging.info("SignalProcessor initialized with all components")

    async def initialize(self):
        """Initialize all components and restore state."""
        logging.info("Initializing SignalProcessor components...")
        
        # Initialize interfaces
        await self.discord_interface.initialize()
        await self.telegram_interface.initialize()
        await self.ib_interface.connect()
        
        # Set callback for order fills
        self.ib_interface.set_order_filled_callback(self._handle_order_filled)
        
        # Restore previous state
        restored_positions, restored_messages = self.state_manager.restore_state()
        
        if restored_positions:
            self.open_positions = restored_positions
            logging.info(f"Restored {len(self.open_positions)} open positions from state file")
        
        if restored_messages:
            for msg_id in restored_messages:
                # Extract channel from message or use first channel as default
                channel_id = self.config.profiles[0]['channel_id'] if self.config.profiles else None
                if channel_id:
                    self._processed_messages[channel_id].append(msg_id)
            logging.info(f"Restored {len(restored_messages)} processed message IDs")
        
        # Perform initial reconciliation with broker
        await self._initial_reconciliation()
        
        # Warm up message cache
        await self._warmup_message_cache()
        
        logging.info("✅ SignalProcessor initialization complete")

    async def _initial_reconciliation(self):
        """
        Sync our tracked positions with broker's actual positions on startup.
        Identifies ghost positions (at broker but not tracked) and phantom positions.
        """
        try:
            logging.info("Performing initial state reconciliation with broker...")
            broker_positions = await self.ib_interface.get_open_positions()
            
            logging.info(f"Reconciliation: Found {len(broker_positions)} active positions at broker")
            
            # Check for positions at broker that we're not tracking
            broker_conIds = {pos.contract.conId for pos in broker_positions}
            tracked_conIds = set(self.open_positions.keys())
            
            # Positions at broker that we're not tracking
            ghost_positions = broker_conIds - tracked_conIds
            if ghost_positions:
                logging.warning(f"Found {len(ghost_positions)} ghost positions at broker: {ghost_positions}")
                # Could add logic to close these or add to tracking
            
            # Positions we're tracking that broker doesn't have
            phantom_positions = tracked_conIds - broker_conIds
            if phantom_positions:
                logging.warning(f"Found {len(phantom_positions)} phantom positions in tracker: {phantom_positions}")
                for conId in phantom_positions:
                    del self.open_positions[conId]
                    logging.info(f"Removed phantom position {conId} from tracking")
            
            # Save reconciled state
            all_processed_ids = []
            for channel_deque in self._processed_messages.values():
                all_processed_ids.extend(list(channel_deque))
            
            self.state_manager.save_state(self.open_positions, all_processed_ids)
            
            logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")
            
        except Exception as e:
            logging.error(f"Error during initial reconciliation: {e}")

    async def _warmup_message_cache(self):
        """
        WARMUP PHASE: Silently mark all existing Discord messages as processed
        to prevent log spam on startup. Only NEW messages after this point will be processed.
        """
        logging.info("🔥 Warming up message cache (marking existing messages as processed)...")
        
        warmup_count = 0
        for profile in self.config.profiles:
            if not profile['enabled']:
                continue
            
            channel_id = profile['channel_id']
            channel_name = profile.get('channel_name', 'Unknown')
            
            try:
                # Get last 50 messages from channel
                processed_ids = list(self._processed_messages[channel_id])
                raw_messages = await self.discord_interface.poll_for_new_messages(channel_id, processed_ids)
                
                # Silently mark all as processed
                for message_data in raw_messages:
                    if isinstance(message_data, tuple):
                        msg_id = message_data[0]
                    else:
                        msg_id = message_data.get('id')
                    
                    if msg_id not in self._processed_messages[channel_id]:
                        self._processed_messages[channel_id].append(msg_id)
                        warmup_count += 1
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logging.warning(f"Could not warmup channel '{channel_name}': {e}")
        
        logging.info(f"✅ Warmup complete: Marked {warmup_count} existing messages as processed")
        logging.info("🎯 Bot is now live and monitoring for NEW signals only")

    async def _poll_discord_for_signals(self):
        """Polls Discord for new signals from all enabled channels."""
        while not self._shutdown_event.is_set():
            try:
                for profile in self.config.profiles:
                    if not profile['enabled']:
                        continue
                    
                    channel_id = profile['channel_id']
                    channel_name = profile.get('channel_name', 'Unknown')
                    
                    # Poll for new messages
                    processed_ids = list(self._processed_messages[channel_id])
                    raw_messages = await self.discord_interface.poll_for_new_messages(channel_id, processed_ids)
                    
                    if raw_messages:
                        await self._process_new_signals(raw_messages, profile)
                    
                    # Small delay between channels
                    await asyncio.sleep(self.config.delay_between_channels)
                
                # Delay after full cycle
                await asyncio.sleep(self.config.delay_after_full_cycle)
                
            except Exception as e:
                logging.error(f"Error polling Discord: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_new_signals(self, raw_messages, profile):
        """Process raw messages looking for trading signals."""
        channel_id = profile['channel_id']
        channel_name = profile.get('channel_name', 'Unknown')
        channel_ignore = profile.get('buzzwords_ignore', [])
        
        for message_data in raw_messages:
            msg_id = None
            try:
                # Handle both tuple and dict formats
                if isinstance(message_data, tuple):
                    msg_id, msg_content, timestamp_str = message_data
                else:
                    msg_id = message_data.get('id')
                    msg_content = message_data.get('content', '')
                    timestamp_str = message_data.get('timestamp', '')
                
                # Skip if already processed
                if msg_id in self._processed_messages[channel_id]:
                    continue
                
                self._processed_messages[channel_id].append(msg_id)
                
                logging.debug(f"Processing message {msg_id} from '{channel_name}'")
                logging.debug(f"Raw content: '{msg_content}'")
                
                # Check timestamp
                try:
                    if isinstance(timestamp_str, str):
                        msg_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        msg_time = timestamp_str
                    
                    # Check if message is from before bot started
                    if msg_time < self._bot_start_time:
                        logging.debug(f"Ignoring stale message {msg_id} (timestamp before bot start)")
                        continue
                        
                    # Check if message is too old (>60 seconds)
                    age = (datetime.now(pytz.UTC) - msg_time).total_seconds()
                    if age > self.config.signal_max_age_seconds:
                        logging.info(f"⏰ Signal too old ({age:.0f}s) from '{channel_name}': {msg_content[:50]}...")
                        continue
                        
                except Exception as e:
                    logging.warning(f"Could not parse timestamp for message {msg_id}: {e}")
                    continue
                
                # Check global cooldown
                if self._last_trade_time:
                    cooldown_remaining = self.config.cooldown_after_trade_seconds - \
                                       (datetime.now() - self._last_trade_time).total_seconds()
                    if cooldown_remaining > 0:
                        logging.info(f"⏸️ Global cooldown active ({cooldown_remaining:.0f}s remaining), skipping signal")
                        continue
                
                # Check channel-specific ignore words
                if channel_ignore and any(word.upper() in msg_content.upper() for word in channel_ignore):
                    logging.info(f"🚫 Message contains channel ignore word, skipping: {msg_content[:50]}...")
                    continue
                
                # Parse the signal
                parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
                
                if not parsed_signal:
                    logging.info(f"⚠️ SIGNAL NOT PARSED - Channel: {channel_name} | Message: '{msg_content[:100]}'")
                    logging.info(f"  Reason: Parser could not extract ticker, strike, type, or expiry")
                    logging.info(f"  Profile settings: assume_buy={profile.get('assume_buy_on_ambiguous')}, ambiguous_expiry={profile.get('ambiguous_expiry_enabled')}")
                    continue
                
                # Add metadata
                parsed_signal['channel'] = channel_name
                parsed_signal['message_id'] = msg_id
                parsed_signal['timestamp'] = msg_time
                parsed_signal['raw_message'] = msg_content
                parsed_signal['profile'] = profile
                
                # Check sentiment if enabled
                if profile.get('sentiment_filter', {}).get('enabled', False):
                    sentiment_passed, sentiment_score = self.sentiment_analyzer.check_sentiment(
                        msg_content, 
                        parsed_signal['action'],
                        profile['sentiment_filter']
                    )
                    
                    if not sentiment_passed:
                        logging.info(f"😐 Signal vetoed by sentiment filter (score: {sentiment_score:.2f})")
                        
                        # Send veto notification if enabled
                        if self.config.telegram_notifications.get('veto', {}).get('enabled', False):
                            await self._send_veto_notification(parsed_signal, sentiment_score)
                        continue
                    
                    parsed_signal['sentiment_score'] = sentiment_score
                else:
                    parsed_signal['sentiment_score'] = 'N/A'
                
                # Execute the trade
                await self._execute_signal(parsed_signal, profile)
                
            except Exception as e:
                logging.error(f"Error processing message {msg_id}: {e}", exc_info=True)

    async def _execute_signal(self, signal: Dict[str, Any], profile: Dict[str, Any]):
        """Execute a parsed trading signal."""
        try:
            logging.info(f"🎯 EXECUTING SIGNAL: {signal['ticker']} {signal['strike']}{signal['contract_type'][0]} exp:{signal['expiry']}")
            
            # Create the option contract
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'],
                signal['expiry'],
                signal['strike'],
                signal['contract_type'][0]  # 'C' or 'P'
            )
            
            if not contract:
                logging.error(f"Failed to create contract for signal: {signal}")
                return
            
            # Determine quantity (default 1 if not specified)
            quantity = signal.get('quantity', 1)
            
            # Add profile to trade for callback
            signal['profile'] = profile
            signal['channel'] = profile.get('channel_name', 'Unknown')
            
            # Place the order with signal metadata
            order = await self.ib_interface.place_order(
                contract,
                'MKT',  # Market order
                quantity,
                signal['action'],  # 'BUY' or 'SELL'
                signal  # Pass the entire signal for metadata
            )
            
            if not order:
                logging.error(f"Failed to place order for signal: {signal}")
                return
            
            # Update last trade time for cooldown
            self._last_trade_time = datetime.now()
            
            logging.info(f"✅ Order placed successfully for {signal['ticker']}")
            
        except Exception as e:
            logging.error(f"Error executing signal: {e}", exc_info=True)

    async def _handle_order_filled(self, trade):
        """Called when an order is filled by IBKR."""
        try:
            contract = trade.contract
            fill = trade.orderStatus
            avg_price = fill.avgFillPrice
            filled_qty = fill.filled
            
            logging.info(f"📈 ORDER FILLED: {contract.localSymbol} @ ${avg_price:.2f} x {filled_qty}")
            
            # Get metadata from trade if available
            signal_data = getattr(trade, 'signal_data', {})
            channel = signal_data.get('channel', 'Unknown')
            profile = signal_data.get('profile', {})
            
            # Determine if this is an entry or exit
            is_entry = trade.order.action == 'BUY'
            
            if is_entry:
                # Track new position
                position_data = {
                    'contract': contract,
                    'entry_price': avg_price,
                    'quantity': filled_qty,
                    'entry_time': datetime.now(),
                    'channel': channel,
                    'profile': profile,
                    'trade': trade,
                    'signal_data': signal_data
                }
                
                self.open_positions[contract.conId] = position_data
                
                # Set up exit strategy
                exit_params = profile.get('exit_strategy', {})
                if exit_params.get('enabled', True):
                    await self.exit_manager.setup_exit_strategy(position_data, exit_params)
                
                # Send entry notification
                if self.config.telegram_notifications.get('entry', {}).get('enabled', False):
                    await self._send_entry_notification(position_data, signal_data)
                    
            else:
                # Position closed
                if contract.conId in self.open_positions:
                    position_data = self.open_positions[contract.conId]
                    exit_price = avg_price
                    pnl = (exit_price - position_data['entry_price']) * filled_qty * 100  # x100 for options
                    
                    # Send exit notification
                    if self.config.telegram_notifications.get('exit', {}).get('enabled', False):
                        await self._send_exit_notification(position_data, exit_price, pnl, "Manual")
                    
                    # Clean up
                    del self.open_positions[contract.conId]
                    logging.info(f"Position closed for {contract.localSymbol}, P&L: ${pnl:.2f}")
            
            # Save updated state
            all_processed_ids = []
            for channel_deque in self._processed_messages.values():
                all_processed_ids.extend(list(channel_deque))
            self.state_manager.save_state(self.open_positions, all_processed_ids)
            
        except Exception as e:
            logging.error(f"Error handling order fill: {e}", exc_info=True)

    async def _send_entry_notification(self, position_data, signal_data):
        """Send formatted entry notification to Telegram."""
        try:
            contract = position_data['contract']
            sentiment = signal_data.get('sentiment_score', 'N/A')
            if isinstance(sentiment, float):
                sentiment = f"{sentiment:.2f}"
            
            msg = f"""
🟢 *ENTRY FILLED*

*Symbol:* {contract.symbol}
*Strike:* ${contract.strike}
*Type:* {'CALL' if contract.right == 'C' else 'PUT'}
*Expiry:* {contract.lastTradeDateOrContractMonth}
*Quantity:* {position_data['quantity']}
*Fill Price:* ${position_data['entry_price']:.2f}
*Channel:* {position_data['channel']}
*Sentiment:* {sentiment}
"""
            await self.telegram_interface.send_message(msg.strip())
        except Exception as e:
            logging.error(f"Failed to send entry notification: {e}")

    async def _send_exit_notification(self, position_data, exit_price, pnl, exit_reason="Manual"):
        """Send formatted exit notification to Telegram."""
        try:
            contract = position_data['contract']
            hold_time = (datetime.now() - position_data['entry_time']).total_seconds() / 60
            
            msg = f"""
🔴 *EXIT FILLED*

*Symbol:* {contract.symbol}
*Strike:* ${contract.strike}
*Type:* {'CALL' if contract.right == 'C' else 'PUT'}
*Entry:* ${position_data['entry_price']:.2f}
*Exit:* ${exit_price:.2f}
*P&L:* ${pnl:+.2f} ({(pnl/(position_data['entry_price']*position_data['quantity']*100)*100):+.1f}%)
*Hold Time:* {hold_time:.1f} min
*Exit Reason:* {exit_reason}
"""
            await self.telegram_interface.send_message(msg.strip())
        except Exception as e:
            logging.error(f"Failed to send exit notification: {e}")

    async def _send_veto_notification(self, signal, sentiment_score):
        """Send notification when signal is vetoed by sentiment."""
        try:
            msg = f"""
⛔ *SIGNAL VETOED*

*Reason:* Sentiment Filter
*Score:* {sentiment_score:.2f}
*Symbol:* {signal['ticker']}
*Strike:* ${signal['strike']}
*Type:* {signal['contract_type']}
*Channel:* {signal['channel']}
"""
            await self.telegram_interface.send_message(msg.strip())
        except Exception as e:
            logging.error(f"Failed to send veto notification: {e}")

    async def _monitor_open_positions(self):
        """Monitor open positions for exit conditions."""
        while not self._shutdown_event.is_set():
            try:
                if self.open_positions:
                    for conId, position_data in list(self.open_positions.items()):
                        # Exit manager handles the actual exit logic
                        pass  # The exit manager has its own monitoring loop
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring positions: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _reconciliation_loop(self):
        """
        Periodic reconciliation with broker to catch ghost positions.
        FIXED: Now properly sets exchange for options contracts.
        """
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.reconciliation_interval_seconds)
            
            try:
                logging.info("Starting periodic position reconciliation...")
                broker_positions = await self.ib_interface.get_open_positions()
                
                # Build sets for comparison
                broker_conIds = {pos.contract.conId for pos in broker_positions}
                tracked_conIds = set(self.open_positions.keys())
                
                # Ghost positions = at broker but not tracked by us
                ghost_positions = broker_conIds - tracked_conIds
                
                if ghost_positions:
                    logging.warning(f"🚨 GHOST ALERT: Found {len(ghost_positions)} untracked positions at broker")
                    
                    # Force-close each ghost position
                    for ghost_conId in ghost_positions:
                        # Find the position object from broker
                        ghost_pos = None
                        for pos in broker_positions:
                            if pos.contract.conId == ghost_conId:
                                ghost_pos = pos
                                break
                        
                        if ghost_pos is None:
                            logging.error(f"Could not find ghost position {ghost_conId} in broker list")
                            continue
                        
                        try:
                            quantity = abs(ghost_pos.position)
                            action = 'SELL' if ghost_pos.position > 0 else 'BUY'
                            
                            # FIX: Properly handle contract exchange like close_all_positions.py
                            if ghost_pos.contract.secType == 'OPT':
                                # Request contract details to get proper exchange
                                logging.info(f"Requesting contract details for ghost option {ghost_pos.contract.localSymbol}")
                                contract_details = await self.ib_interface.ib.reqContractDetailsAsync(ghost_pos.contract)
                                
                                if contract_details:
                                    # Use the contract from details which has proper exchange
                                    contract = contract_details[0].contract
                                    logging.info(f"Got exchange '{contract.exchange}' for {contract.localSymbol}")
                                else:
                                    logging.error(f"Could not get contract details for ghost {ghost_conId}")
                                    continue
                            else:
                                # For stocks, use SMART exchange
                                contract = ghost_pos.contract
                                contract.exchange = 'SMART'
                            
                            logging.warning(f"🔨 FORCE-CLOSING GHOST: {action} {quantity} of {contract.localSymbol} (conId: {ghost_conId})")
                            
                            # Cancel any existing orders first
                            await self.ib_interface.cancel_all_orders_for_contract(contract)
                            
                            # Place market order to force close
                            order = await self.ib_interface.place_order(
                                contract,
                                'MKT',
                                quantity,
                                action
                            )
                            
                            if order:
                                logging.info(f"✅ Ghost position closed: {contract.localSymbol}")
                                
                                # Send Telegram alert
                                try:
                                    msg = f"""
🔨 *GHOST POSITION CLOSED*

*Contract:* {contract.localSymbol}
*Quantity:* {quantity}
*Action:* {action}
*Reason:* Untracked position detected during reconciliation
"""
                                    await self.telegram_interface.send_message(msg.strip())
                                except:
                                    pass  # Don't let Telegram failure stop reconciliation
                            else:
                                logging.error(f"Failed to place close order for ghost {contract.localSymbol}")
                                
                        except Exception as e:
                            logging.error(f"Error closing ghost position {ghost_conId}: {e}", exc_info=True)
                    
                    # Give time for orders to fill
                    await asyncio.sleep(2)
                
                # Phantom positions = we're tracking but broker doesn't have
                phantom_positions = tracked_conIds - broker_conIds
                if phantom_positions:
                    for conId in phantom_positions:
                        logging.warning(f"Removing phantom position {conId} from tracking")
                        del self.open_positions[conId]
                
                # Save reconciled state
                all_processed_ids = []
                for channel_deque in self._processed_messages.values():
                    all_processed_ids.extend(list(channel_deque))
                
                self.state_manager.save_state(self.open_positions, all_processed_ids)
                
                # Log summary
                if not ghost_positions and not phantom_positions:
                    logging.info(f"✅ Reconciliation OK: {len(self.open_positions)} positions verified")
                    
            except Exception as e:
                logging.error(f"Error during reconciliation: {e}", exc_info=True)

    async def _eod_close_monitor(self):
        """Monitor for end-of-day position closing."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(pytz.timezone(self.config.MARKET_TIMEZONE))
                eod_time = now.replace(
                    hour=self.config.eod_close['hour'],
                    minute=self.config.eod_close['minute'],
                    second=0,
                    microsecond=0
                )
                
                # Check if we're past EOD time and have positions
                if now >= eod_time and self.open_positions:
                    logging.warning(f"⏰ EOD CLOSE TRIGGERED at {now.strftime('%H:%M:%S')}")
                    
                    # Close all positions
                    for conId, position_data in list(self.open_positions.items()):
                        try:
                            contract = position_data['contract']
                            quantity = position_data['quantity']
                            
                            # Cancel any existing orders first
                            await self.ib_interface.cancel_all_orders_for_contract(contract)
                            
                            # Place market sell order
                            order = await self.ib_interface.place_order(
                                contract,
                                'MKT',
                                quantity,
                                'SELL'
                            )
                            
                            if order:
                                logging.info(f"EOD close order placed for {contract.localSymbol}")
                            
                        except Exception as e:
                            logging.error(f"Error during EOD close for {conId}: {e}")
                    
                    # Wait until next trading day
                    await asyncio.sleep(3600)  # Sleep 1 hour then check again
                else:
                    # Check again in 30 seconds
                    await asyncio.sleep(30)
                    
            except Exception as e:
                logging.error(f"Error in EOD close monitor: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def run(self):
        """Main run loop."""
        try:
            await self.initialize()
            
            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._poll_discord_for_signals()),
                asyncio.create_task(self._monitor_open_positions()),
                asyncio.create_task(self._reconciliation_loop()),
                asyncio.create_task(self.exit_manager.monitor_exit_strategies()),
            ]
            
            # Add EOD close task if enabled
            if self.config.eod_close.get('enabled', False):
                tasks.append(asyncio.create_task(self._eod_close_monitor()))
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logging.error(f"Fatal error in SignalProcessor: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logging.info("Shutting down SignalProcessor...")
        self._shutdown_event.set()
        
        # Save final state
        all_processed_ids = []
        for channel_deque in self._processed_messages.values():
            all_processed_ids.extend(list(channel_deque))
        
        self.state_manager.save_state(self.open_positions, all_processed_ids)
        
        # Shutdown components
        await self.exit_manager.shutdown()
        await self.ib_interface.disconnect()
        await self.discord_interface.shutdown()
        await self.telegram_interface.shutdown()
        
        logging.info("SignalProcessor shutdown complete")
