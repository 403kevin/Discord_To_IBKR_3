import asyncio
import logging
from datetime import datetime, timedelta, timezone
from collections import deque
import json
import os
import pandas as pd
import numpy as np

class SignalProcessor:
    """
    The orchestrator - manages the full lifecycle of signal processing.
    FIXED VERSION: Includes missing methods and correct order handling.
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
        
        # State tracking
        self.open_positions = {}
        self._processed_messages = {profile['channel_id']: deque(maxlen=100) 
                                    for profile in self.config.profiles if profile['enabled']}
        
        # Bot startup time for filtering old messages - make it timezone aware
        self._bot_start_time = datetime.now(tz=timezone.utc)
        self._last_trade_time = None
        self._shutdown_event = asyncio.Event()
        
        # Initialize bar data storage
        self._bars_data = {}  # conId -> list of bars
        self._last_bar_update = {}  # conId -> last update time
        
        # Register fill handler
        self.ib_interface.set_order_filled_callback(self._handle_order_fill)
        
        logging.info(f"SignalProcessor initialized with {len(self._processed_messages)} active channels")

    async def start(self):
        """Main entry point - starts all processing tasks."""
        logging.info("Starting Signal Processor...")
        
        # Load existing state
        self.open_positions, loaded_message_ids = self.state_manager.load_state()
        
        # Update processed messages with loaded IDs
        for channel_id in self._processed_messages.keys():
            for msg_id in loaded_message_ids:
                self._processed_messages[channel_id].append(msg_id)
        
        logging.info(f"Loaded {len(self.open_positions)} open positions from state")
        
        # Perform initial reconciliation
        await self._initial_reconciliation()
        
        # Create all async tasks
        tasks = [
            asyncio.create_task(self._poll_discord_for_signals()),
            asyncio.create_task(self._monitor_open_positions()),
            asyncio.create_task(self._reconciliation_loop())
        ]
        
        # Wait for all tasks
        await asyncio.gather(*tasks)

    async def _initial_reconciliation(self):
        """Compares broker positions with tracked positions on startup."""
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
                    
                await asyncio.sleep(self.config.polling_interval_seconds)
                
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
                
                logging.info(f"Processing message {msg_id} from '{channel_name}'")
                logging.info(f"Raw content: '{msg_content}'")
                
                # Check timestamp - FIX: Parse timestamp correctly
                try:
                    if isinstance(timestamp_str, str):
                        # Discord timestamp format: '2025-10-15T13:05:00.123456+00:00'
                        msg_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        msg_time = timestamp_str
                    
                    # Check if message is from before bot started
                    if msg_time < self._bot_start_time:
                        logging.info(f"Ignoring stale message {msg_id} (timestamp before bot start)")
                        continue
                    
                    # Check if message is too old (> 60 seconds)
                    age_seconds = (datetime.now(msg_time.tzinfo) - msg_time).total_seconds()
                    if age_seconds > self.config.signal_max_age_seconds:
                        logging.info(f"Ignoring old message {msg_id} (age: {age_seconds:.1f}s > {self.config.signal_max_age_seconds}s)")
                        continue
                        
                except Exception as e:
                    logging.warning(f"Could not parse timestamp for message {msg_id}: {e}")
                    # If we can't parse timestamp, skip the message to be safe
                    continue
                
                # Check ignore keywords (use per-channel reject_if_contains)
                channel_ignore = profile.get('reject_if_contains', [])
                if any(word.lower() in msg_content.lower() for word in channel_ignore):
                    matched_word = self._get_matched_keyword(msg_content, channel_ignore)
                    logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Ignore keyword '{matched_word}' | Message: '{msg_content[:150]}'")
                    continue
                
                # Parse the signal
                parsed_signal = self.signal_parser.parse_signal(msg_content, profile)
                
                if not parsed_signal:
                    logging.info(f"⚠️ NOT PARSED - Channel: {channel_name} | Reason: Invalid format | Message: '{msg_content[:150]}'")
                    continue
                
                # Check cooldown
                if self._last_trade_time:
                    cooldown_seconds = self.config.cooldown_after_trade_seconds
                    time_since_last_trade = (datetime.now() - self._last_trade_time).total_seconds()
                    if time_since_last_trade < cooldown_seconds:
                        remaining = cooldown_seconds - time_since_last_trade
                        logging.info(f"⏳ COOLDOWN - {remaining:.0f}s remaining")
                        continue
                
                # Check VIX if available
                if hasattr(self, 'vix_checker') and self.vix_checker:
                    if not await self.vix_checker.check_vix_threshold():
                        logging.info(f"❌ VIX VETO - Market volatility too high")
                        continue
                
                # Check sentiment if enabled
                if profile.get('sentiment_filter', {}).get('enabled', False):
                    sentiment_score = self.sentiment_analyzer.analyze(msg_content)
                    required_score = profile['sentiment_filter']['required_score']
                    
                    if abs(sentiment_score) < required_score:
                        logging.info(f"❌ SENTIMENT VETO - Score: {sentiment_score:.2f} < Required: {required_score}")
                        continue
                
                # Execute the trade
                logging.info(f"✅ SIGNAL ACCEPTED - Executing trade for {parsed_signal['ticker']}")
                await self._execute_trade(parsed_signal, profile)
                
            except Exception as e:
                logging.error(f"Error processing message {msg_id}: {e}", exc_info=True)

    def _get_matched_keyword(self, text, keywords):
        """Helper to find which keyword matched in the text."""
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return keyword
        return None

    async def _execute_trade(self, parsed_signal, profile):
        """Execute a trade based on the parsed signal."""
        try:
            # Create the contract
            contract = await self.ib_interface.create_option_contract(
                parsed_signal['ticker'],
                parsed_signal['expiry_date'],
                parsed_signal['strike'],
                parsed_signal['contract_type'][0]
            )
            
            if not contract:
                logging.error(f"Failed to create contract for {parsed_signal}")
                return
            
            # Get current price
            ticker_data = await self.ib_interface.get_live_ticker(contract)
            if not ticker_data or ticker_data.last <= 0:
                logging.error(f"No market data for {contract.localSymbol}")
                return
            
            current_price = ticker_data.last
            
            # Calculate quantity
            funds = profile['trading']['funds_allocation']
            min_price = profile['trading']['min_price_per_contract']
            max_price = profile['trading']['max_price_per_contract']
            
            if current_price < min_price or current_price > max_price:
                logging.info(f"Price ${current_price:.2f} outside range [${min_price:.2f}-${max_price:.2f}]")
                return
            
            quantity = self._calculate_quantity(funds, current_price)
            if quantity <= 0:
                logging.info(f"Cannot afford contract at ${current_price:.2f}")
                return
            
            # Place the order
            order_type = profile['trading']['entry_order_type']
            
            # FIX: place_order returns an Order object, not a Trade
            order = await self.ib_interface.place_order(
                contract, 
                order_type,
                quantity, 
                'BUY'
            )
            
            if order:
                # Store pending position info - will be updated when fill comes
                self._pending_order = {
                    'contract': contract,
                    'quantity': quantity,
                    'profile': profile,
                    'parsed_signal': parsed_signal,
                    'order': order
                }
                
                # Native trail will be attached in the fill callback
                logging.info(f"Order placed, waiting for fill confirmation...")
                
        except Exception as e:
            logging.error(f"Error executing trade: {e}", exc_info=True)

    def _calculate_quantity(self, funds, price_per_contract):
        """Calculate how many contracts to buy."""
        contract_cost = price_per_contract * 100
        quantity = int(funds / contract_cost)
        return max(0, quantity)

    async def _handle_order_fill(self, trade):
        """Callback for handling order fills from IB - FIX: Attach native trail here."""
        try:
            contract = trade.contract
            order = trade.order
            fill_price = trade.orderStatus.avgFillPrice
            quantity = int(trade.orderStatus.filled)
            
            if order.action == 'BUY':
                # This is an entry fill
                conId = contract.conId
                
                # Get the pending order info
                if hasattr(self, '_pending_order'):
                    profile = self._pending_order['profile']
                    parsed_signal = self._pending_order['parsed_signal']
                    
                    # Track the position
                    self.open_positions[conId] = {
                        'contract': contract,
                        'quantity': quantity,
                        'entry_price': fill_price,
                        'entry_time': datetime.now(),
                        'highest_price': fill_price,
                        'signal': parsed_signal,
                        'profile': profile,
                        'breakeven_hit': False,
                        'conId': conId
                    }
                    
                    # FIX: Attach native trail AFTER fill
                    safety_net = profile.get('safety_net', {})
                    if safety_net.get('enabled', True):
                        native_trail = safety_net.get('native_trail_percent', 35)
                        trail_result = await self.ib_interface.attach_native_trail(order, native_trail)
                        if trail_result:
                            logging.info(f"✅ Native trail ({native_trail}%) attached successfully")
                        else:
                            logging.warning(f"⚠️ Failed to attach native trail")
                    
                    # Send notifications
                    await self._send_telegram_notification('ENTRY', self.open_positions[conId], None)
                    
                    # Set cooldown
                    self._last_trade_time = datetime.now()
                    
                    # Clean up pending order
                    del self._pending_order
                    
                    logging.info(f"✅ FILLED: {quantity} {contract.localSymbol} @ ${fill_price:.2f}")
                    
        except Exception as e:
            logging.error(f"Error handling order fill: {e}", exc_info=True)

    async def _monitor_open_positions(self):
        """Monitor open positions for exit conditions."""
        while not self._shutdown_event.is_set():
            try:
                for conId, position in list(self.open_positions.items()):
                    current_data = await self.ib_interface.get_live_ticker(position['contract'])
                    
                    if current_data and current_data.last > 0:
                        current_price = current_data.last
                        position['highest_price'] = max(position['highest_price'], current_price)
                        
                        # Update bar data for technical indicators
                        await self._update_bar_data(position, current_price)
                        
                        # Check exit conditions
                        exit_reason = await self._evaluate_exit_conditions(position, current_price)
                        
                        if exit_reason:
                            logging.info(f"📊 EXIT SIGNAL: {exit_reason} for {position['contract'].localSymbol}")
                            await self._close_position(conId, exit_reason, current_price)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring positions: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _update_bar_data(self, position, current_price):
        """Update bar data for technical indicators."""
        conId = position['conId']
        now = datetime.now()
        
        # Initialize if needed
        if conId not in self._bars_data:
            self._bars_data[conId] = []
            self._last_bar_update[conId] = now
        
        # Add new bar every minute
        if (now - self._last_bar_update[conId]).total_seconds() >= 60:
            self._bars_data[conId].append({
                'time': now,
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price
            })
            self._last_bar_update[conId] = now
            
            # Keep only last 100 bars
            if len(self._bars_data[conId]) > 100:
                self._bars_data[conId] = self._bars_data[conId][-100:]

    async def _evaluate_exit_conditions(self, position, current_price):
        """Evaluate all exit conditions based on priority."""
        profile = position['profile']
        exit_strategy = profile['exit_strategy']
        entry_price = position['entry_price']
        
        # Get exit priority
        exit_priority = exit_strategy.get('exit_priority', [
            'breakeven', 'rsi_hook', 'psar_flip', 'atr_trail', 'pullback_stop'
        ])
        
        for exit_type in exit_priority:
            if exit_type == 'breakeven' and not position['breakeven_hit']:
                trigger_percent = exit_strategy.get('breakeven_trigger_percent', 10)
                if ((current_price - entry_price) / entry_price * 100) >= trigger_percent:
                    position['breakeven_hit'] = True
                    if current_price <= entry_price:
                        return "Breakeven Stop"
            
            elif exit_type == 'rsi_hook':
                if exit_strategy.get('momentum_exits', {}).get('rsi_hook_enabled', False):
                    rsi_exit = self._check_rsi_hook(position, current_price)
                    if rsi_exit:
                        return rsi_exit
            
            elif exit_type == 'psar_flip':
                if exit_strategy.get('momentum_exits', {}).get('psar_enabled', False):
                    psar_exit = self._check_psar_flip(position, current_price)
                    if psar_exit:
                        return psar_exit
            
            elif exit_type == 'atr_trail':
                if exit_strategy.get('trail_method') == 'atr':
                    atr_exit = self._check_atr_trail(position, current_price)
                    if atr_exit:
                        return atr_exit
            
            elif exit_type == 'pullback_stop':
                if exit_strategy.get('trail_method') == 'pullback_percent':
                    pullback_exit = self._check_pullback_stop(position, current_price)
                    if pullback_exit:
                        return pullback_exit
        
        return None

    def _check_rsi_hook(self, position, current_price):
        """Check RSI hook exit condition."""
        conId = position['conId']
        if conId not in self._bars_data or len(self._bars_data[conId]) < 14:
            return None
        
        # Calculate RSI
        closes = [bar['close'] for bar in self._bars_data[conId][-14:]]
        rsi = self._calculate_rsi(closes)
        
        settings = position['profile']['exit_strategy'].get('momentum_exits', {}).get('rsi_settings', {})
        overbought = settings.get('overbought_level', 70)
        
        if rsi >= overbought and current_price < position['highest_price'] * 0.98:
            return "RSI Hook"
        
        return None

    def _check_psar_flip(self, position, current_price):
        """Check PSAR flip exit condition."""
        conId = position['conId']
        if conId not in self._bars_data or len(self._bars_data[conId]) < 5:
            return None
        
        # Calculate PSAR
        bars_df = pd.DataFrame(self._bars_data[conId])
        settings = position['profile']['exit_strategy'].get('momentum_exits', {}).get('psar_settings', {})
        
        start = settings.get('start', 0.02)
        increment = settings.get('increment', 0.02)
        maximum = settings.get('max', 0.2)
        
        psar = self._calculate_psar(bars_df, start, increment, maximum)
        
        # If PSAR flips above price, exit
        if psar is not None and psar > current_price:
            return "PSAR Flip"
        
        return None

    def _check_atr_trail(self, position, current_price):
        """Check ATR trailing stop."""
        conId = position['conId']
        if conId not in self._bars_data or len(self._bars_data[conId]) < 14:
            return None
        
        bars_df = pd.DataFrame(self._bars_data[conId])
        settings = position['profile']['exit_strategy']
        
        period = settings.get('atr_period', 14)
        multiplier = settings.get('atr_multiplier', 1.5)
        
        atr = self._calculate_atr(bars_df, period)
        if atr is None:
            return None
        
        stop_price = position['highest_price'] - (atr * multiplier)
        
        if current_price <= stop_price:
            return "ATR Trail"
        
        return None

    def _check_pullback_stop(self, position, current_price):
        """Check percentage pullback stop."""
        settings = position['profile']['exit_strategy']
        pullback_percent = settings.get('pullback_percent', 10)
        
        stop_price = position['highest_price'] * (1 - pullback_percent / 100)
        
        if current_price <= stop_price:
            return "Pullback Stop"
        
        return None

    def _calculate_rsi(self, closes, period=14):
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_psar(self, bars_df, start, increment, maximum):
        """Calculate Parabolic SAR."""
        if len(bars_df) < 2:
            return None
        
        # Simplified PSAR calculation
        try:
            import pandas_ta as ta
            psar_result = ta.psar(
                high=bars_df['high'],
                low=bars_df['low'],
                close=bars_df['close'],
                af0=start,
                af=increment,
                max_af=maximum
            )
            
            if psar_result is not None and len(psar_result) > 0:
                # Return the long (bullish) PSAR value
                return psar_result.iloc[-1]['PSARl_' + str(start) + '_' + str(maximum)]
        except:
            pass
        
        return None

    def _calculate_atr(self, bars_df, period):
        """Calculate Average True Range."""
        if len(bars_df) < period:
            return None
        
        try:
            import pandas_ta as ta
            atr_result = ta.atr(
                high=bars_df['high'],
                low=bars_df['low'],
                close=bars_df['close'],
                length=period
            )
            
            if atr_result is not None and len(atr_result) > 0:
                return atr_result.iloc[-1]
        except:
            pass
        
        return None

    async def _close_position(self, conId, exit_reason, exit_price):
        """Close a position."""
        try:
            position = self.open_positions[conId]
            contract = position['contract']
            quantity = position['quantity']
            
            # Cancel all orders first to prevent ghost trailing stops
            await self.ib_interface.cancel_all_orders_for_contract(contract)
            
            # Place sell order - FIX: use correct signature
            order = await self.ib_interface.place_order(
                contract, 
                'MKT',  # order_type
                quantity, 
                'SELL'
            )
            
            if order:
                # Calculate P&L
                entry_total = position['entry_price'] * quantity * 100
                exit_total = exit_price * quantity * 100
                pnl = exit_total - entry_total
                pnl_percent = (pnl / entry_total) * 100
                
                # Update position info for notification
                position['exit_price'] = exit_price
                position['pnl'] = pnl
                position['pnl_percent'] = pnl_percent
                
                # Send Telegram notification
                await self._send_telegram_notification('EXIT', position, exit_reason)
                
                # Remove from tracking
                del self.open_positions[conId]
                
                logging.info(f"✅ CLOSED: {quantity} {contract.localSymbol} @ ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_percent:.1f}%)")
                
                # Save state
                all_processed_ids = []
                for channel_deque in self._processed_messages.values():
                    all_processed_ids.extend(list(channel_deque))
                
                self.state_manager.save_state(self.open_positions, all_processed_ids)
                
        except Exception as e:
            logging.error(f"Error closing position: {e}", exc_info=True)

    async def _send_telegram_notification(self, status, position_info, exit_reason=None):
        """Send formatted Telegram notifications for trade events."""
        try:
            if status == 'ENTRY':
                message = f"""
🟢 **ENTRY FILLED**

**Symbol:** {position_info['signal']['ticker']}
**Strike:** {position_info['signal']['strike']}{position_info['signal']['contract_type']}
**Expiry:** {position_info['signal']['expiry_date']}
**Quantity:** {position_info['quantity']}
**Entry Price:** ${position_info['entry_price']:.2f}
**Total Cost:** ${position_info['entry_price'] * position_info['quantity'] * 100:.2f}
**Time:** {position_info['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            elif status == 'EXIT':
                pnl_emoji = "🟢" if position_info.get('pnl', 0) > 0 else "🔴"
                message = f"""
{pnl_emoji} **EXIT FILLED**

**Symbol:** {position_info['signal']['ticker']}
**Strike:** {position_info['signal']['strike']}{position_info['signal']['contract_type']}
**Reason:** {exit_reason}
**Entry Price:** ${position_info['entry_price']:.2f}
**Exit Price:** ${position_info.get('exit_price', 0):.2f}
**Quantity:** {position_info['quantity']}
**P&L:** ${position_info.get('pnl', 0):.2f} ({position_info.get('pnl_percent', 0):.1f}%)
"""
            
            await self.telegram_interface.send_message(message)
            
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {e}", exc_info=True)

    async def _reconciliation_loop(self):
        """Periodic reconciliation with broker."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.reconciliation_interval_seconds)
            
            try:
                logging.info("Starting periodic position reconciliation...")
                broker_positions = await self.ib_interface.get_open_positions()
                
                broker_conIds = {pos.contract.conId for pos in broker_positions}
                tracked_conIds = set(self.open_positions.keys())
                
                # Check for ghost positions
                ghost_positions = broker_conIds - tracked_conIds
                if ghost_positions:
                    logging.warning(f"Found {len(ghost_positions)} ghost positions during reconciliation")
                
                # Check for phantom positions
                phantom_positions = tracked_conIds - broker_conIds
                if phantom_positions:
                    for conId in phantom_positions:
                        logging.warning(f"Removing phantom position {conId}")
                        del self.open_positions[conId]
                
                # Save reconciled state
                all_processed_ids = []
                for channel_deque in self._processed_messages.values():
                    all_processed_ids.extend(list(channel_deque))
                
                self.state_manager.save_state(self.open_positions, all_processed_ids)
                
            except Exception as e:
                logging.error(f"Error during reconciliation: {e}", exc_info=True)

    async def shutdown(self):
        """Graceful shutdown."""
        logging.info("Shutting down SignalProcessor...")
        self._shutdown_event.set()
        
        # Save final state
        all_processed_ids = []
        for channel_deque in self._processed_messages.values():
            all_processed_ids.extend(list(channel_deque))
        
        self.state_manager.save_state(self.open_positions, all_processed_ids)
