import asyncio
import logging
from datetime import datetime, timedelta
from collections import deque
import json
import os
import pandas as pd
import numpy as np

class SignalProcessor:
    """
    The orchestrator - manages the full lifecycle of signal processing.
    FIXED VERSION: Includes all missing methods (_initial_reconciliation, shutdown, _reconciliation_loop)
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
        for profile in self.config.profiles:
            if profile['enabled']:
                channel_id = profile['channel_id']
                self._processed_messages[channel_id] = deque(maxlen=1000)
                
        logging.info(f"SignalProcessor initialized with {len(self._processed_messages)} active channels")

    async def start(self):
        """Main entry point - starts all processing tasks."""
        logging.info("Starting Signal Processor...")
        
        # Load state - returns tuple, not a coroutine
        self.open_positions, loaded_message_ids = self.state_manager.load_state()
        
        # Update processed messages with loaded IDs
        for channel_id in self._processed_messages.keys():
            for msg_id in loaded_message_ids:
                self._processed_messages[channel_id].append(msg_id)
        
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

    async def _initial_reconciliation(self):
        """Initial reconciliation on startup - verifies positions with broker."""
        logging.info("Performing initial state reconciliation with broker...")
        
        try:
            ib_positions = await self.ib_interface.get_open_positions()
            
            # Filter out zero positions
            active_positions = [p for p in ib_positions if p.position != 0]
            
            logging.info(f"Reconciliation: Found {len(active_positions)} active positions at broker")
            
            # Check against saved state
            for ib_pos in active_positions:
                if ib_pos.contract.conId in self.open_positions:
                    logging.info(f"Position {ib_pos.contract.localSymbol} verified with broker")
                else:
                    logging.warning(f"Untracked position found: {ib_pos.contract.localSymbol} qty={ib_pos.position}")
            
            # Remove positions from state that broker doesn't have
            for conId in list(self.open_positions.keys()):
                if not any(p.contract.conId == conId for p in active_positions):
                    logging.warning(f"Removing stale position {conId} from state")
                    del self.open_positions[conId]
            
            # Save corrected state
            all_processed_ids = []
            for channel_deque in self._processed_messages.values():
                all_processed_ids.extend(list(channel_deque))
            
            self.state_manager.save_state(self.open_positions, all_processed_ids)
            logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")
            
        except Exception as e:
            logging.error(f"Error during initial reconciliation: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        logging.info("Shutting down SignalProcessor...")
        self._shutdown_event.set()
        
        # Save final state
        all_processed_ids = []
        for channel_deque in self._processed_messages.values():
            all_processed_ids.extend(list(channel_deque))
        
        self.state_manager.save_state(self.open_positions, all_processed_ids)

    async def _reconciliation_loop(self):
        """Periodic reconciliation with broker - runs every 60 seconds."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.reconciliation_interval_seconds)
            
            try:
                logging.info("--- Starting periodic position reconciliation ---")
                
                # Get broker positions
                ib_positions = await self.ib_interface.get_open_positions()
                
                # Filter out zero positions
                active_ib_positions = [p for p in ib_positions if p.position != 0]
                
                logging.info(f"Reconciliation: Found {len(active_ib_positions)} active positions at broker")
                
                # Check for untracked positions
                untracked_count = 0
                for ib_pos in active_ib_positions:
                    if ib_pos.contract.conId not in self.open_positions:
                        untracked_count += 1
                        logging.warning(f"Found untracked position: {ib_pos.contract.localSymbol} qty={ib_pos.position}")
                
                if untracked_count > 0:
                    logging.warning(f"Reconciliation: Found {untracked_count} untracked position(s) at broker")
                
                # Check for positions we track but broker doesn't have
                for conId, position in list(self.open_positions.items()):
                    broker_has_it = any(p.contract.conId == conId for p in active_ib_positions)
                    if not broker_has_it:
                        logging.warning(f"Position {position['signal']['ticker']} not found at broker, removing from tracking")
                        del self.open_positions[conId]
                
                # Save state
                all_processed_ids = []
                for channel_deque in self._processed_messages.values():
                    all_processed_ids.extend(list(channel_deque))
                
                self.state_manager.save_state(self.open_positions, all_processed_ids)
                logging.info("--- Position reconciliation complete ---")
                
            except Exception as e:
                logging.error(f"Error during reconciliation: {e}")

    async def _poll_discord_for_signals(self):
        """Continuously polls Discord channels for new messages."""
        while not self._shutdown_event.is_set():
            for profile in self.config.profiles:
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
                    
            await asyncio.sleep(self.config.polling_interval_seconds)

    async def _process_new_signals(self, raw_messages, profile):
        """Processes raw Discord messages into trade signals with enhanced logging."""
        channel_name = profile.get('channel_name', profile['channel_id'])
        
        for msg in raw_messages:
            msg_id = None  # Initialize OUTSIDE try block
            try:
                # Handle tuple format: (msg_id, msg_content, timestamp)
                if isinstance(msg, tuple):
                    msg_id = msg[0]
                    msg_content = msg[1]
                    msg_timestamp = msg[2].isoformat() if len(msg) > 2 else None
                    msg_author = 'Discord'
                else:
                    # Handle dict format
                    msg_id = msg.get('id', 'UNKNOWN_ID')
                    msg_content = msg.get('content', '')
                    msg_author = msg.get('author', {}).get('username', 'Unknown')
                    msg_timestamp = msg.get('timestamp')
                
                # Check if message is too old
                if msg_timestamp:
                    if isinstance(msg_timestamp, str):
                        msg_time = datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00'))
                    else:
                        msg_time = msg_timestamp
                        
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
                
                logging.info(f"✓ PARSED - Ticker: {parsed_signal['ticker']} | Strike: {parsed_signal['strike']} | Type: {parsed_signal['contract_type']}")
                
                # Check sentiment if enabled
                if profile.get('sentiment_filter', {}).get('enabled', False):
                    sentiment = self.sentiment_analyzer.analyze_sentiment(msg_content)
                    threshold = profile['sentiment_filter'].get('min_compound_score', 0.1)
                    
                    if sentiment < threshold:
                        logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Low sentiment ({sentiment:.2f}) | Signal: {parsed_signal['ticker']}")
                        
                        # Send Telegram notification for sentiment veto
                        veto_msg = (
                            f"🚫 *Trade Vetoed by Sentiment*\n\n"
                            f"Channel: {channel_name}\n"
                            f"Signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type']}\n"
                            f"Sentiment Score: {sentiment:.2f}\n"
                            f"Threshold: {threshold:.2f}\n"
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
                    cooldown = self.config.cooldown_after_trade_seconds
                    if time_since_last < cooldown:
                        logging.info(f"❌ REJECTED - Channel: {channel_name} | Reason: Cooldown ({time_since_last:.0f}s < {cooldown}s) | Signal: {parsed_signal['ticker']}")
                        continue
                
                # Execute the trade
                logging.info(f"🎯 EXECUTING - Channel: {channel_name} | Signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type']}")
                await self._execute_trade(parsed_signal, profile)
                
                # Save state after successful trade
                all_processed_ids = []
                for channel_deque in self._processed_messages.values():
                    all_processed_ids.extend(list(channel_deque))
                
                self.state_manager.save_state(self.open_positions, all_processed_ids)
                
            except Exception as e:
                # msg_id is now guaranteed to be in scope (initialized at top of loop)
                error_msg_id = msg_id if msg_id else 'UNKNOWN'
                logging.error(f"Error processing message {error_msg_id}: {e}", exc_info=True)

    def _contains_keywords(self, text, keywords):
        """Helper to check if text contains any keyword."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _get_matched_keyword(self, text, keywords):
        """Helper to get which keyword was matched."""
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
                parsed_signal['expiry'],
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
            
            # Place the order with native trail
            safety_net = profile.get('safety_net', {})
            native_trail = safety_net.get('native_trail_percent', 0.35)
            
            trade = await self.ib_interface.place_order(
                contract, 
                quantity, 
                'BUY',
                native_trail_percent=native_trail
            )
            
            if trade and trade.orderStatus.status == 'Filled':
                # Track the position
                conId = contract.conId
                self.open_positions[conId] = {
                    'contract': contract,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'highest_price': current_price,
                    'signal': parsed_signal,
                    'profile': profile,
                    'breakeven_triggered': False,
                    'bars_data': [],
                    'atr_values': [],
                    'rsi_values': [],
                    'psar_values': {'sar': current_price, 'ep': current_price, 'af': 0.02}
                }
                
                # Update last trade time
                self._last_trade_time = datetime.now()
                
                # Send Telegram notification
                await self._send_telegram_notification(
                    'ENTRY', 
                    self.open_positions[conId], 
                    'Manual/Unknown reason'
                )
                
                logging.info(f"✅ FILLED: {quantity} {contract.localSymbol} @ ${current_price:.2f}")
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}", exc_info=True)

    def _calculate_quantity(self, allocation, price):
        """Calculate the number of contracts to buy based on allocation and price."""
        if price <= 0:
            return 0
        contract_cost = price * 100
        return int(allocation / contract_cost)

    async def _monitor_open_positions(self):
        """Monitor positions and execute dynamic exit strategies."""
        while not self._shutdown_event.is_set():
            for conId, position in list(self.open_positions.items()):
                try:
                    # Get current market data
                    current_data = await self.ib_interface.get_live_ticker(position['contract'])
                    
                    if current_data and current_data.last > 0:
                        current_price = current_data.last
                        
                        # Update tracking data
                        position['highest_price'] = max(position['highest_price'], current_price)
                        
                        # Wire up the bar data update
                        await self._update_bar_data(position, current_price)
                        
                        # Evaluate exit conditions based on priority
                        exit_reason = await self._evaluate_exit_conditions(position, current_price)
                        
                        if exit_reason:
                            await self._execute_close_trade(conId, exit_reason, current_price)
                            
                except Exception as e:
                    logging.error(f"Error monitoring position {conId}: {e}")
                    
            await asyncio.sleep(5)

    async def _update_bar_data(self, position, current_price):
        """Update bar data for technical indicators."""
        bars = position['bars_data']
        bars.append(current_price)
        
        # Keep only recent bars
        max_bars = 50
        if len(bars) > max_bars:
            position['bars_data'] = bars[-max_bars:]
        
        # Update ATR
        if len(bars) >= 14:
            df = pd.DataFrame({'close': bars})
            df['high'] = df['close'].rolling(window=5).max()
            df['low'] = df['close'].rolling(window=5).min()
            
            # Calculate ATR
            df['tr'] = df[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                             abs(x['high'] - x['close']), 
                             abs(x['low'] - x['close'])), 
                axis=1
            )
            atr = df['tr'].rolling(window=14).mean().iloc[-1]
            position['atr_values'].append(atr)
        
        # Update RSI
        if len(bars) >= 14:
            prices = np.array(bars)
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                position['rsi_values'].append(rsi)
        
        # Update PSAR
        psar = position['psar_values']
        is_call = position['signal']['contract_type'] == 'CALL'
        
        if is_call:
            if current_price > psar['ep']:
                psar['ep'] = current_price
                psar['af'] = min(psar['af'] + 0.02, 0.2)
            psar['sar'] = psar['sar'] + psar['af'] * (psar['ep'] - psar['sar'])
        else:
            if current_price < psar['ep']:
                psar['ep'] = current_price
                psar['af'] = min(psar['af'] + 0.02, 0.2)
            psar['sar'] = psar['sar'] - psar['af'] * (psar['sar'] - psar['ep'])

    async def _evaluate_exit_conditions(self, position, current_price):
        """Evaluate all exit conditions based on priority."""
        profile = position['profile']
        exit_strategy = profile['exit_strategy']
        
        # Priority 1: Breakeven stop (software-based)
        breakeven_percent = exit_strategy.get('breakeven_trigger_percent', 0.10)
        entry_price = position['entry_price']
        
        if not position['breakeven_triggered']:
            pnl_percent = (current_price - entry_price) / entry_price
            if pnl_percent >= breakeven_percent:
                position['breakeven_triggered'] = True
                logging.info(f"Breakeven activated at {pnl_percent*100:.1f}% profit")
        
        if position['breakeven_triggered'] and current_price <= entry_price:
            return "Breakeven Stop"
        
        # Priority 2: Trailing Stop (ATR or Pullback)
        trail_method = exit_strategy.get('trail_method', 'atr')
        
        if trail_method == 'atr' and position['atr_values']:
            atr = position['atr_values'][-1]
            atr_multiplier = exit_strategy.get('atr_multiplier', 2.0)
            stop_distance = atr * atr_multiplier
            trailing_stop = position['highest_price'] - stop_distance
            
            if current_price <= trailing_stop:
                return f"ATR Trail (${trailing_stop:.2f})"
        
        elif trail_method == 'pullback_percent':
            pullback = exit_strategy.get('trail_pullback_percent', 0.10)
            trailing_stop = position['highest_price'] * (1 - pullback)
            
            if current_price <= trailing_stop:
                return f"Pullback Trail ({pullback*100:.0f}%)"
        
        # Priority 3: Momentum exits (PSAR/RSI)
        momentum_exits = exit_strategy.get('momentum_exits', {})
        
        if momentum_exits.get('psar_enabled', False):
            psar_data = position['psar_values']
            is_call = position['signal']['contract_type'] == 'CALL'
            
            if is_call and current_price < psar_data['sar']:
                return "PSAR Flip (Bearish)"
            elif not is_call and current_price > psar_data['sar']:
                return "PSAR Flip (Bullish)"
        
        if momentum_exits.get('rsi_hook_enabled', False) and position['rsi_values']:
            rsi = position['rsi_values'][-1]
            is_call = position['signal']['contract_type'] == 'CALL'
            
            if is_call and rsi > 70:
                if len(position['rsi_values']) >= 2:
                    if position['rsi_values'][-2] > rsi:
                        return "RSI Hook from Overbought"
            
            if not is_call and rsi < 30:
                if len(position['rsi_values']) >= 2:
                    if position['rsi_values'][-2] < rsi:
                        return "RSI Hook from Oversold"
        
        return None

    async def _execute_close_trade(self, conId, exit_reason, exit_price):
        """Close a position and send notifications."""
        try:
            position = self.open_positions[conId]
            contract = position['contract']
            quantity = position['quantity']
            
            # Cancel all orders first to prevent ghost trailing stops
            await self.ib_interface.cancel_all_orders_for_contract(contract)
            
            # Place sell order
            trade = await self.ib_interface.place_order(
                contract, 
                quantity, 
                'SELL'
            )
            
            if trade and trade.orderStatus.status == 'Filled':
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

    async def _send_telegram_notification(self, status, position_info, exit_reason):
        """Send formatted Telegram notifications for trade events."""
        try:
            signal = position_info['signal']
            profile = position_info['profile']
            channel_name = profile.get('channel_name', 'Unknown')
            
            if status == 'ENTRY':
                sentiment_text = "N/A"
                if signal.get('sentiment'):
                    sentiment_text = f"{signal['sentiment']:.2f}"
                
                # Get exit strategy info
                trail_method = profile['exit_strategy'].get('trail_method', 'N/A')
                momentum_exits = []
                if profile['exit_strategy'].get('momentum_exits', {}).get('psar_enabled'):
                    momentum_exits.append('PSAR')
                if profile['exit_strategy'].get('momentum_exits', {}).get('rsi_hook_enabled'):
                    momentum_exits.append('RSI')
                momentum_text = ', '.join(momentum_exits) if momentum_exits else 'None'
                
                message = (
                    f"✅ *Trade Entry Confirmed* ✅\n\n"
                    f"*Source Channel:* {channel_name}\n"
                    f"*Contract:* {signal['ticker']} {position_info['contract'].localSymbol}\n"
                    f"*Quantity:* {position_info['quantity']}\n"
                    f"*Entry Price:* ${position_info['entry_price']:.2f}\n"
                    f"*Vader Sentiment:* {sentiment_text}\n"
                    f"*Trail Method:* {trail_method.upper()}\n"
                    f"*Momentum Exit:* {momentum_text}"
                )
            else:  # EXIT
                pnl = position_info.get('pnl', 0)
                pnl_percent = position_info.get('pnl_percent', 0)
                exit_price = position_info.get('exit_price', 0)
                
                emoji = "🟢" if pnl >= 0 else "🔴"
                
                message = (
                    f"{emoji} *SELL Order Executed*\n\n"
                    f"*Contract:* {signal['ticker']} {position_info['contract'].localSymbol}\n"
                    f"*Exit Price:* ${exit_price:.2f}\n"
                    f"*Reason:* {exit_reason}\n"
                    f"*P&L:* ${pnl:.2f} ({pnl_percent:.1f}%)"
                )
            
            await self.telegram_interface.send_message(message)
            
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {e}")

    async def _handle_order_fill(self, trade):
        """Callback for handling order fills from IB."""
        try:
            order = trade.order
            contract = trade.contract
            fill = trade.fills[-1] if trade.fills else None
            
            if not fill:
                return
            
            logging.info(f"Order filled: {order.action} {fill.execution.shares} {contract.localSymbol} @ ${fill.execution.avgPrice:.2f}")
            
            # Update last trade time for cooldown
            self._last_trade_time = datetime.now()
            
        except Exception as e:
            logging.error(f"Error handling order fill: {e}", exc_info=True)
