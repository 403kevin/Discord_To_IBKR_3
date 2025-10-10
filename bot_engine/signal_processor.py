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
    FIXED VERSION: Constructor matches main.py, dynamic exits properly wired
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
        
        # Load existing state
        loaded_state = await self.state_manager.load_state()
        if loaded_state:
            self.open_positions = loaded_state
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
                if self.config.sentiment_filter.get('enabled', False):
                    sentiment = self.sentiment_analyzer.get_sentiment_score(msg_content)
                    
                    if sentiment is not None:
                        threshold = profile.get('sentiment_threshold', 0.0)
                        
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
            contract = await self.ib_interface.create_option_contract(
                signal['ticker'],
                signal['expiry_date'],
                signal['strike'],
                signal['contract_type']
            )
            
            if not contract:
                logging.error(f"Failed to create contract for {signal}")
                return
            
            # Get market price
            ticker = await self.ib_interface.get_live_ticker(contract)
            if not ticker or not ticker.last:
                logging.error(f"Failed to get market price for {signal['ticker']}")
                return
            
            market_price = ticker.last
            
            # Check price limits
            min_price = profile['trading']['min_price_per_contract']
            max_price = profile['trading']['max_price_per_contract']
            
            if market_price < min_price:
                logging.info(f"Price ${market_price} below minimum ${min_price} for {signal['ticker']}")
                return
            
            if market_price > max_price:
                logging.info(f"Price ${market_price} above maximum ${max_price} for {signal['ticker']}")
                return
            
            # Calculate position size
            allocation = profile['trading']['funds_allocation']
            quantity = self._calculate_position_size(allocation, market_price)
            
            if quantity <= 0:
                logging.info(f"Position size is 0 for {signal['ticker']} at ${market_price}")
                return
            
            # Place the order
            order = await self.ib_interface.place_order(
                contract, 
                profile['trading']['entry_order_type'],
                quantity
            )
            
            if order:
                # Attach native trail if configured
                if profile['safety_net']['enabled']:
                    trail_percent = profile['safety_net']['native_trail_percent']
                    await self.ib_interface.attach_native_trail(order, trail_percent)
                
                # Initialize position tracking
                position_info = {
                    'contract': contract,
                    'quantity': quantity,
                    'entry_price': market_price,
                    'entry_time': datetime.now(),
                    'profile': profile,
                    'signal': signal,
                    'order': order,
                    'highest_price': market_price,
                    'breakeven_triggered': False,
                    'last_bar': None,
                    'bars_history': [],
                    'atr_values': [],
                    'psar_values': {'sar': market_price, 'ep': market_price, 'af': 0.02},
                    'rsi_values': []
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
                        
                        # FIX: Wire up the bar data update (THIS WAS MISSING)
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
        profile = position['profile']
        exit_config = profile['exit_strategy']
        
        # Check if we need to form a new bar
        min_ticks = exit_config.get('min_ticks_per_bar', 5)
        
        if not position['last_bar']:
            position['last_bar'] = {
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'tick_count': 1
            }
        else:
            bar = position['last_bar']
            bar['high'] = max(bar['high'], current_price)
            bar['low'] = min(bar['low'], current_price)
            bar['close'] = current_price
            bar['tick_count'] += 1
            
            # Complete the bar if we have enough ticks
            if bar['tick_count'] >= min_ticks:
                position['bars_history'].append(bar.copy())
                position['last_bar'] = None
                
                # Keep only last 50 bars
                if len(position['bars_history']) > 50:
                    position['bars_history'] = position['bars_history'][-50:]
                
                # Update technical indicators
                await self._update_technical_indicators(position)

    async def _update_technical_indicators(self, position):
        """Update ATR, RSI, and PSAR values."""
        bars = position['bars_history']
        if len(bars) < 2:
            return
        
        profile = position['profile']
        exit_config = profile['exit_strategy']
        
        # Calculate ATR
        if exit_config.get('trail_method') == 'atr':
            atr_period = exit_config['trail_settings'].get('atr_period', 14)
            atr = self._calculate_atr(bars, atr_period)
            if atr:
                position['atr_values'].append(atr)
        
        # Calculate RSI
        if exit_config.get('momentum_exits', {}).get('rsi_hook_enabled'):
            rsi_period = exit_config['momentum_exits']['rsi_settings'].get('period', 14)
            rsi = self._calculate_rsi(bars, rsi_period)
            if rsi:
                position['rsi_values'].append(rsi)
        
        # Update PSAR
        if exit_config.get('momentum_exits', {}).get('psar_enabled'):
            self._update_psar(position, bars[-1])

    def _calculate_atr(self, bars, period):
        """Calculate Average True Range."""
        if len(bars) < period:
            return None
        
        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return np.mean(true_ranges[-period:])
        return None

    def _calculate_rsi(self, bars, period):
        """Calculate Relative Strength Index."""
        if len(bars) < period + 1:
            return None
        
        closes = [bar['close'] for bar in bars]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:]) if gains else 0
        avg_loss = np.mean(losses[-period:]) if losses else 0
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _update_psar(self, position, latest_bar):
        """Update Parabolic SAR values."""
        psar = position['psar_values']
        profile = position['profile']
        settings = profile['exit_strategy'].get('momentum_exits', {}).get('psar_settings', {})
        
        increment = settings.get('increment', 0.02)
        max_af = settings.get('max', 0.2)
        
        # Update extreme point
        if latest_bar['high'] > psar['ep']:
            psar['ep'] = latest_bar['high']
            psar['af'] = min(psar['af'] + increment, max_af)
        
        # Calculate new SAR
        psar['sar'] = psar['sar'] + psar['af'] * (psar['ep'] - psar['sar'])

    async def _evaluate_exit_conditions(self, position, current_price):
        """Evaluate all exit strategies based on configured priority."""
        profile = position['profile']
        exit_config = profile['exit_strategy']
        
        # Get exit priority
        exit_priority = exit_config.get('exit_priority', 
            ['breakeven', 'rsi_hook', 'psar_flip', 'atr_trail', 'pullback_stop'])
        
        for exit_type in exit_priority:
            exit_reason = None
            
            if exit_type == 'breakeven':
                exit_reason = self._check_breakeven(position, current_price)
            elif exit_type == 'rsi_hook' and exit_config.get('momentum_exits', {}).get('rsi_hook_enabled'):
                exit_reason = self._check_rsi_hook(position, current_price)
            elif exit_type == 'psar_flip' and exit_config.get('momentum_exits', {}).get('psar_enabled'):
                exit_reason = self._check_psar_flip(position, current_price)
            elif exit_type == 'atr_trail' and exit_config.get('trail_method') == 'atr':
                exit_reason = self._check_atr_trail(position, current_price)
            elif exit_type == 'pullback_stop' and exit_config.get('trail_method') == 'pullback_percent':
                exit_reason = self._check_pullback_stop(position, current_price)
            
            if exit_reason:
                return exit_reason
        
        return None

    def _check_breakeven(self, position, current_price):
        """Check breakeven stop condition."""
        profile = position['profile']
        exit_config = profile['exit_strategy']
        entry_price = position['entry_price']
        
        trigger_percent = exit_config.get('breakeven_trigger_percent', 0)
        if trigger_percent <= 0:
            return None
        
        trigger_price = entry_price * (1 + trigger_percent / 100)
        
        # Check if we should trigger breakeven
        if not position['breakeven_triggered'] and current_price >= trigger_price:
            position['breakeven_triggered'] = True
            logging.info(f"Breakeven triggered for {position['signal']['ticker']} at ${current_price:.2f}")
        
        # Check if we should exit at breakeven
        if position['breakeven_triggered'] and current_price <= entry_price:
            return "Breakeven Stop"
        
        return None

    def _check_rsi_hook(self, position, current_price):
        """Check RSI hook exit condition."""
        if len(position['rsi_values']) < 2:
            return None
        
        profile = position['profile']
        settings = profile['exit_strategy']['momentum_exits']['rsi_settings']
        overbought = settings.get('overbought_level', 70)
        
        # Check for RSI hook pattern
        if (position['rsi_values'][-2] > overbought and 
            position['rsi_values'][-1] < overbought):
            return "RSI Hook"
        
        return None

    def _check_psar_flip(self, position, current_price):
        """Check PSAR flip exit condition."""
        psar = position['psar_values']
        
        # Exit if price crosses below SAR
        if current_price < psar['sar']:
            return "PSAR Flip"
        
        return None

    def _check_atr_trail(self, position, current_price):
        """Check ATR-based trailing stop."""
        if not position['atr_values']:
            return None
        
        profile = position['profile']
        settings = profile['exit_strategy']['trail_settings']
        multiplier = settings.get('atr_multiplier', 1.5)
        
        # Calculate stop level
        atr = position['atr_values'][-1]
        stop_distance = atr * multiplier
        stop_level = position['highest_price'] - stop_distance
        
        if current_price <= stop_level:
            return f"ATR Trail ({multiplier}x)"
        
        return None

    def _check_pullback_stop(self, position, current_price):
        """Check pullback percentage trailing stop."""
        profile = position['profile']
        settings = profile['exit_strategy']['trail_settings']
        pullback_percent = settings.get('pullback_percent', 10)
        
        # Calculate stop level
        stop_level = position['highest_price'] * (1 - pullback_percent / 100)
        
        if current_price <= stop_level:
            return f"Pullback Stop ({pullback_percent}%)"
        
        return None

    async def _execute_close_trade(self, conId, reason, current_price=None):
        """Close a position."""
        try:
            position = self.open_positions[conId]
            
            # Place sell order
            order = await self.ib_interface.place_order(
                position['contract'],
                'MKT',
                position['quantity'],
                action='SELL'
            )
            
            if order:
                # Calculate P&L
                if current_price:
                    exit_price = current_price
                else:
                    exit_price = position['entry_price']  # Fallback
                
                entry_total = position['entry_price'] * position['quantity'] * 100
                exit_total = exit_price * position['quantity'] * 100
                pnl = exit_total - entry_total
                pnl_percent = (pnl / entry_total) * 100 if entry_total > 0 else 0
                
                # Remove from tracking
                del self.open_positions[conId]
                
                # Send notification
                position['exit_price'] = exit_price
                position['pnl'] = pnl
                position['pnl_percent'] = pnl_percent
                await self._send_trade_notification(position, 'EXIT', reason)
                
                logging.info(f"✅ TRADE CLOSED - {position['signal']['ticker']} | Reason: {reason} | P&L: ${pnl:.2f} ({pnl_percent:.1f}%)")
                
        except Exception as e:
            logging.error(f"Error closing position {conId}: {e}")

    async def _send_trade_notification(self, position_info, status, exit_reason=None):
        """Send formatted Telegram notification."""
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

    async def _reconciliation_loop(self):
        """Periodic reconciliation with broker."""
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
                await self.state_manager.save_state(self.open_positions)
                logging.info("--- Position reconciliation complete ---")
                
            except Exception as e:
                logging.error(f"Error during reconciliation: {e}")

    async def _initial_reconciliation(self):
        """Initial reconciliation on startup."""
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
            
            await self.state_manager.save_state(self.open_positions)
            logging.info(f"Reconciliation complete. Tracking {len(self.open_positions)} verified positions.")
            
        except Exception as e:
            logging.error(f"Error during initial reconciliation: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        logging.info("Shutting down SignalProcessor...")
        self._shutdown_event.set()
        await self.state_manager.save_state(self.open_positions)