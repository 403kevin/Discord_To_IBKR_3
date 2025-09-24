"""
Bot_Engine/signal_processor.py

Author: 403-Forbidden
Purpose: The central nervous system of the trading bot. This module orchestrates
         the entire operational flow from signal ingestion to trade execution
         and real-time position management.
"""
import asyncio
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
from ib_insync import Order, Trade, Contract
from services.sentiment_analyzer import SentimentAnalyzer

class SignalProcessor:
    def __init__(self, config, ib_interface, discord_interface, telegram_interface, signal_parser, state_manager):
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.telegram_interface = telegram_interface
        self.signal_parser = signal_parser
        self.state_manager = state_manager
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self.open_positions = self.state_manager.load_positions()
        self.position_data_cache = {}

    async def start(self):
        logging.info("Starting Signal Processor...")
        await self.telegram_interface.send_message("🤖 Bot is starting up...")
        try:
            await self.ib_interface.connect()
            await self.discord_interface.initialize_and_login()
            await self._initialize_open_positions()
            tasks = [self.poll_discord_for_signals(), self.process_market_data_stream()]
            await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"A critical error occurred in the main start sequence: {e}", exc_info=True)
            await self.telegram_interface.send_message(f"🚨 CRITICAL ERROR: {e}. Shutting down.")
        finally:
            await self.shutdown()

    async def _initialize_open_positions(self):
        if not self.open_positions: return
        logging.info(f"Initializing {len(self.open_positions)} open positions from state file...")
        for conId, pos_data in self.open_positions.items():
            contract = pos_data['contract']
            await self.ib_interface.subscribe_to_market_data(contract)
            await self._update_position_data_cache(contract)
            logging.info(f"Successfully re-initialized and subscribed to data for {contract.localSymbol}")

    async def poll_discord_for_signals(self):
        while True:
            # EOD check is now handled within the market data processor
            try:
                raw_messages = await self.discord_interface.poll_for_new_messages()
                if raw_messages:
                    profile = self.config.profiles[0]
                    signals_with_raw_text = self.signal_parser.parse_messages(raw_messages, profile)
                    for signal, raw_text in signals_with_raw_text:
                        signal['raw_text'] = raw_text
                        await self.execute_trade_from_signal(signal, profile)
            except Exception as e:
                logging.error(f"Error in Discord polling loop: {e}", exc_info=True)
            await asyncio.sleep(self.config.polling_interval_seconds)

    async def process_market_data_stream(self):
        logging.info("Starting market data stream processor...")
        while True:
            try:
                if self._is_eod():
                    await self.flatten_all_positions()
                    break

                ticker = await self.ib_interface.market_data_queue.get()
                conId = ticker.contract.conId
                if conId in self.open_positions:
                    self._append_tick_to_cache(ticker)
                    if self._resample_ticks_to_bar(ticker.contract):
                         profile = self.open_positions[conId]['profile']
                         await self.evaluate_dynamic_exit(ticker.contract, profile)
            except Exception as e:
                logging.error(f"Error in market data processing loop: {e}", exc_info=True)

    async def evaluate_dynamic_exit(self, contract: Contract, profile: dict):
        conId = contract.conId
        position_data = self.open_positions.get(conId)
        if not position_data: return
        
        df = self.position_data_cache[conId].get('bars')
        if df is None or len(df) < 2: return

        exit_settings = profile['graceful_exit']
        is_call = position_data['contract'].right == 'Call'

        # --- Primary Trail Method (ATR or Pullback) ---
        trail_method = exit_settings.get('trail_method', 'atr')
        if trail_method == 'atr':
            atr_settings = exit_settings['trail_settings']
            atr = ta.atr(df['high'], df['low'], df['close'], length=atr_settings['atr_period'])
            if atr is not None and not pd.isna(atr.iloc[-1]):
                stop_price = df['close'].iloc[-1] - (atr.iloc[-1] * atr_settings['atr_multiplier']) if is_call else df['close'].iloc[-1] + (atr.iloc[-1] * atr_settings['atr_multiplier'])
                if (is_call and df['close'].iloc[-1] < stop_price) or (not is_call and df['close'].iloc[-1] > stop_price):
                    await self.execute_close_trade(contract, "ATR Trail Stop")
                    return
        elif trail_method == 'pullback_percent':
            pullback_pct = exit_settings['trail_settings']['pullback_percent'] / 100.0
            high_water_mark = df['high'].max()
            low_water_mark = df['low'].min()
            if (is_call and df['close'].iloc[-1] < high_water_mark * (1 - pullback_pct)) or \
               (not is_call and df['close'].iloc[-1] > low_water_mark * (1 + pullback_pct)):
                await self.execute_close_trade(contract, "Pullback Percent Stop")
                return

        # --- Momentum Exits (RSI or PSAR) ---
        momentum_settings = exit_settings['momentum_exits']
        if momentum_settings['rsi_hook_enabled']:
            rsi_settings = momentum_settings['rsi_settings']
            rsi = ta.rsi(df['close'], length=rsi_settings['period'])
            if rsi is not None and len(rsi) >= 2:
                last_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
                if is_call and prev_rsi > rsi_settings['overbought_level'] and last_rsi < rsi_settings['overbought_level']:
                    await self.execute_close_trade(contract, "RSI Hook (Overbought)")
                    return
                elif not is_call and prev_rsi < rsi_settings['oversold_level'] and last_rsi > rsi_settings['oversold_level']:
                    await self.execute_close_trade(contract, "RSI Hook (Oversold)")
                    return
        
        if momentum_settings['psar_enabled']:
            psar_settings = momentum_settings['psar_settings']
            psar = ta.psar(df['high'], df['low'], **psar_settings)
            if psar is not None:
                psar_up = psar[f'PSARl_{psar_settings["start"]}_{psar_settings["max"]}'].iloc[-1]
                psar_down = psar[f'PSARs_{psar_settings["start"]}_{psar_settings["max"]}'].iloc[-1]
                last_close = df['close'].iloc[-1]
                if is_call and not pd.isna(psar_up) and last_close < psar_up:
                    await self.execute_close_trade(contract, "PSAR Flip to Downtrend")
                    return
                elif not is_call and not pd.isna(psar_down) and last_close > psar_down:
                    await self.execute_close_trade(contract, "PSAR Flip to Uptrend")
                    return

    async def execute_trade_from_signal(self, signal: dict, profile: dict):
        if signal['action'] != 'BTO': return
        
        sentiment_config = profile.get('sentiment_filter', {})
        if sentiment_config.get('enabled', False):
            score = self.sentiment_analyzer.get_sentiment_score(signal['raw_text'])
            is_call = signal['right'] == 'Call'
            call_threshold = sentiment_config.get('sentiment_threshold', 0.05)
            put_threshold = sentiment_config.get('put_sentiment_threshold', -0.05)
            if (is_call and score < call_threshold) or (not is_call and score > put_threshold):
                reason = f"Sentiment score {score:.4f} did not meet threshold for a {signal['right']}."
                await self._veto_trade(signal, profile, reason)
                return

        contract = await self.ib_interface.get_contract(signal['ticker'], 'OPTION', signal['expiry'], signal['strike'], signal['right'])
        if not contract: return
        
        ticker = await self.ib_interface.get_live_ticker(contract)
        if ticker is None or ticker.last <= 0:
            logging.warning(f"Could not retrieve valid live price for {contract.localSymbol}. Aborting.")
            return
        
        price = ticker.ask if ticker.ask > 0 else ticker.last

        trading_config = profile['trading']
        if not (trading_config['min_price_per_contract'] <= price <= trading_config['max_price_per_contract']):
            reason = f"Price ${price:.2f} is outside acceptable range."
            await self._veto_trade(signal, profile, reason)
            return

        quantity = int(trading_config['funds_allocation'] // (price * 100))
        if quantity == 0:
            reason = f"Not enough funds for 1 contract at ${price:.2f}."
            await self._veto_trade(signal, profile, reason)
            return
        
        logging.info(f"Calculated quantity: {quantity} for {contract.localSymbol} at ${price:.2f}")

        order = Order(action="BUY", orderType=trading_config['entry_order_type'], totalQuantity=quantity, tif=trading_config['time_in_force'])
        trade = await self.ib_interface.place_order(contract, order)
        
        if trade:
            trade.filledEvent += self._on_order_filled

    async def _on_order_filled(self, trade: Trade):
        contract = trade.contract
        logging.info(f"TRADE FILLED: {contract.localSymbol} @ {trade.orderStatus.avgFillPrice}")
        await self.telegram_interface.send_message(f"✅ BTO FILLED: {trade.order.totalQuantity} {contract.localSymbol} @ ${trade.orderStatus.avgFillPrice:.2f}")
        
        profile = self.config.profiles[0]
        if profile['safety_net']['enabled']:
            trail_order = Order(action="SELL", orderType="TRAIL", totalQuantity=trade.order.totalQuantity, trailingPercent=profile['safety_net']['native_trail_percent'], tif="GTC")
            trail_trade = await self.ib_interface.place_order(contract, trail_order)
            logging.info(f"Attached native trail stop for {contract.localSymbol}")
        else:
            trail_trade = None

        position_data = {
            'contract': contract, 'profile': profile, 'entry_price': trade.orderStatus.avgFillPrice,
            'quantity': trade.order.totalQuantity, 'native_trail_order_id': trail_trade.order.orderId if trail_trade else None
        }
        self.open_positions[contract.conId] = position_data
        self.state_manager.save_positions(self.open_positions)
        await self.ib_interface.subscribe_to_market_data(contract)
        await self._update_position_data_cache(contract)
        trade.filledEvent -= self._on_order_filled

    async def execute_close_trade(self, contract, reason="Dynamic Exit"):
        conId = contract.conId
        if conId not in self.open_positions: return
        logging.info(f"Executing close for {contract.localSymbol} due to: {reason}")
        position_data = self.open_positions[conId]
        
        close_order = Order(action="SELL", orderType="MKT", totalQuantity=position_data['quantity'])
        await self.ib_interface.place_order(contract, close_order)
        
        await self.ib_interface.unsubscribe_from_market_data(contract)
        del self.open_positions[conId]
        if conId in self.position_data_cache: del self.position_data_cache[conId]
        self.state_manager.save_positions(self.open_positions)
        await self.telegram_interface.send_message(f"⚫ STC Executed: Closed {position_data['quantity']} {contract.localSymbol} due to {reason}.")

    async def _update_position_data_cache(self, contract):
        df = await self.ib_interface.get_historical_data(contract, duration='1 D', bar_size='1 min')
        if df is not None and not df.empty:
            self.position_data_cache[contract.conId] = {'bars': df, 'ticks': []}
            logging.info(f"Populated data cache for {contract.localSymbol} with {len(df)} bars.")

    def _append_tick_to_cache(self, ticker):
        conId = ticker.contract.conId
        if conId in self.position_data_cache:
            self.position_data_cache[conId]['ticks'].append(ticker)

    def _resample_ticks_to_bar(self, contract: Contract) -> bool:
        conId = contract.conId
        cache = self.position_data_cache.get(conId)
        if not cache or not cache['ticks']: return False
        
        now_utc = datetime.now(timezone.utc)
        last_bar_time = cache['bars'].index[-1]
        
        if (now_utc - last_bar_time).total_seconds() >= 60:
            ticks_in_bar = cache['ticks']
            prices = [t.last for t in ticks_in_bar if t.last > 0]
            if not prices: 
                cache['ticks'] = []
                return False
            
            # Create a new bar DataFrame and set its name
            new_bar = pd.DataFrame({
                'open': [prices[0]], 'high': [max(prices)],
                'low': [min(prices)], 'close': [prices[-1]]
            }, index=[last_bar_time + pd.Timedelta(minutes=1)])
            
            # Concatenate and update the cache
            cache['bars'] = pd.concat([cache['bars'], new_bar])
            cache['ticks'] = []
            
            logging.debug(f"New 1-min bar created for {contract.localSymbol}")
            return True
        return False
        
    async def _veto_trade(self, signal, profile, reason):
        logging.warning(f"Trade VETOED for {signal['ticker']}. Reason: {reason}")
        option_str = f"{signal['strike']}{signal['right'][0]}"
        expiry_str = datetime.strptime(signal['expiry'], '%Y%m%d').strftime('%Y-%m-%d')
        veto_message = (f"❌ **Trade Vetoed** ❌\n\n"
                        f"**Ticker:** `{signal['ticker']}`\n"
                        f"**Option:** `{option_str}`\n"
                        f"**Expiry:** `{expiry_str}`\n"
                        f"**Source:** `{profile['channel_name']}`\n\n"
                        f"**Reason:** {reason}")
        await self.telegram_interface.send_message(veto_message)

    def _is_eod(self):
        now = datetime.now()
        eod_time = now.replace(hour=self.config.eod_close['hour'], minute=self.config.eod_close['minute'], second=0, microsecond=0)
        return now >= eod_time

    async def flatten_all_positions(self):
        logging.warning("EOD reached. Flattening all open positions.")
        await self.telegram_interface.send_message("Market close approaching. Flattening all positions.")
        open_conIds = list(self.open_positions.keys())
        for conId in open_conIds:
            contract = self.open_positions[conId]['contract']
            await self.execute_close_trade(contract, reason="EOD Flatten")

    async def shutdown(self):
        logging.info("Shutting down the bot...")
        await self.telegram_interface.send_message("🤖 Bot is shutting down.")
        await self.ib_interface.disconnect()