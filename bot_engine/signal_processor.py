import asyncio
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

class SignalProcessor:
    """
    The main brain of the bot. Orchestrates the entire lifecycle of a trade,
    from signal ingestion to real-time position management and execution.
    """
    def __init__(self, config, ib_interface, telegram_interface, signal_parser, state_manager, sentiment_analyzer):
        self.config = config
        self.ib_interface = ib_interface
        self.telegram_interface = telegram_interface
        self.signal_parser = signal_parser
        self.state_manager = state_manager
        self.sentiment_analyzer = sentiment_analyzer
        
        # In-memory caches for real-time operations
        self.position_data_cache = {}  # {conId: {"df": DataFrame, "ticks": [], "last_bar": datetime}}
        self.active_trades = {}  # {conId: trade_object_from_ib}
        self.processed_message_ids = set()

    async def start(self):
        """The main entry point. Sets up concurrent tasks for all bot operations."""
        logging.info("Starting Signal Processor...")
        await self.telegram_interface.send_message("🤖 Bot is ONLINE.")
        
        # Load existing open positions from state file
        await self._load_positions_from_state()

        tasks = [
            self.poll_discord_for_signals(),
            self.process_market_data_stream()
        ]
        await asyncio.gather(*tasks)

    async def _load_positions_from_state(self):
        """On startup, load positions from state and subscribe to their data."""
        open_positions = self.state_manager.get_open_positions()
        logging.info("Loading %d open positions from state file...", len(open_positions))
        for pos in open_positions:
            contract = pos['contract']
            self.active_trades[contract.conId] = pos # Store the whole position dict
            await self._initialize_position_cache(contract)
            await self.ib_interface.subscribe_to_market_data(contract)
            logging.info("Successfully re-initialized and subscribed to data for open position: %s", contract.localSymbol)


    async def poll_discord_for_signals(self):
        """Continuously polls Discord for new signals based on channel profiles."""
        # This function would contain the logic to iterate through discord channels
        # For brevity, this is a conceptual placeholder for the polling loop.
        # In a real implementation, it would call a method like self.discord_interface.poll()
        while True:
            # Placeholder for actual Discord polling logic
            await asyncio.sleep(self.config.polling_interval_seconds)


    async def process_market_data_stream(self):
        """Continuously processes real-time data from the IBKR queue."""
        while True:
            try:
                ticker = await self.ib_interface.market_data_queue.get()
                conId = ticker.contract.conId

                if conId in self.position_data_cache:
                    await self._append_tick_to_cache(ticker)
                    await self.evaluate_dynamic_exit(ticker.contract)

            except Exception as e:
                logging.error("Error in market data processing loop: %s", e)

    async def execute_trade_from_signal(self, raw_signal, profile):
        """The main workflow for processing a single parsed signal."""
        parsed_signal, raw_message = self.signal_parser.parse_signal(raw_signal, profile)
        if not parsed_signal:
            return

        # --- SENTIMENT ANALYSIS PRE-FLIGHT CHECK ---
        if profile.get('sentiment_filter', {}).get('enabled', False):
            sentiment_score = self.sentiment_analyzer.get_sentiment_score(raw_message)
            is_call = parsed_signal['option_type'] == 'Call'
            
            # Bi-directional sentiment logic
            if is_call and sentiment_score < profile['sentiment_filter']['sentiment_threshold']:
                reason = f"Sentiment score {sentiment_score:.4f} is below threshold {profile['sentiment_filter']['sentiment_threshold']} for a CALL."
                await self._notify_trade_veto(parsed_signal, profile, reason)
                return
            elif not is_call and sentiment_score > profile['sentiment_filter']['put_sentiment_threshold']:
                reason = f"Sentiment score {sentiment_score:.4f} is above threshold {profile['sentiment_filter']['put_sentiment_threshold']} for a PUT."
                await self._notify_trade_veto(parsed_signal, profile, reason)
                return

        # --- POSITION SIZING ---
        contract = await self.ib_interface.get_contract_details(
            parsed_signal['ticker'], parsed_signal['option_type'], 
            parsed_signal['strike'], parsed_signal['expiry_str']
        )
        if not contract:
            return

        live_ticker = await self.ib_interface.get_live_ticker(contract)
        if not live_ticker:
            logging.error("Could not get live ticker for %s, cannot calculate position size.", contract.localSymbol)
            return

        ask_price = live_ticker.ask
        trade_params = profile['trading']
        
        if not (trade_params['min_price_per_contract'] <= ask_price <= trade_params['max_price_per_contract']):
            logging.warning("Trade VETOED for %s: Ask price $%.2f is outside configured limits.", contract.localSymbol, ask_price)
            return

        quantity = int(trade_params['funds_allocation'] / (ask_price * 100))
        if quantity == 0:
            logging.warning("Trade VETOED for %s: Insufficient funds allocation for 1 contract at ask price $%.2f.", contract.localSymbol, ask_price)
            return

        # --- EXECUTION ---
        trade = await self.ib_interface.place_order(contract, parsed_signal['action'], quantity)
        if trade:
            self.active_trades[contract.conId] = trade
            trade.filledEvent += self._on_order_filled

    async def _on_order_filled(self, trade, fill):
        """Callback triggered when a trade order is filled."""
        contract = trade.contract
        logging.info("ORDER FILLED: %s %d %s @ %.2f", contract.symbol, fill.execution.shares, "BUY" if fill.execution.side == "BOT" else "SELL", fill.execution.price)
        
        position_details = {
            "contract": contract,
            "entry_price": fill.execution.price,
            "quantity": fill.execution.shares,
            "entry_timestamp": datetime.now()
        }
        self.state_manager.add_position(position_details)
        
        # HARDENED: Initialize cache and subscribe to data, confirming success.
        await self._initialize_position_cache(contract)
        subscription_success = await self.ib_interface.subscribe_to_market_data(contract)

        fill_message = (
            f"✅ **Trade Filled** ✅\n\n"
            f"**Ticker:** `{contract.symbol}`\n"
            f"**Contract:** `{contract.localSymbol}`\n"
            f"**Action:** `{'BUY' if trade.order.action == 'BUY' else 'SELL'}`\n"
            f"**Quantity:** `{int(trade.order.totalQuantity)}`\n"
            f"**Avg Price:** `${fill.execution.price:.2f}`\n\n"
            f"**Data Stream:** `{'Subscribed' if subscription_success else 'FAILED'}`"
        )
        await self.telegram_interface.send_message(fill_message)

    async def _initialize_position_cache(self, contract):
        """Fetches initial historical data for a new position."""
        # Fetch 2 days of data to ensure enough bars for indicators
        df = await self.ib_interface.get_historical_data(contract, duration='2 D', bar_size='1 min')
        self.position_data_cache[contract.conId] = {
            "df": df if df is not None else pd.DataFrame(),
            "ticks": [],
            "last_bar_timestamp": df.iloc[-1]['date'] if df is not None and not df.empty else datetime.now()
        }

    async def _append_tick_to_cache(self, ticker):
        """Appends a new tick and triggers resampling if a new bar is formed."""
        conId = ticker.contract.conId
        cache = self.position_data_cache[conId]
        
        current_time = datetime.now()
        
        cache["ticks"].append({
            "time": current_time,
            "price": ticker.last
        })
        
        # Check if a minute has passed since the last bar was created
        if current_time >= cache["last_bar_timestamp"] + timedelta(minutes=1):
            await self._resample_ticks_to_bar(conId)

    async def _resample_ticks_to_bar(self, conId):
        """QUALITY CONTROL: Creates a new 1-minute bar from collected ticks."""
        cache = self.position_data_cache[conId]
        ticks_df = pd.DataFrame(cache["ticks"])

        # Quality Control: Only create a bar if there's a minimum number of ticks
        if len(ticks_df) < self.config.min_ticks_per_bar:
            logging.debug("Skipping bar creation for conId %d, not enough ticks (%d)", conId, len(ticks_df))
            # Clear ticks so we don't re-evaluate the same small sample
            cache["ticks"] = []
            cache["last_bar_timestamp"] = datetime.now() # Update timestamp to wait for next interval
            return

        ticks_df.set_index('time', inplace=True)
        
        # Resample to 1-minute bars
        resampled = ticks_df['price'].resample('1Min').ohlc()
        if not resampled.empty:
            new_bar = resampled.iloc[-1]
            new_bar_timestamp = new_bar.name.to_pydatetime()

            new_row = pd.DataFrame([{
                "date": new_bar_timestamp,
                "open": new_bar.open,
                "high": new_bar.high,
                "low": new_bar.low,
                "close": new_bar.close,
                "volume": ticks_df['price'].resample('1Min').count().iloc[-1] # Tick count as volume
            }])
            
            # Append new bar to the main DataFrame
            cache["df"] = pd.concat([cache["df"], new_row], ignore_index=True)
            
            # Clear ticks for the next bar
            cache["ticks"] = []
            cache["last_bar_timestamp"] = new_bar_timestamp
            logging.info("New 1-min bar created for conId %d. Total bars: %d", conId, len(cache["df"]))


    async def evaluate_dynamic_exit(self, contract):
        """HIERARCHY: Evaluates all configured exit conditions in the specified priority order."""
        conId = contract.conId
        position = self.state_manager.get_open_positions(conId) # Assumes get by conId
        profile = self.get_profile_for_position(position) # Assumes helper function
        df = self.position_data_cache[conId]["df"]

        if df.empty or len(df) < profile['exit_strategy']['trail_settings']['atr_period']:
            return # Not enough data to evaluate

        is_call = contract.right == 'C'
        current_price = df.iloc[-1]['close']

        # --- Breakeven Logic ---
        breakeven_pct = profile['exit_strategy']['breakeven_trigger_percent']
        entry_price = position['entry_price']
        profit_pct = (current_price - entry_price) / entry_price * 100

        # This is a conceptual placeholder. Real implementation would require managing a breakeven order state.
        if profit_pct >= breakeven_pct:
            # TODO: Add logic to place/modify a stop order to the entry price
            logging.info("Breakeven triggered for %s.", contract.localSymbol)


        # --- Configurable Exit Priority Loop ---
        for exit_type in profile['exit_strategy']['exit_priority']:
            if exit_type == "atr_trail":
                # Logic for ATR Trail
                pass
            elif exit_type == "pullback_stop":
                # Logic for Pullback Stop
                pass
            elif exit_type == "rsi_hook":
                df.ta.rsi(length=profile['rsi_settings']['period'], append=True)
                last_rsi = df.iloc[-1][f'RSI_{profile["rsi_settings"]["period"]}']
                prev_rsi = df.iloc[-2][f'RSI_{profile["rsi_settings"]["period"]}']
                
                if is_call and last_rsi < profile['rsi_settings']['overbought_level'] and prev_rsi >= profile['rsi_settings']['overbought_level']:
                    await self.execute_close_trade(contract, "RSI Hook Exit (Overbought)")
                    return
                elif not is_call and last_rsi > profile['rsi_settings']['oversold_level'] and prev_rsi <= profile['rsi_settings']['oversold_level']:
                    await self.execute_close_trade(contract, "RSI Hook Exit (Oversold)")
                    return
            # ... add other exit types like psar_flip
    
    async def execute_close_trade(self, contract, reason):
        """Executes a market order to close a position."""
        position = next((p for p in self.state_manager.get_open_positions() if p['contract'].conId == contract.conId), None)
        if not position:
            logging.warning("Attempted to close a position that is not in state: %s", contract.localSymbol)
            return

        quantity = position['quantity']
        trade = await self.ib_interface.place_order(contract, "SELL", quantity)
        
        if trade:
            await self.ib_interface.unsubscribe_from_market_data(contract)
            self.state_manager.remove_position(contract.conId)
            del self.position_data_cache[contract.conId]
            
            close_message = (
                f"🔴 **Position Closed** 🔴\n\n"
                f"**Ticker:** `{contract.symbol}`\n"
                f"**Contract:** `{contract.localSymbol}`\n"
                f"**Reason:** `{reason}`"
            )
            await self.telegram_interface.send_message(close_message)
            trade.filledEvent += lambda t, f: logging.info("CLOSE ORDER FILLED for %s @ %.2f", t.contract.localSymbol, f.execution.price)

    async def _notify_trade_veto(self, parsed_signal, profile, reason):
        """Sends a formatted Telegram message when a trade is vetoed."""
        veto_message = (
            f"❌ **Trade Vetoed** ❌\n\n"
            f"**Ticker:** `{parsed_signal['ticker']}`\n"
            f"**Option:** `{parsed_signal['strike']}{'C' if parsed_signal['option_type'] == 'Call' else 'P'}`\n"
            f"**Expiry:** `{parsed_signal['expiry_str']}`\n"
            f"**Source:** `{profile['channel_name']}`\n\n"
            f"**Reason:** {reason}"
        )
        await self.telegram_interface.send_message(veto_message)
        logging.info("Trade vetoed for %s. Reason: %s", parsed_signal['ticker'], reason)

    # Helper function to find the profile for a given position
    def get_profile_for_position(self, position):
        # This is a conceptual placeholder. You'd need a way to link a position
        # back to the profile that created it, perhaps by storing channel_id with the position.
        return self.config.profiles[0]