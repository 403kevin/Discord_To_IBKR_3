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
from datetime import datetime
from ib_insync import Order, Trade

class SignalProcessor:
    """
    Orchestrates the bot's logic, processing signals and managing trades.
    """
    def __init__(self, config, ib_interface, discord_interface, telegram_interface, signal_parser, state_manager):
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.telegram_interface = telegram_interface
        self.signal_parser = signal_parser
        self.state_manager = state_manager
        
        self.open_positions = self.state_manager.load_positions()
        # This cache will hold the historical + real-time data for each open position.
        # Key: contract.conId, Value: pandas.DataFrame
        self.position_data_cache = {}

    async def start(self):
        """
        The main entry point. Sets up concurrent tasks for all bot operations.
        This is the new, asynchronous, multi-tasking engine.
        """
        logging.info("Starting Signal Processor...")
        await self.telegram_interface.send_message("🤖 Bot is starting up...")
        
        try:
            await self.ib_interface.connect()
            await self.discord_interface.initialize_and_login()
            await self._initialize_open_positions()

            # Define the concurrent tasks for the bot to run.
            tasks = [
                self.poll_discord_for_signals(),
                self.process_market_data_stream()
            ]
            await asyncio.gather(*tasks)

        except Exception as e:
            logging.error(f"A critical error occurred in the main start sequence: {e}")
            await self.telegram_interface.send_message(f"🚨 CRITICAL ERROR: {e}. Shutting down.")
        finally:
            await self.shutdown()

    async def _initialize_open_positions(self):
        """
        On startup, subscribe to market data for any positions loaded from the state file.
        """
        if not self.open_positions:
            return
        
        logging.info(f"Initializing {len(self.open_positions)} open positions from state file...")
        for conId, position_data in self.open_positions.items():
            contract = position_data['contract']
            # Re-subscribe to the live data feed for this position.
            await self.ib_interface.subscribe_to_market_data(contract)
            # Pre-fill the data cache for this position.
            await self._update_position_data_cache(contract)
            logging.info(f"Successfully re-initialized and subscribed to data for {contract.localSymbol}")

    async def poll_discord_for_signals(self):
        """
        Task 1: Continuously polls Discord for new trade signals.
        """
        while True:
            if self._is_eod():
                await self.flatten_all_positions()
                break # Exit the loop to allow for graceful shutdown.

            try:
                raw_messages = await self.discord_interface.poll_for_new_messages()
                if raw_messages:
                    # In a multi-profile setup, you would loop through messages and find the right profile.
                    # For now, we assume one profile.
                    profile = self.config.profiles[0] 
                    parsed_signals = self.signal_parser.parse_messages(raw_messages, profile)
                    
                    for signal in parsed_signals:
                        await self.execute_trade_from_signal(signal, profile)

            except Exception as e:
                logging.error(f"Error in Discord polling loop: {e}")
            
            await asyncio.sleep(self.config.polling_interval_seconds)

    async def process_market_data_stream(self):
        """
        Task 2: The "Skilled Pilot". Continuously processes real-time data from IBKR.
        This is the core of the dynamic exit logic.
        """
        logging.info("Starting market data stream processor...")
        while True:
            try:
                ticker = await self.ib_interface.market_data_queue.get()
                conId = ticker.contract.conId
                
                if conId in self.open_positions:
                    # A new price tick has arrived for one of our open positions.
                    # Append it to our data cache.
                    self._append_tick_to_cache(ticker)
                    
                    # Now, evaluate the exit conditions with the new data.
                    profile = self.open_positions[conId]['profile']
                    await self.evaluate_dynamic_exit(ticker.contract, profile)

            except Exception as e:
                logging.error(f"Error in market data processing loop: {e}")

    async def evaluate_dynamic_exit(self, contract, profile):
        """
        Checks the dynamic exit conditions (RSI, PSAR, etc.) for a position.
        """
        if not profile['dynamic_exit']['USE_DYNAMIC_EXIT']:
            return

        conId = contract.conId
        df = self.position_data_cache.get(conId)
        if df is None or len(df) < profile['dynamic_exit']['RSI_PERIOD']:
            return # Not enough data to calculate indicators.

        # --- Calculate Indicators ---
        exit_settings = profile['dynamic_exit']
        # RSI Hook check
        if exit_settings['rsi_hook_enabled']:
            rsi = ta.rsi(df['close'], length=exit_settings['rsi_settings']['period'])
            if rsi is not None and len(rsi) >= 2:
                # RSI Hook: Exit if RSI was overbought and has now crossed back down.
                if rsi.iloc[-2] > exit_settings['rsi_settings']['overbought_level'] and \
                   rsi.iloc[-1] < exit_settings['rsi_settings']['overbought_level']:
                    logging.info(f"RSI Hook exit triggered for {contract.localSymbol}!")
                    await self.execute_close_trade(contract, "RSI Hook")
                    return # Exit triggered, no need to check others.
        
        # PSAR Flip check
        if exit_settings['psar_enabled']:
            psar = ta.psar(df['high'], df['low'], **exit_settings['psar_settings'])
            if psar is not None:
                # Get the last PSAR direction value (psarl)
                last_psar_long = psar[f'PSARl_{exit_settings["psar_settings"]["start"]}_{exit_settings["psar_settings"]["max"]}'].iloc[-1]
                last_close = df['close'].iloc[-1]
                # If last_psar_long is NaN, it means we are in a short trend. A flip to long is an exit for a PUT.
                # A PSAR flip for a call would be when the price crosses below the PSAR value.
                # This logic needs to be tailored to the position direction (Call/Put).
                # For now, a simple crossover:
                if last_close < last_psar_long:
                     logging.info(f"PSAR Flip exit triggered for {contract.localSymbol}!")
                     await self.execute_close_trade(contract, "PSAR Flip")
                     return


    async def execute_trade_from_signal(self, signal, profile):
        """Validates and executes a trade based on a parsed signal."""
        # Here you would implement logic for opening a trade (BTO) or closing one (STC)
        # For now, we focus on BTO (Buy to Open)
        if signal['action'] == 'BTO':
            # --- Pre-flight checks ---
            # TODO: Add checks for price limits, capital allocation etc.
            
            # --- Get Contract ---
            contract = await self.ib_interface.get_contract(
                signal['ticker'], 'OPTION', 
                expiry=signal['expiry'], 
                strike=signal['strike'], 
                right=signal['right']
            )
            if not contract:
                return

            # --- Place Order ---
            # TODO: Calculate quantity based on funds_allocation
            quantity = 1 # Placeholder
            order = Order(
                action="BUY", 
                orderType=profile['trading']['entry_order_type'], 
                totalQuantity=quantity,
                tif=profile['trading']['time_in_force']
            )
            trade = await self.ib_interface.place_order(contract, order)
            
            # --- Monitor Fill ---
            # This is a simplified fill check. A more robust system would handle partial fills.
            trade.filledEvent += self._on_order_filled
            await asyncio.sleep(profile['trading']['fill_timeout_seconds'])
            if trade.orderStatus.status != 'Filled':
                 logging.warning(f"Order for {contract.localSymbol} did not fill in time. Canceling.")
                 await self.ib_interface.cancel_order(trade.order)
                 trade.filledEvent -= self._on_order_filled
    
    async def _on_order_filled(self, trade: Trade):
        """Callback for when a BTO order is successfully filled."""
        contract = trade.contract
        logging.info(f"TRADE FILLED: {contract.localSymbol} @ {trade.orderStatus.avgFillPrice}")
        await self.telegram_interface.send_message(f"✅ BTO FILLED: {trade.order.totalQuantity} {contract.localSymbol} @ ${trade.orderStatus.avgFillPrice}")
        
        # Immediately attach the native trail stop-loss
        profile = self.config.profiles[0] # Assuming one profile for now
        if profile['safety_net']['enabled']:
            trail_order = Order(
                action="SELL",
                orderType="TRAIL",
                totalQuantity=trade.order.totalQuantity,
                trailingPercent=profile['safety_net']['native_trail_percent'],
                tif="GTC" # Good 'til Canceled
            )
            trail_trade = await self.ib_interface.place_order(contract, trail_order)
            logging.info(f"Attached native trail stop for {contract.localSymbol} (Order ID: {trail_trade.order.orderId})")
        else:
            trail_trade = None

        # --- Update State and Subscribe to Data ---
        position_data = {
            'contract': contract,
            'profile': profile,
            'entry_price': trade.orderStatus.avgFillPrice,
            'quantity': trade.order.totalQuantity,
            'native_trail_order_id': trail_trade.order.orderId if trail_trade else None
        }
        self.open_positions[contract.conId] = position_data
        self.state_manager.save_positions(self.open_positions)
        
        # Subscribe to the real-time data feed for this new position.
        await self.ib_interface.subscribe_to_market_data(contract)
        # Pre-fill the data cache.
        await self._update_position_data_cache(contract)
        
        # Clean up the event listener
        trade.filledEvent -= self._on_order_filled

    async def execute_close_trade(self, contract, reason="Dynamic Exit"):
        """Closes an open position based on a dynamic exit signal."""
        conId = contract.conId
        if conId not in self.open_positions:
            return

        logging.info(f"Executing close for {contract.localSymbol} due to: {reason}")
        position_data = self.open_positions[conId]
        
        # --- Critical Step: Cancel the native trail first! ---
        # TODO: Implement logic to get the trail order object from state and cancel it.
        # native_trail_order = ...
        # await self.ib_interface.cancel_order(native_trail_order)

        # --- Place Market Order to Close ---
        close_order = Order(
            action="SELL",
            orderType="MKT",
            totalQuantity=position_data['quantity']
        )
        close_trade = await self.ib_interface.place_order(contract, close_order)
        
        # --- Clean Up ---
        await self.ib_interface.unsubscribe_from_market_data(contract)
        del self.open_positions[conId]
        if conId in self.position_data_cache:
            del self.position_data_cache[conId]
        self.state_manager.save_positions(self.open_positions)
        
        await self.telegram_interface.send_message(f"⚫ STC Executed: Closed {contract.localSymbol} due to {reason}.")

    # --- Data Cache Management ---
    async def _update_position_data_cache(self, contract):
        """Fetches initial historical data to populate the cache for a position."""
        df = await self.ib_interface.get_historical_data(contract, duration='1 D', bar_size='1 min')
        if df is not None and not df.empty:
            self.position_data_cache[contract.conId] = df
            logging.info(f"Successfully populated data cache for {contract.localSymbol} with {len(df)} bars.")

    def _append_tick_to_cache(self, ticker):
        """Appends a new real-time tick to the cached DataFrame."""
        conId = ticker.contract.conId
        if conId not in self.position_data_cache:
            return # Should not happen if initialized correctly

        # This is a simplified approach. A more robust solution would resample
        # ticks into bars (e.g., 1-minute candles). For now, we append.
        new_row = {
            'date': datetime.now(),
            'open': ticker.last,
            'high': ticker.last,
            'low': ticker.last,
            'close': ticker.last,
            'volume': ticker.volume
        }
        
        # Convert row to DataFrame and append
        new_df_row = pd.DataFrame([new_row]).set_index('date')
        self.position_data_cache[conId] = pd.concat([self.position_data_cache[conId], new_df_row])
        # Trim the cache to prevent it from growing indefinitely
        self.position_data_cache[conId] = self.position_data_cache[conId].iloc[-1000:]


    # --- Utility and Shutdown ---
    def _is_eod(self):
        """Checks if it's past the End-of-Day closing time."""
        now = datetime.now()
        eod_time = now.replace(hour=self.config.eod_close['hour'], 
                               minute=self.config.eod_close['minute'], 
                               second=0, microsecond=0)
        return now >= eod_time

    async def flatten_all_positions(self):
        """Closes all open positions."""
        logging.warning("EOD reached. Flattening all open positions.")
        await self.telegram_interface.send_message("Market close approaching. Flattening all positions.")
        
        # Create a copy of keys to iterate over, as the dict will be modified.
        open_conIds = list(self.open_positions.keys())
        for conId in open_conIds:
            contract = self.open_positions[conId]['contract']
            await self.execute_close_trade(contract, reason="EOD Flatten")

    async def shutdown(self):
        """Gracefully shuts down the bot."""
        logging.info("Shutting down the bot...")
        await self.telegram_interface.send_message("🤖 Bot is shutting down.")
        await self.ib_interface.disconnect()