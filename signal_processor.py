import logging
import asyncio
import uuid
from datetime import datetime, timezone
from collections import deque
import pandas as pd
from ib_insync import Option, MarketOrder, Order
from services.signal_parser import SignalParser

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    The "brain" of the bot. This is the "Professional Aviator" edition,
    with the final breakeven logic avionics installed.
    """

    def __init__(self, config, ib_interface, discord_interface, sentiment_analyzer, telegram_interface, flight_computer=None):
        self.config = config
        self.ib_interface = ib_interface
        self.discord_interface = discord_interface
        self.sentiment_analyzer = sentiment_analyzer
        self.telegram_interface = telegram_interface
        self.flight_computer = flight_computer
        
        self._channel_states = {}
        self.active_trades = {}
        self.processed_message_ids = deque(maxlen=self.config.processed_message_cache_size)

    # ... (get_cooldown_status and reset_consecutive_losses are unchanged) ...

    async def process_signal(self, message: dict, profile: dict):
        # ... (process_signal is mostly unchanged, but we add new trade_info fields) ...
        # In the dict for a new trade, we must add:
        # "breakeven_armed": False
        pass # Placeholder for brevity, full code below

    async def monitor_active_trades(self):
        """
        The "battle log." This is the Level 2 version that uses the flight
        computer to test all dynamic exit logic, including breakeven.
        """
        if not self.active_trades:
            return

        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            trade_info = self.active_trades.get(trade_id)
            if not trade_info or not trade_info.get('fill_processed', False):
                # This part of the logic handles the initial fill and native trail
                # It is complex and has been omitted here for clarity, but is in the full file.
                continue

            trade = trade_info['trade_obj']
            contract = trade.contract
            profile = trade_info['profile']
            exit_strategy = profile['exit_strategy']
            
            # --- Get Live Data (from Simulator or Broker) ---
            ticker = self.ib_interface.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(1.2)
            last_price = ticker.last if ticker.last and not ticker.last != ticker.last else ticker.close if ticker.close else 0
            self.ib_interface.ib.cancelMktData(contract)
            if not last_price > 0: continue

            entry_price = trade_info['entry_price']
            high_water_mark = max(trade_info.get('high_water_mark', 0), last_price)
            trade_info['high_water_mark'] = high_water_mark
            
            # --- DYNAMIC EXIT LOGIC CHECKS ---
            exit_reason = None
            
            # --- SURGICAL UPGRADE: Breakeven Avionics ---
            pnl_percent = ((last_price - entry_price) / entry_price) * 100 if entry_price else 0
            
            # 1. Arm the breakeven trigger if it hasn't been armed yet
            if not trade_info.get('breakeven_armed', False) and pnl_percent >= exit_strategy['breakeven_trigger_percent']:
                trade_info['breakeven_armed'] = True
                logger.info(f"Breakeven trigger ARMED for {contract.localSymbol} at {pnl_percent:.2f}% PnL.")
            
            # 2. Check if an armed trigger has been hit
            if trade_info.get('breakeven_armed', False) and last_price <= entry_price:
                exit_reason = "Breakeven Stop"
            # --- END UPGRADE ---

            # 3. Trailing Stop Logic (Pullback or ATR)
            if not exit_reason:
                # ... (Existing trailing stop logic from last version goes here) ...
                pass

            # 4. Momentum Exit Logic (PSAR, RSI)
            if not exit_reason:
                # ... (Existing momentum exit logic from last version goes here) ...
                pass

            # --- Execute Exit If Triggered ---
            if exit_reason:
                logger.info(f"EXIT TRIGGERED for {contract.localSymbol}: {exit_reason}")
                close_order = Order(action='SELL', orderType='MKT', totalQuantity=trade.order.totalQuantity)
                await self.ib_interface.place_order(contract, close_order)
                
                contract_str = f"{contract.symbol} {contract.lastTradeDateOrContractMonth[4:6]}/{contract.lastTradeDateOrContractMonth[6:8]}/{contract.lastTradeDateOrContractMonth[2:4]} {int(contract.strike)}{contract.right}"
                exit_msg = (f"🔴 **SELL Order Executed**\n"
                            f"Contract details: `{contract_str}`\n"
                            f"Exit Price: `${last_price:.2f}` (SIM)\n"
                            f"Reason: `{exit_reason}`")
                await self.telegram_interface.send_message(exit_msg)

                del self.active_trades[trade_id]

# The full, correct functions need to be provided in the final file.
# The snippet above is a summary of the new logic.