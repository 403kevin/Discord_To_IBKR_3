import time
import logging
from typing import Dict

from ib_insync import Order
from interfaces.ib_interface import IBInterface
from interfaces.telegram_interface import TelegramInterface


class TrailingStopManager:
    def __init__(self, ib_interface: IBInterface, telegram_interface: TelegramInterface):
        self.ib = ib_interface
        self.telegram_interface = telegram_interface
        self.active_trails: Dict[str, dict] = {}

    def add_position(self, symbol: str, contract, entry_price: float, qty: int, rules: dict, trader_name: str,
                     strategy_details: str, safety_net_order_id: int = None):
        if not rules or rules.get("type") != "dynamic_trail":
            return

        logging.info(f"[TRAIL ADD] Activating dynamic trail for {qty}x {symbol} for {trader_name}")
        self.ib.subscribe_to_streaming_data(contract)

        self.active_trails[symbol] = {
            'contract': contract, 'entry_price': entry_price, 'qty': qty,
            'highest_price': entry_price, 'breakeven_hit': False, 'start_time': time.time(),
            'rules': rules, 'trader_name': trader_name, 'strategy_details': strategy_details,
            'safety_net_order_id': safety_net_order_id
        }

    def check_trailing_stops(self):
        if not self.active_trails: return
        for symbol in list(self.active_trails.keys()):
            trail = self.active_trails[symbol]
            rules = trail['rules']
            current_price = self.ib.get_price_from_stream(symbol)
            if current_price is None:
                logging.warning(f"[TRAIL] No valid streaming price for {symbol}, skipping check.")
                continue
            logging.info(f"[TRAIL CHECK] {symbol} | Price: {current_price:.2f} | High: {trail['highest_price']:.2f}")

            hard_stop_percent = rules.get("hard_stop_loss_percent")
            if hard_stop_percent:
                hard_stop_price = trail['entry_price'] * (1 - hard_stop_percent / 100)
                if current_price <= hard_stop_price: self.execute_sell(symbol, trail, "hard_stop",
                                                                       current_price); continue

            breakeven_percent = rules.get("breakeven_trigger_percent")
            if breakeven_percent and not trail['breakeven_hit']:
                if current_price >= trail['entry_price'] * (1 + breakeven_percent / 100):
                    trail['breakeven_hit'] = True;
                    logging.info(f"[TRAIL] BREAKEVEN TRIGGERED for {symbol}")

            pullback_percent = rules.get("pullback_stop_percent")
            if pullback_percent and trail['breakeven_hit']:
                if current_price > trail['highest_price']: trail['highest_price'] = current_price
                pullback_stop_price = trail['highest_price'] * (1 - pullback_percent / 100)
                if current_price < pullback_stop_price: self.execute_sell(symbol, trail, "pullback_stop",
                                                                          current_price); continue

            timeout_minutes = rules.get("timeout_exit_minutes")
            if timeout_minutes and (time.time() - trail['start_time']) / 60 > timeout_minutes:
                self.execute_sell(symbol, trail, "timeout", current_price);
                continue

    async def execute_sell(self, symbol: str, trail_info: dict, reason: str, exit_price: float):
        logging.info(f"Executing sell for {symbol} for {trail_info['trader_name']} due to: {reason}")

        if trail_info['safety_net_order_id']:
            self.ib.cancel_order(trail_info['safety_net_order_id'])
            await asyncio.sleep(1)

        self.ib.unsubscribe_from_streaming_data(symbol)
        sell_order = Order(action="SELL", orderType="MKT", totalQuantity=trail_info['qty'])
        self.ib.ib.placeOrder(trail_info['contract'], sell_order)

        notification_message = (
            f"🔴 *SELL Order Executed*\n\n"
            f"*Trader:* `{trail_info['trader_name']}`\n"
            f"*Contract:* `{symbol}`\n"
            f"*Reason:* `{reason.replace('_', ' ').title()}`\n"
            f"*Exit Price:* `${exit_price:.2f}`"
        )
        await self.telegram_interface.send_message(notification_message)

        del self.active_trails[symbol]
