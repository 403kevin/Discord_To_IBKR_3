# =============================================================================================== #
import discord_interface
import config
import polygon
import ib_interface
import time
from math import floor, isnan
import json
import logging
from datetime import date, datetime, timezone, timedelta
from dateutil.parser import parse as parse_dt_from_str
import custom_logger
import colorama
import message_parsers
from collections import defaultdict
from pprint import pprint
from decimal import Decimal
from trailing_stop_manager import TrailingStopManager
import threading


# =============================================================================================== #

colorama.init()
RUNTIME_LOG_FILE = 'runtime.log'

# =============================================================================================== #

class Main:
    def __init__(self):
        self.dc_client = discord_interface.DiscordChannelClient(config.DISCORD_AUTH_TOKEN)
        self.ib_interface = ib_interface.IBInterface()
        self.parser = getattr(message_parsers, config.CHANNEL_INFO['parser'])()

        self.portfolio_state = {}  # FIXED
        self.trailing_manager = TrailingStopManager(self.ib_interface, self.portfolio_state)

        self.last_signal_id = read_last_signal_log_id()
        self.qty_map = defaultdict(lambda *args: 0)
        self.current_state = {}
        self.BUY_SIGNALS = config.BUY_SIGNALS
        self.SELL_SIGNALS = config.SELL_SIGNALS
        self.TRIM_SIGNALS = config.TRIM_SIGNALS
        self.REJECT_SIGNALS = config.REJECT_SIGNALS
        self.FORMAT_12_BUY = config.FORMAT_12_BUY
        self.MIN_PRICE = config.MIN_PRICE
        self.MAX_PRICE = config.MAX_PRICE
        EXIT_TIME = datetime.now()
        self.EXIT_TIME = EXIT_TIME.replace(hour=config.EXIT_HOUR, minute=config.EXIT_MINUTE, second=0, microsecond=0)
        self.EXIT_TIME = int(self.EXIT_TIME.timestamp())

        if config.TRAILING_STOP_ENABLED:
            threading.Thread(target=self.run_trailing_loop, daemon=True).start()

        logging.info(f'Discord and IB clients initiated')

    def run(self):
        logging.info(f'Initiating worker loop... BEHOLD!!')

        while 1:
            current_datetime = datetime.now()
            current_timestamp = int(current_datetime.timestamp())

            if self.EXIT_TIME < current_timestamp:
                logging.info(f'system stopped due to exit time mentioned in config')
                return

            c_dt = current_dt()
            all_signals = self.dc_client.poll_new_messages(config.CHANNEL_INFO['channel_id'], 50)

            signals = [signal for signal in all_signals
                       if Decimal(signal['id']) > Decimal(self.last_signal_id)
                       and ((c_dt - parse_dt_from_str(signal['timestamp']).replace(tzinfo=timezone.utc)).total_seconds()
                            < config.ALERT_EXPIRY_DURATION)
                       and c_dt > parse_dt_from_str(signal['timestamp']).replace(tzinfo=timezone.utc)]

            if not signals:
                time.sleep(config.SLEEP_DELAY_BETWEEN_POLLS)
                continue

            self.last_signal_id = signals[0]['id']

            if not config.TEST_MODE:
                update_last_signal_log_id(Decimal(self.last_signal_id))

            signals.reverse()
            time.sleep(config.SLEEP_DELAY_BETWEEN_POLLS)

            for signal in signals:
                msg_time = parse_dt_from_str(signal['timestamp']).replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - msg_time > timedelta(seconds=config.SIGNAL_MAX_AGE_SECONDS):
                    print(f"[INFO] Skipping stale signal {signal['id']}")
                    continue
                try:
                    self.process_signal(signal)
                except Exception as _exc:
                    logging.error(f'Exception processing signal. exc: {str(_exc)} | signal: {signal}')
            time.sleep(config.SLEEP_DELAY_BETWEEN_POLLS)

    def run_trailing_loop(self):
        while True:
            try:
                self.trailing_manager.check_trailing_stops()
            except Exception as e:
                logging.error(f"[TRAIL LOOP ERROR] {e}")
            time.sleep(5)  # Check every 5 seconds

    def process_signal(self, signal: dict):
        signal_id = signal['id']

        if len(signal['embeds']) != 1:
            logging.info(f'Processing new signal #{signal["id"]}. signal content: {signal["content"]}')

        if config.ONE_CONTRACT_AT_A_TIME:
            positions = self.ib_interface.get_positions()
            positions = [position for position in positions if position.position != 0]
            if len(positions) == 0:
                signal = self.parser.parse_message(self, signal, state=self.current_state)
            else:
                signal = self.parser.parse_message(self, signal, state=self.current_state)
                if signal != {}:
                    logging.info(f'Skipped signal due to already open positions.ONE_CONTRACT_AT_A_TIME is enabled')
                    return
                else:
                    logging.info(f'Open position closed as positions.ONE_CONTRACT_AT_A_TIME is enabled')
                    return
        else:
            signal = self.parser.parse_message(self, signal, state=self.current_state)

        if signal == {}:
            return

        pprint(signal)

        if config.USE_BRAKET_ORDER and signal['instr'] != 'BUY':
            logging.warning(f'Skipping non-buy signal #{signal["id"]} since we are strictly using '
                            f'our Own position management')
            return
        if config.TRAILING_STOP_ENABLED and signal['instr'] != 'BUY':
            logging.warning(f'Skipping non-buy signal #{signal["id"]} since we have TRAILING_STOP_ENABLED from config')
            return

        if signal['instr'] in ['SMALL', 'HEDGE']:
            logging.warning(f'Skipping signal #{signal["id"]} since it is a {signal["instr"]} signal')
            return

        given_month = signal['exp_month']
        given_day = signal['exp_day']
        current_date = date.today()

        given_date = date(current_date.year, given_month, given_day)

        if given_date < current_date:
            year_for_option_symbol = date(current_date.year + 1, given_month, given_day).year
        else:
            year_for_option_symbol = current_date.year

        option_symbol = polygon.build_option_symbol(signal['underlying'],
                                                    date(year_for_option_symbol, signal['exp_month'],
                                                         signal['exp_day']), signal['p_or_c'], signal['strike'])
        parsed_symbol = polygon.parse_option_symbol(option_symbol)

        if signal['underlying'] in config.RESTRICTED_SYMBOLS:
            logging.warning(f'Skipping signal #{signal["id"]} since the underlying is in restricted symbols')
            return

        contract = self.ib_interface.create_contract(parsed_symbol)
        logging.info(f"[DEBUG] Created IB contract: {contract}")
        price, contract = self.ib_interface.get_realtime_price(contract, asset_type='option',
                                                               parsed_symbol=parsed_symbol)
        logging.info(f"[DEBUG] Real-time market price for {contract.localSymbol}: {price}")

        if not price or price <= 0 or isnan(price):
            logging.warning(f'Skipping signal #{signal["id"]} due to failure in getting real '
                            f'time price for {option_symbol}')
            return

        if not self.MIN_PRICE <= price <= self.MAX_PRICE:
            logging.warning(f'Skipping signal #{signal["id"]} due to price is not within limit '
                            f'time price for {option_symbol}')
            return

        if signal['instr'] in ['TRIM', 'SELL']:
            self.process_sell_or_trim_signal(signal, parsed_symbol)
            return

        qty = floor(config.PER_SIGNAL_FUNDS_ALLOCATION / (price * 100))
        if signal['instr'] != 'ADD':
            qty = floor(config.PER_ADD_SIGNAL_FUNDS_ALLOCATION / (price * 100))

        logging.info(f"[DEBUG] Quantity calc -> price: {price}, qty: {qty}, instr: {signal['instr']}")

        logging.info(f'signal #{signal["id"]} | option {option_symbol} | mkt price from IB: {price} | '
                     f'calculated qty: {qty} | funds allocated: {config.PER_SIGNAL_FUNDS_ALLOCATION}')

        if qty <= 0 or isnan(qty):
            logging.warning(f'Skipping signal #{signal["id"]} option {option_symbol} due to '
                            f'calculated qty being <= 0 | qty: {qty}')
            return

        adaptive_algo_priority = None
        if config.USE_OPTION_ADAPTIVE_ALGO:
            adaptive_algo_priority = config.ADAPTIVE_PRIORITY_TYPE

        order = {'asset_type': 'option',
                 'parsed_symbol': parsed_symbol,
                 'qty': qty,
                 'tp': round(price * (1 + config.TAKE_PROFIT_PERCENTAGE / 100), 1),
                 'sl': round(price * (1 - config.STOP_LOSS_PERCENTAGE / 100), 1)}

        logging.info(f"[DEBUG] Submitting {'bracket' if config.USE_BRAKET_ORDER else 'market'} order: {order}")

        if config.USE_BRAKET_ORDER:
            mkt_trade = self.ib_interface.submit_bracket_order_order(order, adaptive_algo_priority)
        else:
            mkt_trade = self.ib_interface.submit_buy_market_order(order, adaptive_algo_priority)

        while mkt_trade.isActive():
            self.ib_interface.ib.waitOnUpdate()

        logging.info(f'signal #{signal["id"]} market order filled')
        self.qty_map[parsed_symbol.underlying_symbol] += qty
        self.current_state[parsed_symbol.underlying_symbol] = {
    'option_symbol': option_symbol,
    'logged_at': time.time(),
    'msg_id': signal['id'],
    'qty': qty
}

        if config.TRAILING_STOP_ENABLED:
            if config.USE_ADVANCED_TRAILING:
                self.trailing_manager.add_position(
                    symbol=option_symbol,
                    entry_price=price,
                    contract=contract,
                    order={
                        'underlying': parsed_symbol.underlying_symbol,
                        'exp_month': parsed_symbol.expiration_date.month,
                        'exp_day': parsed_symbol.expiration_date.day,
                        'strike': parsed_symbol.strike_price,
                        'p_or_c': parsed_symbol.option_type,
                        'qty': qty
                    },
                    log_prefix=f"signal #{signal['id']}"
                )

            else:
                order2 = {'asset_type': 'option', 'underlying': parsed_symbol.underlying_symbol,
                          'parsed_symbol': parsed_symbol, 'qty': qty, 'trail_percent': config.TRAILING_STOP_PERCENT}
                self.ib_interface.submit_trailing_stop_order(order2)

        self.ib_interface.unsub_market_data(contract)
        return

    def process_sell_or_trim_signal(self, signal: dict, parsed_symbol: polygon.OptionSymbol):
        logging.info(f'signal #{signal["id"]} is a sell signal | symbol: {parsed_symbol}')
        current_qty = self.qty_map[parsed_symbol.underlying_symbol]

        if not current_qty or current_qty <= 0:
            logging.warning(f'Skipping signal #{signal["id"]} since we do not have any position '
                            f'for {parsed_symbol}')
            return

        adaptive_algo_priority = None
        if config.USE_OPTION_ADAPTIVE_ALGO:
            adaptive_algo_priority = config.ADAPTIVE_PRIORITY_TYPE

        current_qty = current_qty if signal['instr'] == 'SELL' else floor(current_qty * config.PERCENT_TO_TRIM / 100)

        self.ib_interface.submit_sell_market_order({'asset_type': 'option', 'parsed_symbol': parsed_symbol,
                                                    'qty': current_qty}, adaptive_algo_priority)

        self.qty_map[parsed_symbol.underlying_symbol] -= current_qty

        if signal['instr'] == 'SELL':
            self.current_state[parsed_symbol.underlying_symbol] = {
    'option_symbol': option_symbol,
    'logged_at': time.time(),
    'msg_id': signal['id'],
    'qty': qty
}

# =============================================================================================== #

def read_last_signal_log_id() -> int:
    try:
        with open('last_log_id.json') as _file:
            return int(json.load(_file)['last_log_id'])
    except (FileNotFoundError, json.decoder.JSONDecodeError, KeyError):
        update_last_signal_log_id(0)
        return 0

def update_last_signal_log_id(log_id: int):
    with open('last_log_id.json', 'w') as _file:
        json.dump({'last_log_id': int(log_id)}, _file)

def current_dt() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

# =============================================================================================== #

if __name__ == '__main__':
    custom_logger.setup_logging(console_log_output="stdout", console_log_level="info",
                                console_log_color=True, logfile_file=RUNTIME_LOG_FILE,
                                logfile_log_level="info", logfile_log_color=False,
                                log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s]"
                                                  " %(message)s%(color_off)s")
    logging.info(f'===================================================================\n')
    main = Main()
    main.run()

# =============================================================================================== #
