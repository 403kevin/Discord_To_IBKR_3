import logging
import sys
import asyncio
from datetime import datetime, time as dt_time
import pytz
from collections import deque

# --- MODULE IMPORTS ---
# These are the specialist modules that perform specific jobs.
from services.config import Config
from interfaces.ib_interface import IBInterface
from interfaces.discord_interface import DiscordInterface
from services.sentiment_analyzer import SentimentAnalyzer
from bot_engine.signal_processor import SignalProcessor

# ==============================================================================
# SECTION 1: GLOBAL STATE & SETUP
# ==============================================================================

# In-memory cache to prevent processing the same Discord message ID twice
processed_message_ids = deque(maxlen=50)

# Dictionary to hold live trade data, managed by the main application loop
live_positions = {}


# ==============================================================================
# SECTION 2: CORE LOGIC FUNCTIONS
# ==============================================================================

def get_live_price(ticker):
    """
    Safely retrieves the last known price for a contract from the data manager.
    Prefers 'ask' for entries, falls back to 'last'.
    """
    if not ticker:
        return None
    price = ticker.ask if ticker.ask and ticker.ask > 0 else ticker.last
    return price if price and price > 0 else None


def calculate_trade_quantity(live_price, profile):
    """
    Calculates the number of contracts to trade based on capital allocation.
    """
    funds = profile['trading']['funds_allocation']
    cost_per_contract = live_price * 100
    if cost_per_contract == 0:
        return 0
    quantity = int(funds / cost_per_contract)
    logging.info(
        f"[SIZER] Funds: ${funds}, Price: ${live_price:.2f}, Cost/Contract: ${cost_per_contract:.2f} => Quantity: {quantity}")
    return quantity


async def process_signal(signal, ib_interface, market_data_manager, notifier, config):
    """
    Main pipeline for processing a validated trading signal.
    """
    try:
        logging.info(f"[PROCESS] Processing signal: {signal}")
        profile = next((p for p in config.profiles if p['channel_name'] == signal['source_channel']), None)
        if not profile:
            logging.warning(f"[REJECT] No profile found for channel: {signal['source_channel']}")
            return

        # --- 1. Get Contract ---
        contract = await ib_interface.get_option_contract(signal)
        if not contract:
            raise ValueError("Failed to qualify contract from signal.")

        # --- Prevent Re-entry ---
        if contract.conId in live_positions:
            logging.warning(f"[REJECT] Signal for an already live position: {contract.localSymbol}. Ignoring.")
            return

        # --- 2. Sentiment Analysis ---
        score = 'N/A'  # Default score
        if profile['sentiment_filter']['enabled']:
            headlines = await ib_interface.get_news_headlines(contract.symbol)
            sentiment_analyzer = SentimentAnalyzer()
            score = sentiment_analyzer.analyze_sentiment(headlines)

            threshold = profile['sentiment_filter']['sentiment_threshold']
            is_call = signal['right'] == 'C'

            if (is_call and score < threshold) or (not is_call and score > -threshold):
                reason = f"Sentiment score {score:.2f} failed threshold {threshold} for a {signal['right']}"
                logging.warning(f"[VETO] Trade for {contract.localSymbol} vetoed. Reason: {reason}")
                await notifier.send_veto_message(signal, profile, reason, score)
                return

        # --- 3. Pre-Trade Checks & Sizing ---
        await market_data_manager.subscribe_to_contract(contract)
        await asyncio.sleep(2)  # Allow a moment for the price to stream
        ticker = market_data_manager.get_ticker(contract.conId)
        live_price = get_live_price(ticker)

        if not live_price:
            raise ValueError("Could not fetch a valid live price for sizing.")

        if not (profile['trading']['min_price'] <= live_price <= profile['trading']['max_price']):
            reason = f"Live price ${live_price:.2f} is outside the allowed range (${profile['trading']['min_price']:.2f} - ${profile['trading']['max_price']:.2f})"
            logging.warning(f"[REJECT] Trade for {contract.localSymbol} rejected. Reason: {reason}")
            await notifier.send_rejection_message(contract.localSymbol, reason)
            return

        quantity = calculate_trade_quantity(live_price, profile)
        if quantity == 0:
            reason = f"Allocated funds not sufficient to buy 1 contract at ${live_price:.2f}"
            logging.warning(f"[REJECT] Trade for {contract.localSymbol} rejected. Reason: {reason}")
            await notifier.send_rejection_message(contract.localSymbol, reason)
            return

        # --- 4. Place Entry Order & Safety Net ---
        entry_trade = await ib_interface.place_entry_order(contract, quantity)
        logging.info(
            f"[EXECUTE] Entry order placed for {quantity}x {contract.localSymbol}. OrderId: {entry_trade.order.orderId}")

        # Add to live positions immediately to prevent re-entry
        live_positions[contract.conId] = {
            "contract": contract, "profile": profile,
            "entry_price": None, "high_water_mark": 0,
            "status": "pending_fill", "trade_object": entry_trade
        }

        # Wait for fill before placing trail
        fill_confirmed = await ib_interface.wait_for_fill(entry_trade)
        if not fill_confirmed:
            logging.error(
                f"[ERROR] Did not receive fill confirmation for OrderId {entry_trade.order.orderId}. Cancelling.")
            await ib_interface.cancel_order(entry_trade.order)
            del live_positions[contract.conId]
            return

        # Update position with actual fill price
        fill_price = entry_trade.orderStatus.avgFillPrice
        live_positions[contract.conId].update({
            "entry_price": fill_price,
            "high_water_mark": fill_price,
            "status": "live"
        })
        logging.info(f"[FILLED] Filled at ${fill_price:.2f}. Position is now live.")
        await notifier.send_fill_confirmation(entry_trade.fills[0], score, profile['channel_name'])

        # Place the native trail order
        if profile['safety_net']['enabled']:
            await ib_interface.place_native_trail(contract, quantity, profile['safety_net']['native_trail_percent'])

    except Exception as e:
        logging.error(f"[CRITICAL] Unhandled error in process_signal: {e}", exc_info=True)
        await notifier.send_message(
            f"🚨 **Critical Bot Error** 🚨\n\nFailed to process signal.\n`{signal}`\n\nError: `{e}`")


def monitor_positions(live_positions, market_data_manager, ib_interface, notifier):
    """
    The main monitoring loop that runs in the primary async thread.
    """
    for conId, pos_data in list(live_positions.items()):
        if pos_data.get('status') != 'live':
            continue

        ticker = market_data_manager.get_ticker(conId)
        live_price = get_live_price(ticker)
        if not live_price:
            continue

        contract = pos_data['contract']
        profile = pos_data['profile']
        entry_price = pos_data['entry_price']

        # Update high-water mark
        pos_data['high_water_mark'] = max(pos_data['high_water_mark'], live_price)
        high_water_mark = pos_data['high_water_mark']

        # --- Dynamic Exit Logic ---
        exit_strategy = profile['exit_strategy']

        # 1. Breakeven Stop
        if not pos_data.get('breakeven_set'):
            profit_percent = (live_price - entry_price) / entry_price if entry_price != 0 else 0
            if profit_percent >= (exit_strategy['breakeven_trigger_percent'] / 100):
                logging.info(
                    f"[BREAKEVEN] Triggered for {contract.localSymbol}. Moving stop to entry price ${entry_price:.2f}")
                pos_data['breakeven_set'] = True
                # The logic below will now enforce the breakeven stop

        # 2. Trailing Stop Calculation
        stop_price = 0
        if pos_data.get('breakeven_set'):
            stop_price = entry_price
        else:
            pullback_amount = high_water_mark * (exit_strategy['pullback_stop_percent'] / 100)
            stop_price = high_water_mark - pullback_amount

        # 3. Check for exit
        if live_price <= stop_price:
            logging.info(
                f"[EXIT] Trailing stop hit for {contract.localSymbol}. Price: ${live_price:.2f}, Stop: ${stop_price:.2f}")
            # In a real implementation, you'd place a sell order here
            # For now, we'll just log and remove from monitoring
            del live_positions[conId]
            asyncio.create_task(
                notifier.send_message(f"ℹ️ **Position Closed (Trail Stop)** ℹ️\n\n`{contract.localSymbol}`"))
            continue

        logging.info(
            f"[MONITOR] {contract.localSymbol} | Price: {live_price:.2f} | High: {high_water_mark:.2f} | Stop: {stop_price:.2f}")


# ==============================================================================
# SECTION 3: MAIN APPLICATION
# ==============================================================================

async def main():
    """
    The main asynchronous function that runs the bot.
    """
    # Initialize components
    config = Config()

    ib_interface = IBInterface(config)

    discord_interface = DiscordInterface(config)

    try:
        # Connect to IBKR
        await ib_interface.connect()

        # Set the market data manager's price update handler
        ib_interface.ib.pendingTickersEvent += market_data_manager.on_price_update

        logging.info("Main application loop started. Bot is now live.")

        # --- Main Loop ---
        while True:
            # 1. Check for new Discord messages
            new_messages = await discord_interface.poll_new_messages()
            for msg in new_messages:
                msg_id = msg['id']
                if msg_id in processed_message_ids:
                    continue
                processed_message_ids.append(msg_id)

                # 2. Parse the message content
                parser = SignalParser(config)
                signal = parser.parse(msg['content'], msg['channel_name'])

                if signal:
                    # Don't wait for processing, just start the task
                    asyncio.create_task(process_signal(signal, ib_interface, market_data_manager, notifier, config))

            # 3. Monitor live positions
            monitor_positions(live_positions, market_data_manager, ib_interface, notifier)

            # 4. Intelligent wait
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutdown signal received.")
    except Exception as e:
        logging.critical(f"[FATAL] The main application loop has crashed: {e}", exc_info=True)
        await notifier.send_message(f"🚨 **FATAL BOT CRASH** 🚨\n\n`{e}`")
    finally:
        logging.info("Shutting down bot...")
        await ib_interface.disconnect()
        logging.info("Bot has been shut down.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)-5s] [%(module)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure asyncio to use the ProactorEventLoop on Windows, which is
    # required for ib_insync to function correctly in this architecture.
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot terminated by user.")

