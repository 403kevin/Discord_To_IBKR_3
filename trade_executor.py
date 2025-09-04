# bot_engine/trade_executor.py
import logging


def execute_trade(signal: dict, profile: dict, ib_interface, notifier, position_monitor):
    """
    Executes a fully validated trade signal.
    This is the final step before an order is placed.
    """
    symbol = signal['symbol']
    strike = signal['strike']
    right = signal['right']
    expiry = signal['expiry']
    quantity = profile['trading']['trade_quantity']

    try:
        # --- Step 1: Get the Official Contract ---
        contract = ib_interface.get_option_contract(symbol, strike, right, expiry)
        if not contract:
            raise ValueError(f"Could not find a valid contract for {symbol} {expiry} {strike}{right}.")

        # Prevent duplicate entries if a signal is sent multiple times
        if position_monitor.is_position_active(contract.conId):
            logging.warning(
                f"Duplicate signal received for an already active position: {contract.localSymbol}. Ignoring.")
            return

        # --- Step 2: Execute Proven Entry Logic ---
        entry_trade = ib_interface.place_entry_order(
            contract,
            quantity,
            profile['entry_order_type']
        )

        # --- Step 3: Attach Native Safety Net (if enabled) ---
        safety_net_config = profile.get('safety_net', {})
        if safety_net_config.get('enabled', False):
            ib_interface.place_native_trail_stop(
                entry_trade,
                safety_net_config.get('native_trail_percent', 50)
            )

        # --- Step 4: Notify the Watchtower ---
        # The position monitor is now officially guarding this trade.
        position_monitor.add_new_position(
            conId=contract.conId,
            contract=contract,
            profile=profile,
            entry_trade=entry_trade,
            sentiment_score=signal.get('sentiment_score', None)  # <-- PASS THE SCORE
        )

        logging.info(f"Trade for {contract.localSymbol} executed and is now being monitored.")

        # --- Step 5: Set up the Confirmation Handler ---
        # Define what happens when we get a confirmed fill from the broker.
        def on_fill(trade, fill):
            entry_price = fill.execution.price

            # Update the monitor with the official entry price
            position_monitor.update_position_on_fill(contract.conId, entry_price)

            # Send the final, detailed Telegram notification
            score_text = f"\n*Sentiment Score:* `{signal.get('sentiment_score', 'N/A'):.4f}`" if signal.get(
                'sentiment_score') is not None else ""
            message = (
                f"✅ **Trade Entry Confirmed** ✅\n\n"
                f"*Symbol:* `{fill.contract.localSymbol}`\n"
                f"*Quantity:* `{int(fill.execution.shares)}`\n"
                f"*Entry Price:* `${entry_price:.2f}`\n"
                f"*Source Channel:* `{profile['channel_name']}`"
                f"{score_text}"
            )
            notifier.send_message(message)

        # Connect the handler to the trade's fill event
        entry_trade.fillEvent += on_fill

    except Exception as e:
        logging.error(f"CRITICAL ERROR during trade execution for {symbol}: {e}", exc_info=True)
        notifier.send_message(f"🚨 **Execution Error** 🚨\n`{symbol} {strike}{right}`\n`Error:` {e}")

