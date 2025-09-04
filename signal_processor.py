# bot_engine/signal_processor.py
import logging
from datetime import datetime, timezone

# Import project modules
from services.message_parsers import MessageParser

class SignalProcessor:
    """
    The Decision Maker. This class is the central nervous system for trade
    decisions. It takes a raw message, runs it through a series of validation
    gates, and if everything passes, hands it off to the trade_executor.
    """
    def __init__(self, config, sentiment_analyzer, trade_executor, channel_states, state_lock):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.trade_executor = trade_executor
        self.parser = MessageParser(config)
        self.channel_states = channel_states
        self.state_lock = state_lock

    def process_message(self, message_data):
        """
        The main entry point for processing a new message from Discord.
        This function contains the multi-gate validation logic.
        """
        channel_id = message_data["channel_id"]
        message_content = message_data["content"]
        
        # === Gate 0: Kill Switch Check ===
        # Is this channel currently on a cooldown?
        with self.state_lock:
            state = self.channel_states.get(channel_id)
            if state and state["cooldown_until"] and datetime.now(timezone.utc) < state["cooldown_until"]:
                logging.info(f"Signal from channel {channel_id} ignored. Channel is on cooldown until {state['cooldown_until']}.")
                return # Stop processing immediately

        # === Gate 1: Profile Check ===
        # Do we have a valid, enabled profile for this channel?
        profile = self._get_profile_for_channel(channel_id)
        if not profile:
            return # No active profile for this channel, ignore.

        # === Gate 2: Keyword Filter ===
        # Does the message contain any words that should cause a rejection?
        for reject_word in profile.get("reject_if_contains", []):
            if reject_word.lower() in message_content.lower():
                logging.info(f"Signal rejected from {profile['channel_name']}. Contains reject word: '{reject_word}'.")
                return

        # === Gate 3: Translation Check ===
        # Pass the profile to the parser so it knows how to handle ambiguity.
        signal = self.parser.parse(message_content, profile)
        if not signal:
            logging.debug(f"Message from {profile['channel_name']} did not parse into a valid signal.")
            return
        
        logging.info(f"Successfully parsed signal from {profile['channel_name']}: {signal}")

        # === Gate 4: Sentiment Check ===
        # If enabled, does the news sentiment support this trade?
        sentiment_score = 0.0
        if self.config.sentiment_filter["enabled"]:
            headlines = self.trade_executor.ib_interface.get_news_headlines(signal["symbol"])
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(headlines)

            trade_veto = False
            threshold = self.config.sentiment_filter["sentiment_threshold"]
            if signal['right'] == 'C' and sentiment_score < threshold:
                trade_veto = True
                reason = f"Sentiment score {sentiment_score:.2f} is below threshold {threshold} for a CALL."
            elif signal['right'] == 'P' and sentiment_score > -threshold:
                trade_veto = True
                reason = f"Sentiment score {sentiment_score:.2f} is above threshold {-threshold} for a PUT."

            if trade_veto:
                logging.warning(f"TRADE VETOED for {signal['symbol']} {signal['strike']}{signal['right']}. Reason: {reason}")
                self.trade_executor.notifier.send_message(f"❌ *Trade Vetoed* ❌\nSymbol: `{signal['symbol']} {signal['strike']}{signal['right']}`\nReason: {reason}")
                return

        signal["sentiment_score"] = sentiment_score

        # === Gate 5: Final Approval ===
        # If all gates passed, hand off to the Trader for execution.
        logging.info(f"Signal approved. Handing off to trade executor: {signal}")
        self.trade_executor.execute_trade(signal, profile)

    def _get_profile_for_channel(self, channel_id):
        """Finds the active profile for a given channel ID."""
        for profile in self.config.profiles:
            if str(profile["channel_id"]) == str(channel_id) and profile["enabled"]:
                return profile
        return None

