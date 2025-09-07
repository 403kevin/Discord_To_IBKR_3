# bot_engine/signal_processor.py
import logging
from datetime import datetime, timezone

# Import project modules
from services.message_parsers import MessageParser

class SignalProcessor:
    """
    The Decision Maker. This is the final version with enhanced,
    detailed notifications for vetoed trades.
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
        """
        channel_id = message_data["channel_id"]
        message_content = message_data["content"]
        
        # Gate 0: Kill Switch Check
        with self.state_lock:
            state = self.channel_states.get(channel_id)
            if state and state["cooldown_until"] and datetime.now(timezone.utc) < state["cooldown_until"]:
                logging.info(f"Signal from channel {channel_id} ignored. Channel is on cooldown.")
                return

        # Gate 1: Profile Check
        profile = self._get_profile_for_channel(channel_id)
        if not profile:
            return

        # Gate 2: Keyword Filter
        for reject_word in profile.get("reject_if_contains", []):
            if reject_word.lower() in message_content.lower():
                logging.info(f"Signal rejected from {profile['channel_name']}. Contains reject word: '{reject_word}'.")
                return

        # Gate 3: Translation Check
        signal = self.parser.parse(message_content, profile)
        if not signal:
            logging.debug(f"Message from {profile['channel_name']} did not parse into a valid signal.")
            return
        
        logging.info(f"Successfully parsed signal from {profile['channel_name']}: {signal}")

        # Gate 4: Sentiment Check
        sentiment_score = 0.0
        if self.config.sentiment_filter["enabled"]:
            headlines = self.trade_executor.ib_interface.get_news_headlines(signal["symbol"])
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(headlines)

            trade_veto = False
            threshold = self.config.sentiment_filter["sentiment_threshold"]
            reason = ""
            if signal['right'] == 'C' and sentiment_score < threshold:
                trade_veto = True
                reason = f"Sentiment score {sentiment_score:.4f} is below threshold {threshold} for a CALL."
            elif signal['right'] == 'P' and sentiment_score > -threshold:
                trade_veto = True
                reason = f"Sentiment score {sentiment_score:.4f} is above threshold {-threshold} for a PUT."

            if trade_veto:
                logging.warning(f"TRADE VETOED for {signal['symbol']} {signal['strike']}{signal['right']}. Reason: {reason}")
                
                # --- THIS IS THE UPGRADED NOTIFICATION ---
                veto_message = (
                    f"❌ *Trade Vetoed* ❌\n\n"
                    f"*Ticker:* `{signal['symbol']}`\n"
                    f"*Option:* `{signal['strike']}{signal['right']}`\n"
                    f"*Expiry:* `{signal['expiry']}`\n"
                    f"*Source Channel:* `{profile['channel_name']}`\n\n"
                    f"*Reason:* {reason}"
                )
                self.trade_executor.notifier.send_message(veto_message)
                return

        signal["sentiment_score"] = sentiment_score

        # Gate 5: Final Approval
        logging.info(f"Signal approved. Handing off to trade executor: {signal}")
        self.trade_executor.execute_trade(signal, profile)

    def _get_profile_for_channel(self, channel_id):
        """Finds the active profile for a given channel ID."""
        for profile in self.config.profiles:
            if str(profile["channel_id"]) == str(channel_id) and profile["enabled"]:
                return profile
        return None

