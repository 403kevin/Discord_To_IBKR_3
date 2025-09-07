# services/config.py
import os
from dotenv import load_dotenv

# This line finds and loads the variables from your .env file.
load_dotenv()


class Config:
    """
    Central configuration class for the trading bot.
    This file acts as the single source of truth for all settings,
    API keys, and trading strategies. It securely loads API keys
    from a .env file.
    """

    def __init__(self):
        # =================================================================
        # --- LEGEND: GLOBAL BOT SETTINGS ---
        # These settings control the bot's high-level operational behavior.
        # =================================================================
        self.polling_interval_seconds = 10

        # A value between 1-2 seconds is recommended to avoid rate limiting.
        self.delay_between_channels = 2

        # The delay, in seconds, after the bot has completed a full cycle.
        self.delay_after_full_cycle = 6

        self.signal_max_age_seconds = 60

        self.master_shutdown_enabled = True
        self.master_shutdown_channel_id = "1392531225348014180"
        self.master_shutdown_command = "terminate"
        self.oversold_monitor_enabled = True

        # =================================================================
        # --- LEGEND: API & CONNECTION SETTINGS ---
        # All external service credentials and connection parameters are
        # now securely loaded from your local .env file.
        # =================================================================
        self.ibkr_host = "127.0.0.1"  # IP address of the machine running TWS/Gateway.
        self.ibkr_port = 7497  # 7497 for Trader Workstation (TWS), 4001 for IB Gateway.
        self.ibkr_client_id = 1  # A unique ID for this bot's connection.

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_user_token = os.getenv("DISCORD_AUTH_TOKEN")

        if not all([self.telegram_bot_token, self.telegram_chat_id, self.discord_user_token]):
            raise ValueError(
                "One or more required environment variables (tokens/IDs) are missing. Please check your .env file.")

        # =================================================================
        # --- LEGEND: SENTIMENT ANALYSIS (VADER) ---
        # MODIFIED: This now reflects the lightweight VADER engine.
        # Settings for the pre-trade news sentiment analysis filter.
        # =================================================================
        self.sentiment_filter = {
            "enabled": True,  # Master switch to turn the sentiment filter on or off.
            "headlines_to_fetch": 20,  # The number of recent news headlines to analyze.
            "sentiment_threshold": 0.05  # VADER's compound score is also -1 to +1.
            # Vetoes CALLS if score < threshold, Vetoes PUTS if score > -threshold.
        }

        # =================================================================
        # --- LEGEND: CHANNEL PROFILES ---
        # The core of the strategy engine. Each dictionary represents a
        # unique trading profile for a specific Discord channel.
        # =================================================================
        self.profiles = [
            {
                # --- Profile Identification ---
                "channel_id": "1392531225348014180",
                "channel_name": "Test Server",
                "enabled": True,

                # --- Pre-Trade Signal Filters ---
                "assume_buy_on_ambiguous": True,
                "reject_if_contains": ["RISK", "earnings", "play"],

                # --- Channel-Specific Risk Management ---
                "consecutive_loss_monitor": {
                    "enabled": True,
                    "max_losses": 3,
                    "cooldown_minutes": 60
                },

                # --- Entry Order Configuration ---
                "entry_order_type": "MKT",
                "fill_timeout_seconds": 20,

                # --- Primary Exit Strategy ---
                "exit_strategy": {
                    "breakeven_trigger_percent": 15,
                    "timeout_exit_minutes": 15,

                    # --- Trailing Stop Method ---
                    "trail_method": "atr",
                    "trail_settings": {
                        "percentage": 10,
                        "atr_period": 14,
                        "atr_multiplier": 1.5
                    },

                    # --- Optional: Momentum-Based Early Exits ---
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70}
                    }
                },

                # --- Native Safety Net (Fail-Safe) ---
                "safety_net": {
                    "enabled": True,
                    "native_trail_percent": 25
                }
            },
        ]

