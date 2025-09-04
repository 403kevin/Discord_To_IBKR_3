# services/config.py
import os
from dotenv import load_dotenv

# This command finds the .env file in your project root and loads all the
# variables from it into the environment, making them accessible to our script.
load_dotenv()


class Config:
    """
    Central configuration class for the trading bot.
    This file acts as the single source of truth for all settings,
    API keys, and trading strategies.
    """

    def __init__(self):
        # =================================================================
        # --- LEGEND: API & CONNECTION SETTINGS ---
        # All external service credentials and connection parameters go here.
        # Secrets are securely loaded from your local .env file.
        # =================================================================
        self.ibkr_host = "127.0.0.1"  # IP address of the machine running TWS/Gateway.
        self.ibkr_port = 7497  # 7497 for Trader Workstation (TWS), 4001 for IB Gateway.
        self.ibkr_client_id = 1  # A unique ID for this bot's connection.

        # --- Securely loads secrets from your .env file ---
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_user_token = os.getenv("DISCORD_AUTH_TOKEN")  # Note: Key matches your .env file

        # --- Crucial check to ensure secrets were loaded correctly ---
        if not all([self.telegram_bot_token, self.telegram_chat_id, self.discord_user_token]):
            raise ValueError(
                "CRITICAL ERROR: One or more API tokens/keys are missing. Please ensure your .env file is in the project root and contains DISCORD_AUTH_TOKEN, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID.")

        # =================================================================
        # --- LEGEND: GLOBAL BOT SETTINGS ---
        # These settings control the bot's high-level operational behavior.
        # =================================================================
        # --- NEW: Polling Speed Configuration ---
        # A value between 1-2 seconds is recommended to avoid rate limiting.
        self.delay_between_channels_seconds = 2
        # The delay, in seconds, after the bot has completed a full cycle of polling
        # all channels. A value between 5-10 seconds is recommended.
        self.delay_after_full_cycle_seconds = 6

        # --- Master Shutdown Command ---
        self.master_shutdown_enabled = True  # Enables the master shutdown command listener.
        self.master_shutdown_channel_id = "YOUR_PRIVATE_DISCORD_CHANNEL_ID"
        self.master_shutdown_command = "terminate"  # The word that will shut down the bot.
        self.oversold_monitor_enabled = True  # Enables the emergency flatten safety net.

        # =================================================================
        # --- LEGEND: SENTIMENT ANALYSIS (FINBERT) ---
        # Settings for the pre-trade news sentiment analysis filter.
        # =================================================================
        self.sentiment_filter = {
            "enabled": True,  # Master switch to turn the sentiment filter on or off.
            "headlines_to_fetch": 20,  # The number of recent news headlines to analyze.
            "sentiment_threshold": 0.1  # Vetoes CALLS if score < threshold, Vetoes PUTS if score > -threshold.
        }

        # =================================================================
        # --- LEGEND: CHANNEL PROFILES ---
        # This is the core of the strategy engine. Each dictionary in this
        # list represents a unique trading profile that will be applied to
        # a specific Discord channel. You can have different strategies
        # for different signal providers.
        # =================================================================
        self.profiles = [
            {
                # --- Profile Identification ---
                "channel_id": "YOUR_TARGET_DISCORD_CHANNEL_ID",  # The ID of the Discord channel to monitor.
                "channel_name": "Pro Scalpers",  # A friendly name for logging and notifications.
                "enabled": True,  # Master switch for this entire profile.

                # --- Pre-Trade Signal Filters ---
                "assume_buy_on_ambiguous": True,  # If a signal has no BUY/SELL keyword, assume it's a BUY.
                "reject_if_contains": ["RISK", "earnings", "play"],  # Ignores any signal containing these words.

                # --- Channel-Specific Risk Management ---
                "consecutive_loss_monitor": {
                    "enabled": True,  # If True, stops trading this channel after too many losses.
                    "max_losses": 3,  # Number of consecutive losses to trigger the cooldown.
                    "cooldown_minutes": 60  # How long to pause trading this channel (in minutes).
                },

                # --- Entry Order Configuration ---
                "entry_order_type": "MKT",  # MKT for Market Order. More types can be added later.
                "fill_timeout_seconds": 20,  # How long to wait for a fill confirmation before warning.

                # --- Primary Exit Strategy ---
                "exit_strategy": {
                    # --- Core Exit Rules ---
                    "breakeven_trigger_percent": 15,  # Moves stop to entry price after this % profit is reached.
                    "timeout_exit_minutes": 15,  # Hard time stop; exits the trade after this many minutes.

                    # --- Trailing Stop Method (The bot's primary internal stop) ---
                    "trail_method": "atr",  # "atr" for Average True Range, "percentage" for a fixed %.
                    "trail_settings": {
                        "percentage": 10,  # The fixed % for the "percentage" method.
                        "atr_period": 14,  # The lookback period for calculating ATR.
                        "atr_multiplier": 1.5  # The stop will be placed this many ATRs away from the price.
                    },

                    # --- Optional: Momentum-Based Early Exits ---
                    "momentum_exits": {
                        "psar_enabled": False,  # Exit when the Parabolic SAR indicator flips.
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,  # Exit if RSI crosses down from an overbought level.
                        "rsi_settings": {"period": 14, "overbought_level": 70}
                    }
                },

                # --- Native Safety Net (The broker-side fail-safe) ---
                "safety_net": {
                    "enabled": True,  # If True, attaches a wide native TRAIL order on entry.
                    "native_trail_percent": 50  # The % for the broker-side TRAIL. This is a fail-safe.
                }
            },
            # You can copy the entire dictionary above and paste it here
            # to create a new profile for a different channel_id.
        ]

