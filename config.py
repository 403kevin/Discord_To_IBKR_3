# services/config.py
import os
from dotenv import load_dotenv

# This line finds and loads the variables from your .env file.
load_dotenv()


class Config:
    """
    Central configuration class for the trading bot. This is the final,
    definitive version with capital-based position sizing.
    """

    def __init__(self):
        # =================================================================
        # --- GLOBAL BOT SETTINGS ---
        # =================================================================
        self.polling_interval_seconds = 10
        self.delay_between_channels = 2
        self.delay_after_full_cycle = 6
        self.signal_max_age_seconds = 60

        self.buzzwords_buy = ["BTO", "BUY"]
        self.buzzwords_sell = ["STC", "SELL"]
        self.buzzwords = self.buzzwords_buy + self.buzzwords_sell

        self.master_shutdown_enabled = True
        self.master_shutdown_channel_id = "YOUR_PRIVATE_DISCORD_CHANNEL_ID"
        self.master_shutdown_command = "terminate"
        self.oversold_monitor_enabled = False

        # =================================================================
        # --- API & CONNECTION SETTINGS ---
        # =================================================================
        self.ibkr_host = "127.0.0.1"
        self.ibkr_port = 7497 #4001 for IB Gateway
        self.ibkr_client_id = 1

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_user_token = os.getenv("DISCORD_AUTH_TOKEN")

        if not all([self.telegram_bot_token, self.telegram_chat_id, self.discord_user_token]):
            raise ValueError(
                "One or more required environment variables (tokens/IDs) are missing. Please check your .env file.")

        self.sentiment_filter = {
            "enabled": False,
            "headlines_to_fetch": 20,
            "sentiment_threshold": 0.05
        }

        # =================================================================
        # --- CHANNEL PROFILES ---
        # =================================================================
        self.profiles = [
            {
                "channel_id": "1392531225348014180",
                "channel_name": "Test Server",
                "enabled": True,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                "consecutive_loss_monitor": {"enabled": True, "max_losses": 3, "cooldown_minutes": 60},

                # --- MODIFIED: Capital-Based Trade Sizing & Filters ---
                "trading": {
                    # The amount of capital to allocate for each new position.
                    # The bot will calculate the number of contracts that can be bought with this amount.
                    "funds_allocation": 1000,

                    # Pre-trade price filters. The bot will fetch the price before buying.
                    "min_price_per_contract": 0.30,  # e.g., $30 per contract
                    "max_price_per_contract": 10.0,  # e.g., $1000 per contract

                    "entry_order_type": "MKT",
                    "fill_timeout_seconds": 20,
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 15,
                    "timeout_exit_minutes": 15,
                    "trail_method": "atr",
                    "trail_settings": {"percentage": 10, "atr_period": 14, "atr_multiplier": 1.5},
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 50}
            },
        ]

