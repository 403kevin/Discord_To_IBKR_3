# services/config.py
import os
from dotenv import load_dotenv

# This line finds and loads the variables from your .env file.
load_dotenv()

class Config:
    """
    Central configuration class for the trading bot. This is the definitive,
    final version based on the master GitHub file.
    """
    def __init__(self):
        # =================================================================
        # --- LEGEND: GLOBAL BOT SETTINGS ---
        # =================================================================
        self.polling_interval_seconds = 10
        self.delay_between_channels = 2
        self.delay_after_full_cycle = 6
        self.signal_max_age_seconds = 60

        # --- NEW: Short-Term Memory ---
        # Remembers the last N message IDs to prevent double-processing in a race condition.
        self.processed_message_cache_size = 25

        self.buzzwords_buy = ["BTO", "BUY"]
        self.buzzwords_sell = ["STC", "SELL"]
        self.buzzwords = self.buzzwords_buy + self.buzzwords_sell

        self.pre_market_trading = {
            "enabled": True,
            "# NOTE: Times are in US/Eastern timezone": "",
            "start_time": "04:00",
            "end_time": "09:30",
            "symbols": ["SPX", "SPY"],
            "trade_quantity": 1
        }

        self.master_shutdown_enabled = True
        self.master_shutdown_channel_id = "YOUR_PRIVATE_DISCORD_CHANNEL_ID"
        self.master_shutdown_command = "terminate"
        self.oversold_monitor_enabled = True

        # =================================================================
        # --- LEGEND: API & CONNECTION SETTINGS ---
        # =================================================================
        self.ibkr_host = "127.0.0.1"
        self.ibkr_port = 7497
        self.ibkr_client_id = 1

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_user_token = os.getenv("DISCORD_AUTH_TOKEN")

        if not all([self.telegram_bot_token, self.telegram_chat_id, self.discord_user_token]):
            raise ValueError("One or more required environment variables (tokens/IDs) are missing. Please check your .env file.")

        # =================================================================
        # --- LEGEND: SENTIMENT ANALYSIS (VADER) ---
        # =================================================================
        self.sentiment_filter = {
            "enabled": True,
            "headlines_to_fetch": 20,
            "sentiment_threshold": 0.05
        }

        # =================================================================
        # --- LEGEND: CHANNEL PROFILES ---
        # =================================================================
        self.profiles = [
            {
                "channel_id": "YOUR_TARGET_DISCORD_CHANNEL_ID",
                "channel_name": "Test Server",
                "enabled": True,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                "consecutive_loss_monitor": { "enabled": True, "max_losses": 3, "cooldown_minutes": 60 },

                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY",
                    "fill_timeout_seconds": 20
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 15,
                    "timeout_exit_minutes": 30,
                    
                    # --- UPDATED: Graceful Exit Strategy Toggle ---
                    # Defines the primary trailing stop method.
                    # Options: "atr", "pullback_percent"
                    "trail_method": "atr",
                    
                    "trail_settings": {
                        "pullback_percent": 10,
                        "atr_period": 14,
                        "atr_multiplier": 1.5
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": { "start": 0.02, "increment": 0.02, "max": 0.2 },
                        "rsi_hook_enabled": False,
                        "rsi_settings": { "period": 14, "overbought_level": 70 }
                    }
                },
                "safety_net": { "enabled": True, "native_trail_percent": 50 }
            },
        ]

