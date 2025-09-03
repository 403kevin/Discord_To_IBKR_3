# services/config.py

class Config:
    """
    Central configuration class for the trading bot.
    All settings, profiles, and strategies are defined here.
    """
    def __init__(self):
        # =================================================================
        # --- GLOBAL BOT SETTINGS ---
        # =================================================================

        # How often the bot checks Discord for new messages, in seconds.
        self.polling_interval_seconds = 10

        # Master shutdown command listener
        self.master_shutdown_enabled = True
        self.master_shutdown_channel_id = "YOUR_PRIVATE_DISCORD_CHANNEL_ID"
        self.master_shutdown_command = "terminate"

        # Global safety monitor for oversold/negative positions
        self.oversold_monitor_enabled = True

        # =================================================================
        # --- IBKR CONNECTION SETTINGS ---
        # =================================================================
        self.ibkr_host = "127.0.0.1"
        self.ibkr_port = 7497
        self.ibkr_client_id = 1

        # =================================================================
        # --- TELEGRAM NOTIFIER SETTINGS ---
        # =================================================================
        self.telegram_bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
        self.telegram_chat_id = "YOUR_TELEGRAM_CHAT_ID"

        # =================================================================
        # --- CHANNEL PROFILES ---
        # =================================================================
        # Each dictionary in this list is a separate trading profile.
        # The bot will apply these specific rules to the given channel_id.
        self.profiles = [
            {
                "channel_id": "YOUR_TARGET_DISCORD_CHANNEL_ID", # Must be a string
                "channel_name": "Pro Scalpers",
                "enabled": True,

                "assume_buy_on_ambiguous": True,
                "reject_if_contains": ["RISK", "earnings", "play"],

                "consecutive_loss_monitor": {
                    "enabled": True,
                    "max_losses": 3,
                    "cooldown_minutes": 60 # Stop trading this channel for 1 hour after 3 losses
                },

                # --- Entry Order Configuration ---
                "entry_order_type": "MKT",  # Options: "MKT" for now
                "fill_timeout_seconds": 20, # How long to wait for a fill confirmation

                # --- Exit Strategy Configuration ---
                "exit_strategy": {
                    # This is the primary, bot-managed trailing stop.
                    "breakeven_trigger_percent": 15,
                    "timeout_exit_minutes": 15,

                    # --- Trailing Stop Method ---
                    "trail_method": "atr",  # Options: "atr", "percentage"
                    "trail_settings": {
                        "percentage": 10,
                        "atr_period": 14,       # Standard ATR setting
                        "atr_multiplier": 1.5
                    },

                    # --- Momentum-Based Early Exits (Optional) ---
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {
                            "start": 0.02,
                            "increment": 0.02,
                            "max": 0.2
                        },
                        "rsi_hook_enabled": False,
                        "rsi_settings": {
                            "period": 14,
                            "overbought_level": 70
                        }
                    }
                },

                # --- Native Safety Net (Fail-Safe) ---
                # This is a wide, broker-side trail order attached on entry.
                # It only triggers if the bot crashes or the primary exit fails.
                "safety_net": {
                    "enabled": True,
                    "native_trail_percent": 50
                }
            },
            # You can add another profile for another channel here
        ]

        # =================================================================
        # --- SENTIMENT ANALYSIS (FINBERT) ---
        # =================================================================
        self.sentiment_filter = {
            "enabled": True,
            "headlines_to_fetch": 20,
            "sentiment_threshold": 0.1 # Veto CALLS if score is below this, veto PUTS if score is above -this.
        }
