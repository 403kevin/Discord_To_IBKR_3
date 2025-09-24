import os
from dotenv import load_dotenv
from pathlib import Path

# --- GPS for the .env file ---
current_dir = Path(__file__).parent
env_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Central configuration class for the trading bot. This is the definitive,
    final version based on the master GitHub file.
    """

    def __init__(self):
        # =================================================================
        # --- LEGEND: GLOBAL BOT SETTINGS ---
        # =================================================================
        # --- SURGICAL UPGRADE: The "Dashboard" Legend ---
        # This is the bot's heartbeat. It defines the polling cycle.
        # Total cycle time is roughly (delay_between_channels * num_channels) + delay_after_full_cycle.
        # With 1 channel, this is ~8 seconds. With 2, ~10 seconds.
        # WARNING: Setting these too low can risk getting flagged by Discord.
        self.polling_interval_seconds = 3  # Legacy setting, not used by the modern async loop.
        self.delay_between_channels = 2  # Time to wait after checking each channel.
        self.delay_after_full_cycle = 4  # Time to wait after checking ALL channels.

        # This is the granular, per-channel cooldown managed by the SignalProcessor.
        self.DISCORD_COOLDOWN_SECONDS = int(os.getenv("DISCORD_COOLDOWN_SECONDS", 30))



        # =================================================================
        # --- LEGEND: BACKTESTING ENGINE ---
        # =================================================================
        self.backtesting = {
            "lookback_days": 30,  # How many days of historical data to fetch.
            "bar_size": "1 min"  # The resolution of the data (e.g., "1 min", "5 mins", "1 hour").
        }

        self.signal_max_age_seconds = 60
        self.processed_message_cache_size = 25
        self.buzzwords_buy = ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE",
                              "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"]
        self.buzzwords_sell = ["STC", "SELL"]
        self.buzzwords_ignore = os.getenv("IGNORE_WORDS", "ignore,test").split(',') # <-- SURGICAL INSERTION
        self.TRADE_LOG_FILE_PATH = os.getenv("TRADE_LOG_FILE_PATH", "trade_log.csv")


        self.jargon_words = ["SWING", "LONG", "SHORT", "LEAPS", "DAY", "TRADE", "SMALL","HIGH","RISK","RISKY","LOTTO"] #ignores
        self.buzzwords = self.buzzwords_buy + self.buzzwords_sell + self.jargon_words

        self.daily_expiry_tickers = ["SPX", "SPY", "QQQ", "SPXW"]

        self.eod_close = {
            "enabled": True,
            "hour": 13,  # US/Mountain timezone
            "minute": 00
        }


        self.master_shutdown_enabled = False
        self.master_shutdown_channel_id = "1392531225348014180"
        self.master_shutdown_command = "terminate"
        self.oversold_monitor_enabled = True
        
        # =================================================================
        # --- LEGEND: ANALYSIS & RESAMPLING ---
        # =================================================================
        self.min_ticks_per_bar = int(os.getenv("MIN_TICKS_PER_BAR", 20)) # <-- SURGICAL INSERTION

        # =================================================================
        # --- LEGEND: API & CONNECTION SETTINGS ---
        # =================================================================
        self.ibkr_host = "127.0.0.1"
        self.ibkr_port = 7497  # 4002 GATEWAY 7497 TWS
        self.ibkr_client_id = 1

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_user_token = os.getenv("DISCORD_AUTH_TOKEN")

        if not all([self.telegram_bot_token, self.telegram_chat_id, self.discord_user_token]):
            raise ValueError(
                "One or more required environment variables (tokens/IDs) are missing. Please check your .env file.")

        # =================================================================
        # --- LEGEND: SENTIMENT ANALYSIS (VADER) ---
        # =================================================================
        self.sentiment_filter = {
            "enabled": False,
            "headlines_to_fetch": 10,
            "sentiment_threshold": 0.05,
            "put_sentiment_threshold": -0.05 # <-- SURGICAL INSERTION
        }

        # =================================================================
        # --- LEGEND: CHANNEL PROFILES ---
        # =================================================================
        self.profiles = [

            # ===================
            # --- MASTER ---
            # ===================

            {
                "channel_id": "1392531225348014180",
                "channel_name": "test_server",
                "enabled": False,
                "assume_buy_on_ambiguous": False, # NO buy buzzword
                "ambiguous_expiry_enabled": True, # next available expiry
                "reject_if_contains": ["RISK", "earnings", "play"],
                "consecutive_loss_monitor": {"enabled": False, "max_losses": 3, "cooldown_minutes": 60},

                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY",
                    "fill_timeout_seconds": 20
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 10,
                    "timeout_exit_minutes": 120,

                    # --- UPDATED: Graceful Exit Strategy Toggle ---
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"], # <-- SURGICAL INSERTION
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
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30} # <-- SURGICAL INSERTION
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

# Other channels here

        ]