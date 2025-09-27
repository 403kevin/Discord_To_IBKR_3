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
        # --- LEGEND: MASTER CONTROL & SIMULATION ---
        # =================================================================
        self.USE_MOCK_BROKER = os.getenv('USE_MOCK_BROKER', 'False').lower() in ('true', '1', 't')

        # =================================================================
        # --- LEGEND: GLOBAL BOT SETTINGS ---
        # =================================================================
        self.polling_interval_seconds = 3
        self.delay_between_channels = 2
        self.delay_after_full_cycle = 4
        self.DISCORD_COOLDOWN_SECONDS = int(os.getenv("DISCORD_COOLDOWN_SECONDS", 30))
        
        # --- THE "PAUSE BUTTON" ---
        # After a trade is successfully placed, the bot will enter a global cooldown
        # and will not look for any new signals for this duration (in seconds).
        self.cooldown_after_trade_seconds = 300 # 5 minutes

        self.reconciliation_interval_seconds = 300 

        # =================================================================
        # --- LEGEND: BACKTESTING ENGINE ---
        # =================================================================
        self.backtesting = {
            "lookback_days": 30,
            "bar_size": "1 min"
        }

        # =================================================================
        # --- LEGEND: PARSING & STATE MANAGEMENT ---
        # =================================================================
        self.signal_max_age_seconds = 60
        self.processed_message_cache_size = 25
        
        self.STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "state/open_positions.json")
        self.TRADE_LOG_FILE_PATH = os.getenv("TRADE_LOG_FILE_PATH", "logs/trade_log.csv")

        self.buzzwords_buy = ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE",
                              "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"]
        self.buzzwords_sell = ["STC", "SELL"]
        self.buzzwords_ignore = ["IGNORE"]
        self.jargon_words = ["SWING", "LONG", "SHORT", "LEAPS", "DAY", "TRADE", "SMALL","HIGH","RISK","RISKY","LOTTO"]
        self.buzzwords = self.buzzwords_buy + self.buzzwords_sell + self.jargon_words

        self.daily_expiry_tickers = ["SPX", "SPY", "QQQ", "SPXW"]

        self.eod_close = {
            "enabled": True,
            "hour": 14,
            "minute": 00
        }
        self.MARKET_TIMEZONE = "US/Mountain" 

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
            raise ValueError(
                "One or more required environment variables (tokens/IDs) are missing. Please check your .env file.")

        # =================================================================
        # --- LEGEND: SENTIMENT ANALYSIS (VADER) ---
        # =================================================================
        self.sentiment_filter = {
            "enabled": False,
            "headlines_to_fetch": 10,
            "sentiment_threshold": 0.05,
            "put_sentiment_threshold": -0.05
        }

        # =================================================================
        # --- LEGEND: CHANNEL PROFILES ---
        # =================================================================
        self.profiles = [
            {
                "channel_id": "1392531225348014180",
                "channel_name": "test_server",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
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
                    "min_ticks_per_bar": 5,
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
                    
                    "trail_method": "atr",

                    "trail_settings": {
                        "pullback_percent": 10,
                        "atr_period": 14,
                        "atr_multiplier": 1.5
                    },
                    "momentum_exits": {
                        "psar_enabled": True,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": True,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },
        ]