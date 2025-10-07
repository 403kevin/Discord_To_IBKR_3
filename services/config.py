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
        self.polling_interval_seconds = 3
        self.delay_between_channels = 2
        self.delay_after_full_cycle = 4
        
        self.MARKET_TIMEZONE = "US/Mountain" 

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
        self.cooldown_after_trade_seconds = 30  # Global pause after any fill
        self.processed_message_cache_size = 1000
        
        self.STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "state/open_positions.json")
        self.TRADE_LOG_FILE_PATH = os.getenv("TRADE_LOG_FILE_PATH", "logs/trade_log.csv")

        # FIX: Consolidated buzzword logic
        # BUY words trigger entries
        self.buzzwords_buy = ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE",
                              "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"]
        
        # SELL words moved to IGNORE - we have smart exits, don't want false signals
        # Any message with these words will be rejected
        self.buzzwords_ignore = ["RISK", "LOTTO", "EARNINGS", "PLAY", "IGNORE", 
                                 "STC", "SELL", "SOLD", "CLOSE", "TRIM", "TAKING"]
        
        self.jargon_words = ["SWING", "LONG", "SHORT", "LEAPS", "DAY", "TRADE", "SMALL","HIGH","RISK","RISKY","LOTTO"]
        
        # Only BUY buzzwords are used for parsing now
        self.buzzwords = self.buzzwords_buy + self.jargon_words

        self.daily_expiry_tickers = ["SPX", "SPY", "QQQ", "SPXW"]

        self.eod_close = {
            "enabled": False,
            "hour": 14,
            "minute": 00
        }
        
        # FIX: Reconciliation interval now actually used by signal_processor
        self.reconciliation_interval_seconds = 60  # Check every 60 seconds for ghost positions

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
            "sentiment_threshold": 0.05,
            "put_sentiment_threshold": -0.05
        }

        # =================================================================
        # --- MASTER ---
        # =================================================================
        self.profiles = [
            {
                "channel_id": "1392531225348014180",
                "channel_name": "test_server",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 10,
                    "min_ticks_per_bar": 5,
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
                    
                    "trail_method": "atr", # ** pullback_percent or atr **

                    "trail_settings": {
                        "pullback_percent": 10,
                        "atr_period": 14,
                        "atr_multiplier": 1.5
                    },
                    "momentum_exits": {
                        "psar_enabled": True,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },
            
            # ==========
            # Channel 01
            # ==========
            
            {
                "channel_id": "916540757236678706",
                "channel_name": "QIQO_1",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 02
            # ==========
            
            {
                "channel_id": "1330376524498731028",
                "channel_name": "Zeus_2",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": False,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 03
            # ==========
            
            {
                "channel_id": "1196227916032376883",
                "channel_name": "Arrow_3",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": False,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 04
            # ==========
            
            {
                "channel_id": "852526152252784651",
                "channel_name": "Goldman_4",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 05
            # ==========
            
            {
                "channel_id": "1322024595712118814",
                "channel_name": "Mitro_5",
                "enabled": True,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 06
            # ==========
            
            {
                "channel_id": "1240051753861382249",
                "channel_name": "Nitri_6",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 07
            # ==========
            
            {
                "channel_id": "925217078569500723",
                "channel_name": "Prophet_day_7",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 08
            # ==========
            
            {
                "channel_id": "1081319049771876372",
                "channel_name": "Flint_8",
                "enabled": True,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 09
            # ==========
            
            {
                "channel_id": "1260988811832328263",
                "channel_name": "DD_Alerts_9",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 10
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_10",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 11
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_11",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 12
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_12",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 13
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_13",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 14
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_14",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },

            # ==========
            # Channel 15
            # ==========
            
            {
                "channel_id": "REPLACE_WITH_CHANNEL_ID",
                "channel_name": "Channel_15",
                "enabled": False,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,
                "reject_if_contains": ["RISK", "earnings", "play"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
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
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 35}
            },
        ]
