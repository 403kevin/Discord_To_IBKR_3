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
        self.polling_interval_seconds = 2
        self.delay_between_channels = 2
        self.delay_after_full_cycle = 2
        
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

        # Kept for backward compatibility with jargon_words only
        self.jargon_words = ["SWING", "LONG", "SHORT", "LEAPS", "DAY", "TRADE", "SMALL","HIGH","RISK","RISKY","LOTTO"]
        self.buzzwords = self.jargon_words  # Only jargon now - buy/ignore moved to profiles

        self.daily_expiry_tickers = ["SPX", "SPY", "QQQ", "SPXW"]

        self.eod_close = {
            "enabled": True, #False while testing
            "hour": 13,
            "minute": 30
        }
        
        self.reconciliation_interval_seconds = 60  # Check every 60 seconds for ghost positions

        # =================================================================
        # --- LEGEND: API & CONNECTION SETTINGS ---
        # =================================================================
        self.ibkr_host = "127.0.0.1"
        self.ibkr_port = 7497 #4002 for gateway
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
        # THIS ALREADY EXISTS IN YOUR CONFIG - KEEP IT AS-IS
        self.sentiment_filter = {
            "enabled": False,
            "sentiment_threshold": 0.05,
            "put_sentiment_threshold": -0.05
        }

        # =================================================================
        # --- LEGEND: VIX VOLATILITY FILTER (TradingView Data Source) ---
        # =================================================================
        # ADD THIS NEW SECTION BELOW sentiment_filter
        # No API key required - uses TradingView's public endpoints
        self.vix_filter = {
            "enabled": True,
            
            # Basic VIX range
            "vix_max": 30,      # Don't trade when VIX > 30 (market panic/extreme fear)
            "vix_min": 12,      # Don't trade when VIX < 12 (too calm, small moves expected)
            
            # Caching to avoid hammering TradingView
            "cache_duration": 300,  # Cache VIX for 5 minutes (300 seconds)
            
            # Fallback behavior if TradingView fetch fails
            "fail_open": True,  # True = allow trades if VIX unavailable, False = block trades
            
            # ADVANCED: Multi-metric volatility regime detection
            # Set "enabled": True to use sophisticated vol filtering
            "advanced_metrics": {
                "enabled": True,  # Set True to enable advanced filtering
                
                "vvix_max": 130,   # VIX of VIX - don't trade if volatility itself is too volatile
                                   # Normal range: 70-100, High: 100-130, Extreme: >130
                
                "skew_min": 130,   # CBOE Skew Index - measure of tail risk (crash protection)
                                   # Normal: 130-150, Low (<130) = elevated crash risk
                
                "avoid_backwardation": False,  # True = don't trade when VIX futures in backwardation
                                               # Backwardation = near-term fear > long-term fear (danger)
                                               # Contango = normal market (safe to trade)
            }
        }

        # =================================================================
        # --- LEGEND: SPREAD/LIQUIDITY FILTER ---
        # =================================================================
        # ADD THIS NEW SECTION BELOW vix_filter
        # Uses data already available from IBKR - no external API needed
        self.spread_filter = {
            "enabled": False,
            
            # Bid-ask spread check
            "max_spread_percent": 10,   # Max allowed spread as % of mid price
                                        # Example: $5.00 mid, $0.50 spread = 10%
                                        # Wider spreads = you get ripped off on entry/exit
            
            # Volume/liquidity checks
            "min_volume": 100,          # Minimum daily volume (contracts traded today)
                                        # Low volume = hard to exit, wide spreads
            
            "min_open_interest": 500,   # Minimum open interest (total contracts outstanding)
                                        # Low OI = illiquid, manipulable
        }

        # =================================================================
        # --- LEGEND: TIME-OF-DAY FILTER ---
        # =================================================================
        # ADD THIS NEW SECTION BELOW spread_filter
        # No external data needed - uses system clock
        self.time_filter = {
            "enabled": False,
            
            # Trading window (in market timezone)
            "trading_hours": {
                "start": "09:45",  # Start trading 15 min after market open
                                   # Avoids 9:30-9:45 open volatility/manipulation
                
                "end": "15:45"     # Stop trading 15 min before close
                                   # Avoids 15:45-16:00 close manipulation/pin risk
            },
            
            # Optional: Skip low-volume lunch period
            "blackout_lunch": False,  # Set True to skip 12:00-13:00 EST
                                      # (lower volume, choppy price action)
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
                
                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["RISK", "LOTTO", "EARNINGS", "PLAY", "IGNORE", 
                                     "STC", "SELL", "SOLD", "CLOSE", "TRIM", "TAKING"],
                
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
                        "psar_enabled": False,
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
                
                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["hut"],
                
                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 5,
                    "min_ticks_per_bar": 5,
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
                    "trail_method": "atr",
                    "trail_settings": {
                        "pullback_percent": 10,
                        "atr_period": 5,
                        "atr_multiplier": 3
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 25}
            },

            # ==========
            # Channel 02
            # ==========

            {
                "channel_id": "1240051753861382249",
                "channel_name": "NITRO_1",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["bun"],

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
                        "pullback_percent": 8,
                        "atr_period": 5,
                        "atr_multiplier": 2.5
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 25}
            },

            # ==========
            # Channel 03
            # ==========

            {
                "channel_id": "799402744011292702",
                "channel_name": "Money_Mo",
                "enabled": True,
                "assume_buy_on_ambiguous": True,
                "ambiguous_expiry_enabled": True,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["cheese"],

                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 5,
                    "min_ticks_per_bar": 5,
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
                    "trail_method": "pullback_percent",
                    "trail_settings": {
                        "pullback_percent": 12,
                        "atr_period": 5,
                        "atr_multiplier": 0.5
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 25}
            },

            # ==========
            # Channel 04
            # ==========

            {
                "channel_id": "1005363564296540243",
                "channel_name": "EXPO",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": False,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["hug"],

                "trading": {
                    "funds_allocation": 1000,
                    "min_price_per_contract": 0.30,
                    "max_price_per_contract": 10.0,
                    "entry_order_type": "MKT",
                    "time_in_force": "DAY"
                },

                "exit_strategy": {
                    "breakeven_trigger_percent": 15,
                    "min_ticks_per_bar": 5,
                    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
                    "trail_method": "atr",
                    "trail_settings": {
                        "pullback_percent": 8,
                        "atr_period": 30,
                        "atr_multiplier": 3.0
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 30}
            },

            # ==========
            # Channel 05
            # ==========

            {
                "channel_id": "1196227916032376883",
                "channel_name": "arrow",
                "enabled": False,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["limb"],

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
                        "pullback_percent": 8,
                        "atr_period": 30,
                        "atr_multiplier": 3.0
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
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
                "channel_id": "1330376524498731028",
                "channel_name": "ZEUS",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["limb"],

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
                    "trail_method": "pullback_percent",
                    "trail_settings": {
                        "pullback_percent": 10,
                        "atr_period": 5,
                        "atr_multiplier": 0.5
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 25}
            },

            
            # ==========
            # Channel 07
            # ==========

            {
                "channel_id": "925217078569500723",
                "channel_name": "Prophet_Day",
                "enabled": True,
                "assume_buy_on_ambiguous": False,
                "ambiguous_expiry_enabled": True,

                # Per-channel buzzwords
                "buzzwords_buy": ["BTO", "BUY", "BOUGHT", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES",
                                  "HERE",
                                  "OPENING", "ADDED", "ENTERING", "GRABBED", "POSITION"],
                "buzzwords_ignore": ["limb"],

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
                        "pullback_percent": 8,
                        "atr_period": 5,
                        "atr_multiplier": 0.5
                    },
                    "momentum_exits": {
                        "psar_enabled": False,
                        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
                        "rsi_hook_enabled": False,
                        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
                    }
                },
                "safety_net": {"enabled": True, "native_trail_percent": 25}
            },

        ]
