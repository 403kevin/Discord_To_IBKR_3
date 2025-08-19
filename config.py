# ==============================================================================
# SECTION 1: SCRAPER & POLLING SETTINGS
# ==============================================================================

# The delay, in seconds, between polling each individual channel in the list below.
# A value between 1-2 seconds is recommended to avoid rate limiting.
DELAY_BETWEEN_CHANNELS = 1.5

# The delay, in seconds, after the bot has completed a full cycle of polling
# all channels. A value between 5-10 seconds is recommended.
DELAY_AFTER_FULL_CYCLE = 7

# Ignore signals that are older than this many seconds when the bot first sees them.
SIGNAL_MAX_AGE_SECONDS = 60


# ==============================================================================
# SECTION 2: CHANNEL & STRATEGY PROFILES
# ==============================================================================
# This is the core configuration for the bot. Each dictionary in this list
# represents a "profile" for a specific Discord channel, allowing you to
# assign a unique strategy to each signal provider.

CHANNEL_PROFILES = [
    {
        "channel_id": 1392531225348014180,  # Replace with the actual Channel ID
        "channel_name": "TEST CHANNEL",
        "enabled": True,  # Set to False to temporarily disable this profile

        # If a signal has no BUY/SELL keyword, assume it's a BUY.
        "assume_buy_on_ambiguous": False,

        # A list of words that, if found in a message, will cause the bot to ignore it.
        "reject_if_contains": ["lotto", "risky", "yolo", "swing"],

        # --- NEW: Entry Order Configuration ---
        "entry_order_type": "MKT",  # Options: "MKT", "PEG_MID", "ADAPTIVE_URGENT"
        "fill_timeout_seconds": 30,  # Timeout for non-market orders

        # The exit strategy to use for trades from this channel.
        # The 'type' field determines which settings are used.
        "exit_strategy": {
            "type": "dynamic_trail",  # The primary strategy is our internal logic.

            # --- Settings for the dynamic_trail ---
            "breakeven_trigger_percent": 20,
            "pullback_stop_percent": 15,
            "hard_stop_loss_percent": 35,
            "timeout_exit_minutes": 60,

            # --- NEW: Optional Safety Net Settings ---
            "safety_net": {
                "enabled": False,  # Set to True to attach a wide native trail on entry
                "native_trail_percent": 65  # The wide percentage for the safety net
            }
        }
    },
    # PROFILE 2
    {
        "channel_id": 1404464845419446322,  # Replace with the actual Channel ID
        "channel_name": "PROFILE_2",
        "enabled": True,  # Set to False to temporarily disable this profile

        # If a signal has no BUY/SELL keyword, assume it's a BUY.
        "assume_buy_on_ambiguous": False,

        # A list of words that, if found in a message, will cause the bot to ignore it.
        "reject_if_contains": ["lotto", "risky", "yolo", "swing"],

        # --- NEW: Entry Order Configuration ---
        "entry_order_type": "MKT",  # Options: "MKT", "PEG_MID", "ADAPTIVE_URGENT"
        "fill_timeout_seconds": 20,  # Timeout for non-market orders

        # The exit strategy to use for trades from this channel.
        # The 'type' field determines which settings are used.
        "exit_strategy": {
            "type": "dynamic_trail",  # The primary strategy is our internal logic.

            # --- Settings for the dynamic_trail ---
            "breakeven_trigger_percent": 20,
            "pullback_stop_percent": 15,
            "hard_stop_loss_percent": 35,
            "timeout_exit_minutes": 60,

            # --- NEW: Optional Safety Net Settings ---
            "safety_net": {
                "enabled": False,  # Set to True to attach a wide native trail on entry
                "native_trail_percent": 40  # The wide percentage for the safety net
            }
        }
    },
]


# ==============================================================================
# SECTION 3: TRADE SIZING & FILTERS
# ==============================================================================

PER_SIGNAL_FUNDS_ALLOCATION = 1000
MIN_PRICE = 0.20
MAX_PRICE = 10.0
RESTRICTED_SYMBOLS = []


# ==============================================================================
# SECTION 4: INTERACTIVE BROKERS (IBKR) CONNECTION
# ==============================================================================

ENABLE_PAPER_TRADING = True
USE_TWS = True
USE_GATEWAY = not USE_TWS
TWS_SETTINGS = {'IP': '127.0.0.1', 'PORT': 7497, 'CLIENT_ID': 53}
GATEWAY_SETTINGS = {'IP': '127.0.0.1', 'PORT': 4002, 'CLIENT_ID': 50}
ACCOUNT_NUMBER = ""


# ==============================================================================
# SECTION 5: KEYWORD CONFIGURATION
# ==============================================================================

BUY_KEYWORDS = ["BTO", "BUY", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE", "OPENING"]
SELL_KEYWORDS = ["STC", "SELL", "SOLD", "OUT", "EXIT", "CLOSE", "CUT", "STOPPED", "LOSS", "PROFITS"]
TRIM_KEYWORDS = ["TRIM", "SCALE", "LFG", "HOLDING", "TAKE", "UPDATE", "GAINS", "NOW", "REDUCE", "SECURE", "SAFETY"]
DAILY_EXPIRY_TICKERS = ["SPX", "SPY", "QQQ", "SPXW"]


# ==============================================================================
# SECTION 6: END-OF-DAY (EOD) AUTO-CLOSE
# ==============================================================================

EOD_CLOSE_ENABLED = True
EOD_CLOSE_HOUR = 15
EOD_CLOSE_MINUTE = 50

# In config.py, for example at the end of the file

# ==============================================================================
# SECTION 7: TELEGRAM NOTIFICATIONS
# ==============================================================================

TELEGRAM_SETTINGS = {
    "enabled": True,
    "bot_token_name": "TELEGRAM_BOT_TOKEN", # Name of the variable in your .env file
    "chat_id_name": "TELEGRAM_CHAT_ID"    # Name of the variable in your .env file
}
