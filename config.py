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
        "channel_id": 1021379480503205898,  # Replace with the actual Channel ID
        "channel_name": "Test_Server_Trader",
        "enabled": True,  # Set to False to temporarily disable this profile

        # If a signal has no BUY/SELL keyword, assume it's a BUY.
        "assume_buy_on_ambiguous": False,

        # A list of words that, if found in a message, will cause the bot to ignore it.
        "reject_if_contains": ["lotto", "risky", "yolo", "swing"],

        # The exit strategy to use for trades from this channel.
        # The 'type' field determines which settings are used.
        "exit_strategy": {
            "type": "dynamic_trail",  # Options: "dynamic_trail", "bracket", "native_trail", "none"

            # --- Settings for "dynamic_trail" ---
            "breakeven_trigger_percent": 20,
            "pullback_stop_percent": 15,
            "hard_stop_loss_percent": 25,
            "timeout_exit_minutes": 30,

            # --- Settings for "bracket" ---
            "take_profit_percent": 25,
            "stop_loss_percent": 25,

            # --- Settings for "native_trail" ---
            "trailing_percent": 25
        }
    },
    # --- Add another profile for a different trader below ---
    # {
    #     "channel_id": 987654321098765432,
    #     "channel_name": "Trader_Jane_Smith",
    #     "enabled": True,
    #     "assume_buy_on_ambiguous": False,
    #     "reject_if_contains": ["earnings", "play"],
    #     "exit_strategy": {
    #         "type": "bracket",
    #         "take_profit_percent": 20,
    #         "stop_loss_percent": 15,
    #     }
    # }
]


# ==============================================================================
# SECTION 3: TRADE SIZING & FILTERS
# ==============================================================================

# The maximum amount of funds to allocate for a single opening trade.
PER_SIGNAL_FUNDS_ALLOCATION = 2000

# Price filters: ignore trades for options outside this price range.
MIN_PRICE = 0.20
MAX_PRICE = 10.0

# A list of tickers to ignore all signals for. Example: ['SPY', 'AMZN']
RESTRICTED_SYMBOLS = []


# ==============================================================================
# SECTION 4: INTERACTIVE BROKERS (IBKR) CONNECTION
# ==============================================================================

# Set to True for paper trading, False for live trading.
ENABLE_PAPER_TRADING = True

# TWS is for manual oversight; Gateway is better for automated bots.
# Set one to True and the other to False.
USE_TWS = True
USE_GATEWAY = not USE_TWS

TWS_SETTINGS = {'IP': '127.0.0.1', 'PORT': 7497, 'CLIENT_ID': 53}
GATEWAY_SETTINGS = {'IP': '127.0.0.1', 'PORT': 4002, 'CLIENT_ID': 50}

# Your IBKR account number. Leave as an empty string ("") to use the default.
ACCOUNT_NUMBER = ""


# ==============================================================================
# SECTION 5: KEYWORD CONFIGURATION
# ==============================================================================
# The parser uses these lists to understand the intent of a signal.

BUY_KEYWORDS = ["BTO", "BUY", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE", "OPENING"]
SELL_KEYWORDS = ["STC", "SELL", "SOLD", "OUT", "EXIT", "CLOSE", "CUT", "STOPPED", "LOSS", "PROFITS"]
TRIM_KEYWORDS = ["TRIM", "SCALE", "LFG", "HOLDING", "TAKE", "UPDATE", "GAINS", "NOW", "REDUCE", "SECURE", "SAFETY"]

# A list of tickers that typically have 0DTE or daily expiries.
DAILY_EXPIRY_TICKERS = ["SPX", "SPY", "QQQ", "SPXW"]

