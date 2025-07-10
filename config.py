# =============================
# DISCORD + SIGNAL PARSING LOGIC
# =============================

DISCORD_AUTH_TOKEN = '0000000000000'
CHANNEL_INFO = {
    'format_name': 'CommonParser',
    'channel_id': '1021379480503205898', #TEST SERVER
    'parser': 'CommonParser'
}

# Signal Keywords
BUY_SIGNALS = ["BTO", "BUY", "ADD", "ENTRY", "IN", "OPEN", "ENTER", "BOT", "ENTRIES", "HERE", "OPENING"]
SELL_SIGNALS = ["STC", "SELL", "SOLD", "OUT", "EXIT", "CLOSE", "CUT", "STOPPED", "LOSS", "PROFITS"]
TRIM_SIGNALS = ["TRIM", "SCALE", "LFG", "HOLDING", "TAKE", "UPDATE", "GAINS", "NOW", "REDUCE", "SECURE", "SAFETY"]
REJECT_SIGNALS = ["PLAY", "EARNING", "LOTTO", "LIGHT", "RISK", "RISKY", "STRADDLE", "WATCH", "COMMENT", "ON"]

DAILY_EXPIRY_SIGNALS = ["spx", "spy", "qqq", "spxw"]

# FORMAT_12_BUY = True will trigger a buy even if no explicit buy keywords exist
FORMAT_12_BUY = False


# =============================
# PRICE FILTERING + RUNTIME BEHAVIOR
# =============================

MIN_PRICE = 0.2
MAX_PRICE = 10.0

ALERT_EXPIRY_DURATION = 60  # Ignore signals older than this (seconds)
SLEEP_DELAY_BETWEEN_POLLS = 1  # Delay between Discord polls (seconds)
SIGNAL_MAX_AGE_SECONDS = 60  # or whatever value you prefer

TEST_MODE = False
ENABLE_PAPER_TRADING = True

# Restrict trades on these tickers
RESTRICTED_SYMBOLS = ['QQQ']

# Position sizing
PER_SIGNAL_FUNDS_ALLOCATION = 2000
PER_ADD_SIGNAL_FUNDS_ALLOCATION = 2000
PER_TRIM_AMOUNT_ALLOCATION = 1


# =============================
# TRAILING STOP SETTINGS
# =============================

# Master switch to enable/disable all trailing logic
TRAILING_STOP_ENABLED = True

# Use advanced adaptive trailing logic if True, else use IB native only
USE_ADVANCED_TRAILING = True

# Standard trailing percentage (used in basic IB trail or as fallback)
TRAILING_STOP_PERCENT = 2000  # per cent x 100 (1000 = 10%)

# Advanced adaptive trailing parameters
BREAKEVEN_TRIGGER_PERCENT = 5     # When price exceeds entry by this %, lock in MAX_LOSS_STOP_PERCENT
MAX_LOSS_STOP_PERCENT = 25        # Allowed pullback from highest price after breakeven
TIMEOUT_EXIT_MINUTES = 1          # Exit after this many minutes regardless of price

# Attach IB native trailing stop alongside adaptive logic
FALLBACK_IB_TRAIL_ENABLED = False




# =============================
# TRIM / SELL MANAGEMENT
# =============================

PERCENT_TO_TRIM = 100  # % of position to trim when TRIM command detected


# =============================
# BRACKET ORDER SETTINGS
# =============================

USE_BRAKET_ORDER = False
TAKE_PROFIT_PERCENTAGE = 15
STOP_LOSS_PERCENTAGE = 20


# =============================
# IB ORDER CONFIGS
# =============================

USE_OPTION_ADAPTIVE_ALGO = True
ADAPTIVE_PRIORITY_TYPE = 'Urgent'  # Options: Patient, Normal, Urgent

# TWS / Gateway
USE_TWS = True
USE_GATEWAY = not USE_TWS
PAPER_TRADING = True

# If using TWS and Paper
TWS = {'IP': '127.0.0.1', 'PORT': 7497, 'CLIENT_ID': 53}
GATEWAY = {'IP': '127.0.0.1', 'PORT': 4002, 'CLIENT_ID': 50}



# =============================
# MISC STRATEGY FLAGS
# =============================


ACCOUNT_NUMBER = ""  # Leave blank for default account

NEXT_FRIDAY_IS_A_HOLIDAY = False
ONE_CONTRACT_AT_A_TIME = False  # NOTE: Was not functional last checked

IGNORE_NEW_SIGNAL_IF_PL_GREATER_THAN = False
IGNORE_NEW_SIGNAL_IF_PL_GREATER_THAN_VALUE = 200

EXIT_HOUR = 23
EXIT_MINUTE = 0
