import csv
import os
from datetime import datetime

LOG_FILE = 'trade_log.csv'
HEADER = [
    'timestamp', 'symbol', 'qty', 'price', 'action', 'reason',
    'trader_name', 'strategy_details'
]

# Write headers if the log file doesn't exist or is empty
if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def log_trade(symbol: str, qty: int, price: float, action: str, reason: str, trader_name: str, strategy_details: str):
    """
    Logs a trade to the CSV file with the new, enhanced details.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, symbol, qty, price, action, reason,
            trader_name, strategy_details
        ])
