import csv
import os
from datetime import datetime

LOG_FILE = 'trade_log.csv'

# Write headers if file doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'qty', 'price', 'action', 'reason'])

def log_trade(symbol, qty, price, action, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, symbol, qty, price, action, reason])
