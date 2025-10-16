#!/usr/bin/env python3
"""Quick Databento test with real signal"""
import os
import sys
from datetime import datetime, timedelta

# Get API key from command line
if len(sys.argv) < 2:
    print("Usage: python test_databento_now.py YOUR_API_KEY")
    sys.exit(1)

api_key = sys.argv[1]

print("="*80)
print("DATABENTO QUICK TEST")
print("="*80)

# Test connection
try:
    import databento as db
    client = db.Historical(api_key)
    print("✅ Connected to Databento")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test signal: SPY 580C expiring 10/18/2024
ticker = "SPY"
strike = 580
right = "C"
expiry = "20241018"

print(f"\n📥 Testing: {ticker} {strike}{right} exp {expiry}")

# Build OCC symbol
strike_str = f"{int(strike * 1000):08d}"
occ_symbol = f"{ticker}{expiry}{right}{strike_str}"
print(f"   OCC Symbol: {occ_symbol}")

# Get 1 day of data
exp_date = datetime.strptime(expiry, '%Y%m%d')
start = exp_date - timedelta(days=1)
end = exp_date

print(f"   Requesting: {start.date()} to {end.date()}")

try:
    data = client.timeseries.get_range(
        dataset='OPRA.pillar',
        symbols=[occ_symbol],
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        schema='trades',
        limit=1000
    )
    
    df = data.to_df()
    
    if df.empty:
        print("   ⚠️  No data (symbol might not exist)")
    else:
        print(f"   ✅ Downloaded {len(df)} trades")
        print(f"\n   Sample data:")
        print(f"   First trade: {df.index[0]} @ ${df['price'].iloc[0]:.2f}")
        print(f"   Last trade:  {df.index[-1]} @ ${df['price'].iloc[-1]:.2f}")
        print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        # Save to CSV
        output_file = f"test_{ticker}_{expiry}_{strike}{right}.csv"
        df.to_csv(output_file)
        print(f"\n   💾 Saved to: {output_file}")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ TEST COMPLETE - Databento works!")
print("="*80)
