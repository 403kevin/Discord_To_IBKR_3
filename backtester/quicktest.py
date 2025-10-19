import os
import glob

# List all CSV files in historical_data
files = glob.glob('backtester/historical_data/*.csv')
print(f"Found {len(files)} CSV files:\n")
for f in files:
    print(os.path.basename(f))

# If you want to see SPX files specifically:
spx_files = [f for f in files if 'SPX' in f]
print(f"\nSPX files ({len(spx_files)}):")
for f in spx_files:
    print(os.path.basename(f))
