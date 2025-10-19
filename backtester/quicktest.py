import os
import pandas as pd

# Get the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Build path to historical_data relative to script location
data_dir = os.path.join(script_dir, 'historical_data')
print(f"Data directory: {data_dir}")

# List files
import glob
files = glob.glob(os.path.join(data_dir, '*.csv'))
print(f"\nFound {len(files)} CSV files:")
for f in files:
    print(f"  - {os.path.basename(f)}")

# Now let's check the SPX 6650P file
test_file = os.path.join(data_dir, 'SPX_20250929_6650P_databento.csv')
if os.path.exists(test_file):
    df = pd.read_csv(test_file)
    print(f"\nFirst 10 rows of SPX_20250929_6650P_databento.csv:")
    print(df.head(10))
else:
    # Try without the P
    test_file = os.path.join(data_dir, 'SPX_20250929_6650_P_databento.csv')
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        print(f"\nFirst 10 rows of SPX_20250929_6650_P_databento.csv:")
        print(df.head(10))
