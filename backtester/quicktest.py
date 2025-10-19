import pandas as pd
df = pd.read_csv('backtester/historical_data/SPX_20250929_6650_P_databento.csv')
print("First 20 rows:")
print(df.head(20))
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
