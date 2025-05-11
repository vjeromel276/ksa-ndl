import pandas as pd

# Load the master SEP
df = pd.read_parquet("sep_dataset/SHARADAR_SEP.parquet")

# Print overall shape
print("Total rows × columns:", df.shape)

# Print unique tickers × unique dates
n_tickers = df["ticker"].nunique()
n_dates   = pd.to_datetime(df["date"]).dt.date.nunique()
print(f"Unique tickers: {n_tickers}")
print(f"Unique dates:   {n_dates}")

# Expected rows = n_tickers × n_dates
print("Expected rows:", n_tickers * n_dates)

