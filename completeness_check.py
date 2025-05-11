import pandas as pd
import pandas_market_calendars as mcal

# 1) Load your master SEP
df = pd.read_parquet("sep_dataset/SHARADAR_SEP.parquet", columns=["ticker","date"])
df["date"] = pd.to_datetime(df["date"]).dt.date

# 2) NYSE calendar between your min/max dates
nyse = mcal.get_calendar("NYSE")
start, end = df["date"].min(), df["date"].max()
sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
trading_days = sched.index.date

# 3) Unique tickers
tickers = df["ticker"].unique()

# 4) Full index vs. actual
full_idx = pd.MultiIndex.from_product([tickers, trading_days], names=["ticker","date"])
have_idx = pd.MultiIndex.from_frame(df[["ticker","date"]])
missing_idx = full_idx.difference(have_idx)

# 5) Report
print("Total ticker×day pairs expected:", len(full_idx))
print("Total present in data:       ", len(have_idx))
print("Missing pairs:               ", len(missing_idx))

if len(missing_idx) > 0:
    miss = pd.DataFrame(index=missing_idx).reset_index()
    print("\nFirst 10 missing (ticker, date):")
    print(miss.head(10).to_string(index=False))
