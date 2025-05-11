import pandas as pd

# 1) Load master SEP & tickers metadata
df = pd.read_parquet("sep_dataset/SHARADAR_SEP.parquet", columns=["ticker","date"])
df["date"] = pd.to_datetime(df["date"]).dt.date

meta = pd.read_parquet("sep_dataset/SHARADAR_TICKERS_2.parquet", 
                       columns=["ticker","exchange","security_type",
                                "firstpricedate","lastpricedate"])
meta["firstpricedate"] = pd.to_datetime(meta["firstpricedate"], errors="coerce").dt.date
meta["lastpricedate"]  = pd.to_datetime(meta["lastpricedate"],  errors="coerce").dt.date

# 2) Filter for common-stock on major exchanges
clean_meta = meta[
    (meta["security_type"] == "Common Stock") &
    (meta["exchange"].isin(["XNYS","XNAS"]))
]

# 3) Optional: enforce a minimum life span
#    e.g. at least 60 trading days on calendar
from pandas_market_calendars import get_calendar
nyse = get_calendar("NYSE")
all_days = nyse.schedule(start_date=df["date"].min().isoformat(),
                         end_date=df["date"].max().isoformat()).index.date

def lifespan_days(row):
    start, end = row["firstpricedate"], row["lastpricedate"] or all_days[-1]
    return sum(1 for d in all_days if start <= d <= end)

clean_meta["lifespan_days"] = clean_meta.apply(lifespan_days, axis=1)
clean_meta = clean_meta[clean_meta["lifespan_days"] >= 60]

# 4) Restrict your SEP to that universe
tickers_final = set(clean_meta["ticker"])
df_clean = df[df["ticker"].isin(tickers_final)]

# 5) Recompute shape
n_rows, n_cols = df_clean.shape
n_tickers      = df_clean["ticker"].nunique()
n_dates        = df_clean["date"].nunique()
expected      = n_tickers * n_dates

print("Clean universe shape:")
print(" rows × cols:", n_rows, "×", n_cols)
print(" tickers × dates:", n_tickers, "×", n_dates, f"= {expected}")
print(" coverage %:", f"{n_rows/expected:.2%}")
