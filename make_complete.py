#!/usr/bin/env python3
import os
import json
import pandas as pd
import nasdaqdatalink
from datetime import datetime

# 1) Setup
nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_API_KEY"]
MASTER   = "sep_dataset/SHARADAR_SEP.parquet"
SRC_TABLE = "SHARADAR/SEP"

# 2) Load missing map
with open("missing_pairs_common.json") as fp:
    missing_map = json.load(fp)

# 3) Load master once
df_master = pd.read_parquet(MASTER)
df_master["date"] = pd.to_datetime(df_master["date"]).dt.date

# 4) Loop tickers & dates
for ticker, dates in missing_map.items():
    for dt_str in dates:
        print(f"Fetching {ticker} @ {dt_str} ...", end=" ")
        df_day = nasdaqdatalink.get_table(SRC_TABLE,
                                          ticker=ticker,
                                          date=dt_str,
                                          paginate=True)
        if df_day.empty:
            print("no data")
            continue

        # normalize date
        df_day["date"] = pd.to_datetime(df_day["date"]).dt.date

        # drop any existing stale rows for this ticker/date
        mask = ~((df_master["ticker"]==ticker) & (df_master["date"]==df_day["date"].iloc[0]))
        df_master = df_master[mask]

        # append
        df_master = pd.concat([df_master, df_day], ignore_index=True)
        print(f"appended {len(df_day)} rows")

# 5) Write back full master
df_master.to_parquet(MASTER, index=False)
print("✅ All missing rows fetched & master SEP updated.")
