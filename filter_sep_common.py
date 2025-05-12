#!/usr/bin/env python3
import pandas as pd

# 1) Load full SEP master
df = pd.read_parquet("sep_dataset/SHARADAR_SEP.parquet")

# 2) Load your clean‐universe tickers
clean = pd.read_csv("ticker_universe_clean.csv")
valid = set(clean["ticker"])

# 3) Filter to common‐stock universe
df_common = df[df["ticker"].isin(valid)]

# 4) Write out
out_path = "sep_dataset/SHARADAR_SEP_common.parquet"
df_common.to_parquet(out_path, index=False)
print(f"Wrote {out_path} ({df_common.shape[0]} rows × {df_common.shape[1]} cols)")
