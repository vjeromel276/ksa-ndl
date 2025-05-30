#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from datetime import date

# ———————————————
# CONFIG
# ———————————————
DATA_START = date(1997, 12, 31)

# 1) LOAD THE PARQUET
df = pd.read_parquet("sep_dataset/SHARADAR_TICKERS_2.parquet")

# 2) PICK THE DATE COLUMN
date_cols = [c for c in df.columns if "date" in c.lower()]
if not date_cols:
    raise RuntimeError("No date-like column found; please adjust.")
date_col = date_cols[0]

# 3) PICK A TICKER TO TEST
T = "AAPL"

# 4) EXTRACT & NORMALIZE ALL ITS DATES
all_dates = (
    pd.to_datetime(df.loc[df["ticker"] == T, date_col], errors="coerce")
      .dropna()
      .dt.date
      .sort_values()
      .unique()
)

# 5) TRIM TO YOUR DATA RANGE
in_range = all_dates[all_dates >= DATA_START]
if len(in_range) == 0:
    print(f"No `{T}` dates on/after {DATA_START}; nothing to check.")
    sys.exit(0)

# 6) SPOT GAPS IN THAT FILTERED SERIES
# — Fixed comparison against 1 day
diffs = np.diff(in_range.astype("datetime64[D]"))
gap_idxs = np.where(diffs > np.timedelta64(1, "D"))[0]

if gap_idxs.size:
    print(f"⚠️  Found {len(gap_idxs)} gap(s) for {T}:")
    for idx in gap_idxs[:5]:
        print(f"  • gap from {in_range[idx]} → {in_range[idx+1]} ({diffs[idx]})")
else:
    print(f"✅  No gaps in {T}’s data (post-{DATA_START}).")

# 7) OPTIONAL: compare to business calendar
master = pd.bdate_range(start=in_range[0], end=in_range[-1]).date
missing = sorted(set(master) - set(in_range))
print(f"\nCompared to bdate_range: {len(missing)} missing days.")
if missing:
    print(" e.g.", missing[:3], "...", missing[-3:])
