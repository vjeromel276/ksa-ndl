#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import date

# ———————————————
# CONFIG
# ———————————————
DATA_START = date(1997, 12, 31)
SEP_PATH   = "sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet"

# ———————————————
# SETUP LOGGING
# ———————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_intervals(dts):
    # returns list of (start, end) for contiguous runs in sorted list of dates
    runs = []
    start = prev = dts[0]
    for curr in dts[1:]:
        if (curr - prev).days > 1:
            runs.append((start, prev))
            start = curr
        prev = curr
    runs.append((start, prev))
    return runs

def main():
    # 1) load filtered SEP
    logging.info(f"Loading filtered SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 2) compute per‐ticker metrics
    grp = sep.groupby("ticker").agg(
        trading_days=("date","nunique"),
        avg_price    =("close","mean"),
        avg_vol      =("volume","mean")
    )
    # 3) apply your filters
    sel = grp[
        (grp.trading_days >= 252) &
        (grp.avg_price    > 5  ) 
    ]
    tickers = sel.index.to_list()
    logging.info(f"{len(tickers):,} tickers pass your filters")

    # 4) find gaps for each ticker
    gaps = []
    for tk in tickers:
        dates = sorted(sep.loc[sep["ticker"] == tk, "date"].unique())
        # ignore anything before DATA_START
        dates = [d for d in dates if d >= DATA_START]
        if len(dates) < 2:
            continue

        intervals = group_intervals(dates)
        for start, end in intervals:
            if (end - start).days == 0:
                continue
            # if the run itself has no holes, skip
            # gaps are between runs, so we need diffs between end of one run and start of next
            # better: build master and diff—but simplest: detect intervals where start!=prev+1
            # group_intervals gives contiguous runs, so compute gaps between runs:
        # gaps between runs:
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            gap_start = e1 + pd.Timedelta(days=1)
            gap_end   = s2 - pd.Timedelta(days=1)
            gap_len   = (gap_end - gap_start).days + 1
            gaps.append({
                "ticker":      tk,
                "gap_start":   gap_start,
                "gap_end":     gap_end,
                "gap_length":  gap_len
            })

    # 5) emit CSV
    gaps_df = pd.DataFrame(gaps)
    out_name = f"filtered_trading_gaps_{date.today().isoformat()}.csv"
    gaps_df.to_csv(out_name, index=False)
    logging.info(f"Wrote {len(gaps_df):,} gap records to {out_name}")

if __name__ == "__main__":
    main()
