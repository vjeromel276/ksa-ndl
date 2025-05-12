#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import pandas_market_calendars as mcal

"""
completeness_check.py

Check for missing (ticker, date) pairs in your SEP master file against the NYSE calendar.

Usage:
    python completeness_check.py --master path/to/SEP.parquet
"""

def main(args=None):
    p = argparse.ArgumentParser(description="Check SEP completeness vs NYSE calendar")
    p.add_argument(
        "--master",
        help="Path to the SEP Parquet file (overrides SEP_MASTER env var)",
        default=None
    )
    # opts = p.parse_args(args) # changed to opts = p.parse_args(args or [])
    opts = p.parse_args(args or [])

    sep_path = opts.master or os.environ.get(
        "SEP_MASTER",
        "sep_dataset/SHARADAR_SEP.parquet"
    )

    df = pd.read_parquet(sep_path, columns=["ticker","date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    nyse = mcal.get_calendar("NYSE")
    start, end = df["date"].min(), df["date"].max()
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    trading_days = sched.index.date

    tickers = df["ticker"].unique()
    full_idx = pd.MultiIndex.from_product([tickers, trading_days], names=["ticker","date"])
    have_idx = pd.MultiIndex.from_frame(df[["ticker","date"]])
    missing_idx = full_idx.difference(have_idx)

    print(f"Total ticker×day pairs expected: {len(full_idx)}")
    print(f"Total present in data: {len(have_idx)}")
    print(f"Missing pairs: {len(missing_idx)}")

    if len(missing_idx) > 0:
        miss = pd.DataFrame(index=missing_idx).reset_index()
        print("\nFirst 10 missing (ticker, date):")
        print(miss.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
