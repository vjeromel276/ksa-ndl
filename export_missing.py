#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import pandas_market_calendars as mcal
import json

"""
export_missing.py

Scan your SEP master for missing (ticker, date) pairs and dump them
into missing_pairs.json.

Usage:
    python export_missing.py \
      [--master path/to/SEP.parquet] \
      [--out path/to/missing_pairs.json]
"""

def main(args=None):
    p = argparse.ArgumentParser(description="Export missing ticker×date pairs")
    p.add_argument(
        "--master",
        help="Path to SEP master Parquet (overrides SEP_MASTER env var)",
        default=None
    )
    p.add_argument(
        "--out",
        help="Output JSON path",
        default="missing_pairs.json"
    )
    opts = p.parse_args(args or [])

    sep_path = opts.master or os.environ.get(
        "SEP_MASTER",
        "sep_dataset/SHARADAR_SEP.parquet"
    )
    out_path = opts.out

    # 1) Load SEP master
    df = pd.read_parquet(sep_path, columns=["ticker","date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 2) NYSE calendar
    nyse = mcal.get_calendar("NYSE")
    start, end = df["date"].min(), df["date"].max()
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    trading_days = sched.index.date

    # 3) Build full vs actual indices
    tickers = df["ticker"].unique()
    full_idx = pd.MultiIndex.from_product(
        [tickers, trading_days], names=["ticker","date"]
    )
    have_idx = pd.MultiIndex.from_frame(df[["ticker","date"]])
    missing_idx = full_idx.difference(have_idx)

    # 4) Report summary
    print(f"Total ticker×day pairs expected: {len(full_idx)}")
    print(f"Total present in data: {len(have_idx)}")
    print(f"Missing pairs: {len(missing_idx)}")

    if len(missing_idx) > 0:
        miss = pd.DataFrame(index=missing_idx).reset_index()
        print("\nFirst 10 missing (ticker, date):")
        print(miss.head(10).to_string(index=False))

        # 5) Group by ticker → list of ISO dates
        missing_map = (
            miss.groupby("ticker")["date"]
                .apply(lambda dates: [d.isoformat() for d in dates])
                .to_dict()
        )

        # 6) Write JSON
        with open(out_path, "w") as fp:
            json.dump(missing_map, fp, indent=2)

        print(f"Exported {len(missing_map)} tickers to {out_path}")

if __name__ == "__main__":
    main()
