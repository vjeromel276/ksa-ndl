#!/usr/bin/env python3
"""
validate_sep_master_completeness.py

Check that your SEP master Parquet has 100% of expected trading-day rows
for every ticker in your metadata, using fully vectorized datetime64 arrays.
"""
import argparse
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np

def main():
    p = argparse.ArgumentParser(
        description="Validate completeness of SEP master"
    )
    p.add_argument("--sep-master", required=True,
                   help="Path to SHARADAR_SEP_2.parquet")
    p.add_argument("--meta",       required=True,
                   help="Path to SHARADAR_TICKERS_2.csv")
    args = p.parse_args()

    # 1) Load SEP master
    sep = pd.read_parquet(args.sep_master, columns=["ticker","date"])
    sep["date"] = pd.to_datetime(sep["date"])
    tickers_in_sep = sep["ticker"].unique()
    print(f"[INFO] SEP master has {len(tickers_in_sep)} unique tickers")

    # 2) Load & collapse metadata to one row per ticker (keep datetime64)
    meta = pd.read_csv(
        args.meta,
        usecols=["ticker","firstpricedate","lastpricedate"],
        parse_dates=["firstpricedate","lastpricedate"]
    )
    meta_unique = (
        meta
        .groupby("ticker", as_index=True)
        .agg({
            "firstpricedate":"min",
            "lastpricedate": "max"
        })
    )

    # 3) Restrict to tickers actually in SEP
    meta_unique = meta_unique.loc[meta_unique.index.intersection(tickers_in_sep)]
    print(f"[INFO] {len(meta_unique)} metadata tickers appear in SEP master")

    # 4) Build NYSE calendar over SEP date range as numpy.datetime64 array
    start, end = sep["date"].min(), sep["date"].max()
    nyse  = mcal.get_calendar("NYSE")
    sched = nyse.schedule(
        start_date=start.isoformat(),
        end_date=end.isoformat()
    )
    calendar = sched.index.values  # np.ndarray[datetime64[ns]]

    # 5) Compute expected dates count per ticker via searchsorted
    fpd = meta_unique["firstpricedate"].values  # datetime64[ns] array
    lpd = meta_unique["lastpricedate"].values
    left  = np.searchsorted(calendar, fpd, side="left")
    right = np.searchsorted(calendar, lpd, side="right")
    meta_unique["possible_dates"] = (right - left).astype(int)

    # 6) Compute actual_dates from SEP
    actual = (
        sep.groupby("ticker")["date"]
           .nunique()
           .rename("actual_dates")
    )

    # 7) Join and compare
    report = meta_unique.join(actual, how="inner")
    report["aligned"] = report["possible_dates"] == report["actual_dates"]

    # 8) Summarize
    total    = len(report)
    aligned  = report["aligned"].sum()
    missing  = total - aligned
    pct_align = aligned/total*100

    print("\n=== SEP MASTER COMPLETENESS ===")
    print(f"Tickers total:        {total}")
    print(f"Fully aligned:        {aligned} ({pct_align:.1f}%)")
    print(f"Not aligned:          {missing} ({100-pct_align:.1f}%)")
    if missing:
        print("\nSample mis-aligned tickers:")
        print(report[~report.aligned]
              [["possible_dates","actual_dates"]]
              .head(10))

if __name__ == "__main__":
    main()
