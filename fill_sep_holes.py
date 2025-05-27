#!/usr/bin/env python3
"""
fill_sep_holes.py

Identify missing (ticker, date) rows in your SEP master so you can download them.
"""
import argparse
import pandas as pd
import pandas_market_calendars as mcal

def main():
    p = argparse.ArgumentParser(
        description="Find missing (ticker, date) rows in your SEP master"
    )
    p.add_argument("--universe",   required=True,
                   help="CSV of clean ticker universe (ticker column)")
    p.add_argument("--meta",       required=True,
                   help="SHARADAR_TICKERS_2.csv (with firstpricedate, lastpricedate)")
    p.add_argument("--sep-master", required=True,
                   help="Golden SEP master Parquet (all tickers, all dates)")
    p.add_argument("--out",        default="missing_dates.csv",
                   help="Output CSV of ticker,date pairs to fetch")
    args = p.parse_args()

    # 1) Load clean universe
    uni = pd.read_csv(args.universe, usecols=["ticker"])
    tickers = uni["ticker"].unique().tolist()
    print(f"[INFO] {len(tickers)} tickers to inspect")

    # 2) Load & collapse metadata to one row per ticker
    meta = pd.read_csv(
        args.meta,
        usecols=["ticker","firstpricedate","lastpricedate"],
        parse_dates=["firstpricedate","lastpricedate"]
    )
    meta_unique = (
        meta
        .groupby("ticker", as_index=True)
        .agg({
            "firstpricedate": "min",
            "lastpricedate":  "max"
        })
    )
    # convert to plain date for easy comparison
    meta_unique["firstpricedate"] = meta_unique["firstpricedate"].dt.date
    meta_unique["lastpricedate"]  = meta_unique["lastpricedate"] .dt.date

    # restrict metadata to tickers in universe
    meta_unique = meta_unique.loc[meta_unique.index.intersection(tickers)]
    print(f"[INFO] Metadata contains {len(meta_unique)} universe tickers")

    # 3) Load SEP master and extract dates
    sep = pd.read_parquet(args.sep_master, columns=["ticker","date"])
    sep["date"] = pd.to_datetime(sep["date"]).dt.date

    # 4) Build full NYSE calendar over SEP range
    start, end = sep["date"].min(), sep["date"].max()
    nyse  = mcal.get_calendar("NYSE")
    sched = nyse.schedule(
        start_date=start.isoformat(),
        end_date=end.isoformat()
    )
    calendar = sorted(set(sched.index.date))

    # 5) Map existing dates per ticker
    existing = sep.groupby("ticker")["date"].apply(set).to_dict()

    # 6) Find missing dates
    rows = []
    for t in tickers:
        if t not in meta_unique.index:
            print(f"[WARN] No metadata for {t}, skipping")
            continue
        fpd = meta_unique.at[t, "firstpricedate"]
        lpd = meta_unique.at[t, "lastpricedate"]
        # clamp window to SEP range
        win_start = max(fpd, start)
        win_end   = min(lpd, end)
        # expected trading days
        exp_days = [d for d in calendar if win_start <= d <= win_end]
        have     = existing.get(t, set())
        # missing days
        miss = sorted(d for d in exp_days if d not in have)
        for d in miss:
            rows.append({"ticker": t, "date": d})

    missing = pd.DataFrame(rows)
    print(f"[INFO] Found {len(missing)} missing rows")

    # 7) Write out
    missing.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    print(f"[INFO] Written missing dates to {args.out}")

if __name__ == "__main__":
    main()
