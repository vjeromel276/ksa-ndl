#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import nasdaqdatalink

"""
make_complete.py

Backfill missing (ticker, date) rows into your SEP master Parquet
by fetching from the Nasdaq Data Link API.

Usage:
  python make_complete.py \
    --master SEP_MASTER.parquet \
    --missing missing_pairs.json
"""

def main(args=None):
    p = argparse.ArgumentParser(description="Backfill missing SEP rows")
    p.add_argument(
        "--master",
        help="Path to master SEP Parquet (overrides MASTER_PATH env var)",
        default=None
    )
    p.add_argument(
        "--missing",
        help="Path to missing_pairs.json (overrides MISSING_JSON env var)",
        default=None
    )
    opts = p.parse_args(args)

    master_path = opts.master or os.environ.get("MASTER_PATH", "sep_dataset/SHARADAR_SEP.parquet")
    missing_json = opts.missing or os.environ.get("MISSING_JSON", "missing_pairs.json")

    # Load missing map
    with open(missing_json, "r") as fp:
        missing_map = json.load(fp)

    # Load or init master
    df_master = pd.read_parquet(master_path)
    df_master["date"] = pd.to_datetime(df_master["date"]).dt.date

    # For each ticker/date, fetch and append if exists
    appended = 0
    for ticker, dates in missing_map.items():
        for date_str in dates:
            date_obj = pd.to_datetime(date_str).date()
            print(f"Fetching {ticker} @ {date_str} ...", end=" ")
            try:
                df_new = nasdaqdatalink.get_table(
                    "SHARADAR/SEP",
                    ticker=ticker,
                    date=date_str,
                    paginate=True
                )
            except Exception:
                df_new = pd.DataFrame()

            if df_new.empty:
                print("no data")
                continue

            # normalize date column
            df_new["date"] = pd.to_datetime(df_new["date"]).dt.date
            # drop any existing row for this ticker/date
            df_master = df_master[~((df_master["ticker"] == ticker) & (df_master["date"] == date_obj))]
            # append
            df_master = pd.concat([df_master, df_new], ignore_index=True)
            appended += len(df_new)
            print(f"appended {len(df_new)} rows")

    # Write back
    df_master.to_parquet(master_path, index=False)
    print(f"✅ All missing rows fetched & master SEP updated ({appended} rows appended).")

if __name__ == "__main__":
    main()
