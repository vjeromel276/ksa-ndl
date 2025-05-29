#!/usr/bin/env python3
"""
daily_download_with_fallback.py

Download Sharadar’s daily SEP CSV for a given date, detect any missing tickers
(for your target universe), and backfill those holes from yfinance in bulk.

Usage:
    python daily_download_with_fallback.py \
      --date YYYY-MM-DD \
      --data-dir data/sharadar_daily \
      --universe ticker_universe_clean.csv \
      [--out-csv path/to/filled.csv]
"""
import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def load_sharadar_csv(date: str, data_dir: str) -> pd.DataFrame:
    """
    Load the raw Sharadar CSV for the given date.
    """
    path = os.path.join(data_dir, f"SHARADAR_SEP_{date}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sharadar CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def backfill_missing_bulk(
    df: pd.DataFrame,
    universe: list[str],
    date: str
) -> pd.DataFrame:
    """
    Detect tickers missing from df for that date, fetch all of them in one
    bulk yfinance call, and append to df.
    """
    have = set(df["ticker"])
    missing = [t for t in universe if t not in have]
    if not missing:
        print(f"[INFO] No tickers missing on {date}")
        return df

    print(f"[INFO] {len(missing)} tickers missing on {date}; fetching via yfinance…")
    start = date
    end   = (pd.to_datetime(date) + timedelta(days=1)).strftime("%Y-%m-%d")

    # Bulk download for all missing tickers
    raw = yf.download(
        tickers=missing,
        start=start,
        end=end,
        progress=False,
        group_by='ticker',
        auto_adjust=False
    )

    rows = []
    for t in missing:
        # yfinance returns a MultiIndex DataFrame when multiple tickers are passed
        sub = raw[t] if t in raw else None
        if sub is None or sub.empty:
            print(f"[WARN] no yfinance data for {t} on {date}")
            continue
        bar = sub.iloc[0]
        rows.append({
            "ticker":      t,
            "date":        pd.to_datetime(date),
            "open":        bar["Open"],
            "high":        bar["High"],
            "low":         bar["Low"],
            "close":       bar["Close"],
            "volume":      bar["Volume"],
            "closeadj":    bar["Close"],
            "closeunadj":  bar["Close"],
            "lastupdated": datetime.utcnow()
        })

    if rows:
        df_fallback = pd.DataFrame(rows)
        df = pd.concat([df, df_fallback], ignore_index=True)
        print(f"[INFO] Appended {len(rows)} fallback rows")
    return df

def main():
    p = argparse.ArgumentParser(
        description="Download Sharadar SEP CSV and fill missing tickers via yfinance"
    )
    p.add_argument(
        "--date", required=True,
        help="Date to download in YYYY-MM-DD"
    )
    p.add_argument(
        "--data-dir", default="data/sharadar_daily",
        help="Directory containing Sharadar daily CSVs"
    )
    p.add_argument(
        "--universe", required=True,
        help="CSV file of tickers (column 'ticker') that should appear each day"
    )
    p.add_argument(
        "--out-csv", default=None,
        help="Path to write the filled CSV (defaults to overwriting original)"
    )
    args = p.parse_args()

    # 1) Load Sharadar CSV
    df = load_sharadar_csv(args.date, args.data_dir)

    # 2) Load your ticker universe
    uni = pd.read_csv(args.universe, usecols=["ticker"])
    universe = uni["ticker"].astype(str).tolist()

    # 3) Backfill missing tickers in bulk from yfinance
    df_filled = backfill_missing_bulk(df, universe, args.date)

    # 4) Write out the result
    out_path = args.out_csv or os.path.join(
        args.data_dir, f"SHARADAR_SEP_{args.date}_filled.csv"
    )
    df_filled.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"[INFO] Wrote filled CSV → {out_path}")

if __name__ == "__main__":
    main()
