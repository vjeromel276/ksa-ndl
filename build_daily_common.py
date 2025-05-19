# build_daily_common.py
#!/usr/bin/env python3
"""
build_daily_common.py

Driver to take a date‐stamped SEP snapshot, backfill missing common‐stock rows,
filter to your ticker universe, and write a date‐versioned common‐stock Parquet.

Usage:
  python build_daily_common.py 2025-05-12 \
    --sep-source sep_dataset/SHARADAR_SEP_2025-05-12.parquet \
    --universe ticker_universe_clean.csv \
    --out-dir sep_dataset
"""
import argparse
import logging
import os
import pandas as pd

from export_missing import export_missing_map
from filter_missing_common import filter_missing_pairs, missing_map_to_df
from make_complete import make_complete
from tqdm.auto import tqdm
from filter_sep_common import filter_sep_common

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

def build_common(date: str, sep_source: str, universe_csv: str, out_dir: str):
    logging.info(f"🔄 Building common‐stock for {date}")
    # 1) load the SEP snapshot
    df = pd.read_parquet(sep_source)
    logging.info(f"Loaded SEP snapshot: {len(df)} rows")

    # 2) compute missing for **this** date only
    missing_map = export_missing_map(df)
    common_map  = filter_missing_pairs(missing_map)
    # keep only the target date
    iso = date  # already "YYYY-MM-DD"
    common_map = {
        ticker: [d for d in dates if d == iso]
        for ticker, dates in common_map.items()
    }
    # drop tickers with no missing for this date
    common_map = {t: ds for t, ds in common_map.items() if ds}
    logging.info(f"Tickers missing on {iso}: {len(common_map)}")

    # expand into DataFrame (just one row per ticker)
    missing_df = pd.DataFrame([
        {"ticker": t, "date": pd.to_datetime(iso)}
        for t in common_map
    ])
    logging.info(f"Missing pairs DataFrame for {iso}: {len(missing_df)} rows")

    if not missing_df.empty:
        logging.info(f"Backfilling {len(missing_df)} missing rows")
        df = make_complete(df, missing_df)
    else:
        logging.info("No missing common‐stock rows")

    # 3) filter to approved common‐stock universe
    common_df = filter_sep_common(df, universe_csv)
    logging.info(f"Filtered to common‐stock universe: {len(common_df)} rows")

    # 4) write the date‐stamped common file
    out_path = os.path.join(out_dir, f"SHARADAR_SEP_common_{date}.parquet")
    common_df.to_parquet(out_path, index=False)
    logging.info(f"✨ Wrote common‐stock Parquet: {out_path}")

def main():
    setup_logging()
    p = argparse.ArgumentParser(
        description="Build a date‐stamped common‐stock SEP Parquet"
    )
    p.add_argument("date", help="YYYY-MM-DD")
    p.add_argument("--sep-source", required=True,
                   help="Path to SHARADAR_SEP_<date>.parquet")
    p.add_argument("--universe",   required=True,
                   help="CSV of approved common‐stock tickers")
    p.add_argument("--out-dir",    default="sep_dataset",
                   help="Directory to write the common‐stock Parquet")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_common(args.date, args.sep_source, args.universe, args.out_dir)

if __name__ == "__main__":
    main()
