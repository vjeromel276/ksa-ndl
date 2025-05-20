#!/usr/bin/env python3
import argparse
import subprocess
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal

# ----------------------------------------
# daily_pipeline.py
# ----------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )


def get_last_trading_day(ref_date: datetime) -> str:
    nyse = mcal.get_calendar("NYSE")
    start = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end = ref_date.strftime("%Y-%m-%d")
    sched = nyse.schedule(start_date=start, end_date=end)
    return sched.index[-1].strftime("%Y-%m-%d")


def run_step(cmd: list[str], name: str):
    logging.info(f"▶️  {name}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅  {name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌  {name} failed (exit {e.returncode})")
        sys.exit(e.returncode)


def filter_to_common_stock(raw_sep_path: str, meta_path: str, common_sep_path: str):
    """
    Filters a SEP parquet to only common stock tickers based on Sharadar metadata.
    Uses the 'category' field in SHARADAR_TICKERS_2.parquet to identify Common Stock.
    """
    # Load SEP snapshot
    sep = pd.read_parquet(raw_sep_path)
    # Load metadata with ticker and category
    meta = pd.read_parquet(meta_path, columns=["ticker", "category"])
    # Select only common stock tickers
    common_tickers = meta.loc[
        meta["category"].str.contains("Common Stock", case=False, na=False),
        "ticker"
    ]
    sep_common = sep[sep["ticker"].isin(common_tickers)]
    # Write filtered SEP
    sep_common.to_parquet(common_sep_path)
    logging.info(f"Filtered SEP to common stock: {len(sep_common)} rows")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated SHARADAR Daily Pipeline"
    )
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD to process; defaults to last NYSE trading day",
        default=None
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Determine processing date
    if args.date:
        proc_date = args.date
    else:
        proc_date = get_last_trading_day(datetime.now())
    logging.info(f"Processing date: {proc_date}")

    # Skip if already up to date (scheduled runs)
    if not args.date:
        gold_path = "sep_dataset/SHARADAR_SEP_2.parquet"
        try:
            df_gold = pd.read_parquet(gold_path, columns=["date"])
            last_date = df_gold["date"].max().strftime("%Y-%m-%d")
            if last_date >= proc_date:
                logging.info(f"No new data: master gold already up to {last_date}. Exiting.")
                sys.exit(0)
        except Exception as e:
            logging.warning(f"Could not read {gold_path}: {e}")

    # Step 1: Download and merge snapshot
    run_step(["python3", "daily_download.py", proc_date, "--merge"], "Download & Merge")
    run_step(["python3", "merge_daily_download.py", proc_date, "--update-gold"], "Promote to Gold Master")

    # Step 2: Filter SEP to common stock using metadata
    raw_sep = f"sep_dataset/SHARADAR_SEP_{proc_date}.parquet"
    common_sep = f"sep_dataset/SHARADAR_SEP_common_{proc_date}.parquet"
    filter_to_common_stock(raw_sep, "sep_dataset/SHARADAR_TICKERS_2.parquet", common_sep)

    # Step 3: Compute coverage & volume, produce clean universe CSV
    universe_csv = f"sep_dataset/ticker_universe_clean_{proc_date}.csv"
    run_step([
        "python3", "compute_per_ticker.py",
        "--common-sep", common_sep,
        "--meta-table", "sep_dataset/SHARADAR_TICKERS_2.parquet",
        "--cov-thresh", "0.99",
        "--vol-thresh", "100000",
        "--out-coverage", f"sep_dataset/ticker_coverage_{proc_date}.csv",
        "--out-vol", f"sep_dataset/ticker_coverage_vol_{proc_date}.csv",
        "--out-universe", universe_csv
    ], "Compute Coverage & Volume → Clean Universe")

    logging.info(f"🎉 Pipeline complete for {proc_date}")


if __name__ == "__main__":
    main()