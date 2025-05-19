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
        format="%(asctime)s %(levelname)s %(message)s"
    )

def get_last_trading_day(ref_date: datetime) -> str:
    nyse = mcal.get_calendar("NYSE")
    start = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = ref_date.strftime("%Y-%m-%d")
    sched = nyse.schedule(start_date=start, end_date=end)
    return sched.index[-1].strftime("%Y-%m-%d")

def run_step(cmd, name):
    logging.info(f"▶️  {name}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅  {name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌  {name} failed (exit {e.returncode})")
        sys.exit(e.returncode)

def main():
    setup_logging()
    p = argparse.ArgumentParser(
        description="Automated SHARADAR daily pipeline"
    )
    p.add_argument(
        "--date",
        help="YYYY-MM-DD to process; defaults to last NYSE trading day",
        default=None
    )
    args = p.parse_args()

    # 1) Determine proc_date
    if args.date:
        proc_date = args.date
    else:
        proc_date = get_last_trading_day(datetime.now())
    logging.info(f"Processing date: {proc_date}")

    # Only auto-skip when no explicit --date (i.e. in scheduled runs)
    if args.date is None:
        master_gold = "sep_dataset/SHARADAR_SEP_2.parquet"
        try:
            df_gold = pd.read_parquet(master_gold, columns=["date"])
            last_date = df_gold["date"].max().strftime("%Y-%m-%d")
            if last_date >= proc_date:
                logging.info(f"No new data: master gold already up to {last_date}. Exiting.")
                sys.exit(0)
        except Exception as e:
            logging.warning(f"Could not read {master_gold} (will proceed): {e}")

    # 3) Run steps
    run_step(
        ["python3", "daily_download.py", proc_date, "--merge"],
        "download & snapshot-merge"
    )
    run_step(
        ["python3", "merge_daily_download.py", proc_date, "--update-gold"],
        "promote to gold masters"
    )
    run_step(
        [
            "python3", "build_daily_common.py", proc_date,
            "--sep-source", f"sep_dataset/SHARADAR_SEP_{proc_date}.parquet",
            "--universe", "ticker_universe_clean.csv",
            "--out-dir", "sep_dataset"
        ],
        "build common-stock dataset"
    )

    logging.info(f"🎉 Pipeline complete for {proc_date}")

if __name__ == "__main__":
    main()
