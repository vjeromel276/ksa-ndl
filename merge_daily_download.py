#!/usr/bin/env python3
# ===============================
# merge_daily_download.py
# ===============================
import argparse
import logging
import os
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s"
    )

def merge_table(table: str, master_dir: str, data_dir: str,
                output_dir: str, date: str, update_gold: bool = False):
    """
    Merge a daily SEP CSV into a date-stamped Parquet snapshot.
    Optionally overwrite the gold master if update_gold is True.
    """
    master_path = os.path.join(master_dir, f"SHARADAR_{table}_2.parquet")
    daily_csv   = os.path.join(data_dir,   f"SHARADAR_{table}_{date}.csv")
    snapshot    = os.path.join(output_dir, f"SHARADAR_{table}_{date}.parquet")

    logging.debug(f"Loading master {table} from {master_path}")
    master_df = pd.read_parquet(master_path)
    logging.debug(f"Master {table} rows: {len(master_df)}")

    logging.debug(f"Reading daily {table} CSV from {daily_csv}")
    daily_df = pd.read_csv(daily_csv, parse_dates=['date'])
    logging.debug(f"Daily {table} rows: {len(daily_df)}")

    combined = pd.concat([master_df, daily_df], ignore_index=True)
    logging.debug(f"After concat: {len(combined)} rows")

    before = len(combined)
    # For SEP: dedupe on ticker+date
    combined = combined.drop_duplicates(subset=['ticker','date'])
    logging.debug(f"Dropped {before - len(combined)} duplicates")

    # Ensure datetime dtype & sort
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.sort_values(['ticker','date'])

    # ——— drop unused Sharadar columns early —————————————
    unused = ["closeunadj", "lastupdated"]
    combined.drop(columns=unused, errors="ignore", inplace=True)
    logging.info("Dropped unused columns during merge: %s", unused)

    logging.debug(f"Writing snapshot to {snapshot}")
    combined.to_parquet(snapshot, index=False)
    logging.info(f"Wrote snapshot: {snapshot}")

    if update_gold:
        logging.info(f"Overwriting gold master at {master_path}")
        combined.to_parquet(master_path, index=False)
        logging.info(f"Gold master updated: {master_path}")

if __name__ == "__main__":
    setup_logging()
    p = argparse.ArgumentParser(
        description="Merge daily SHARADAR SEP into date-stamped Parquet snapshot"
    )
    p.add_argument("date", help="YYYY-MM-DD for daily data")
    p.add_argument(
        "--master-dir",
        default="sep_dataset",
        help="Directory of master '_2.parquet'"
    )
    p.add_argument(
        "--data-dir",
        default="data/sharadar_daily",
        help="Directory of daily CSVs"
    )
    p.add_argument(
        "--output-dir",
        default="sep_dataset",
        help="Where to write the dated Parquet"
    )
    p.add_argument(
        "--update-gold",
        action="store_true",
        help="Also overwrite the gold master after merging"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    merge_table(
        table="SEP",
        master_dir=args.master_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        date=args.date,
        update_gold=args.update_gold
    )
