#!/usr/bin/env python3
# ===============================
# merge_daily_download.py
# ===============================
# Example Usage:
#   # Create snapshots only:
#   python merge_daily_download.py 2025-05-12
#
#   # Create snapshots and overwrite the gold master:
#   python merge_daily_download.py 2025-05-12 --update-gold
#
# This script merges daily CSV downloads into date-stamped Parquet snapshots for SEP, ACTIONS, and METRICS.
# It reads from master files suffixed with '_2.parquet' and handles schema mismatches gracefully.

import argparse
import logging
import os
import pandas as pd

# Tables to process
TABLES = ["SEP", "ACTIONS", "METRICS"]


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s"
    )


def merge_table(table: str, master_dir: str, data_dir: str, output_dir: str, date: str, update_gold: bool):
    """
    Merge a daily CSV into a date-stamped Parquet snapshot for the given table.
    Optionally overwrite the master Parquet if update_gold is True.
    The master Parquet is expected as SHARADAR_<table>_2.parquet in master_dir.
    """
    master_path = os.path.join(master_dir, f"SHARADAR_{table}_2.parquet")
    daily_csv = os.path.join(data_dir, f"SHARADAR_{table}_{date}.csv")
    snapshot_path = os.path.join(output_dir, f"SHARADAR_{table}_{date}.parquet")

    # Load daily data
    logging.debug(f"Reading daily {table} CSV from {daily_csv}")
    cols = pd.read_csv(daily_csv, nrows=0).columns.tolist()
    parse_dates = ['date'] if 'date' in cols else None
    daily_df = pd.read_csv(daily_csv, parse_dates=parse_dates)
    daily_count = len(daily_df)
    logging.debug(f"Daily {table} loaded: {daily_count} rows, cols={daily_df.columns.tolist()}")

    # Load or init master
    if os.path.exists(master_path):
        logging.debug(f"Loading master {table} from {master_path}")
        master_df = pd.read_parquet(master_path)
        master_count = len(master_df)
        logging.debug(f"Master {table} loaded: {master_count} rows, cols={master_df.columns.tolist()}")
        # Align 'date' dtype
        if 'date' in master_df.columns and 'date' in daily_df.columns:
            master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
        # Convert categorical to str
        for col in master_df.select_dtypes(['category']).columns:
            master_df[col] = master_df[col].astype(str)
            if col in daily_df.columns:
                daily_df[col] = daily_df[col].astype(str)
    else:
        logging.warning(f"Master file for {table} not found; initializing empty master.")
        master_df = daily_df.iloc[0:0].copy()
        master_count = 0

    # Concatenate
    logging.debug(f"Concatenating master and daily for {table}")
    combined = pd.concat([master_df, daily_df], ignore_index=True)
    combined_count = len(combined)
    logging.debug(f"After concat: {combined_count} rows")

    # Deduplicate
    if table == "SEP":
        before = combined_count
        combined = combined.drop_duplicates(subset=['ticker', 'date'])
        deduped = before - len(combined)
        logging.debug(f"Dropped {deduped} duplicates on ['ticker','date']")
    else:
        before = combined_count
        combined = combined.drop_duplicates()
        deduped = before - len(combined)
        logging.debug(f"Dropped {deduped} full-row duplicates")

    final_count = len(combined)

    # Ensure 'date' dtype
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')

    # Sort
    logging.debug(f"Sorting combined {table}")
    combined = combined.sort_values(by=combined.columns.tolist())

    # Write snapshot
    logging.debug(f"Writing snapshot for {table} to {snapshot_path}")
    combined.to_parquet(snapshot_path, index=False)
    logging.info(f"Snapshot written: {snapshot_path}")

    # Update gold if requested
    if update_gold:
        logging.info(f"Overwriting gold master for {table} at {master_path}")
        combined.to_parquet(master_path, index=False)
        logging.info(f"Gold master updated: {master_path}")

    # Summary
    logging.info(f"[{table}] master={master_count}, daily={daily_count}, combined={combined_count}, "
                 f"deduped={deduped}, final={final_count}")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Merge daily SHARADAR tables into date-stamped Parquet snapshots"
    )
    parser.add_argument(
        "date",
        help="Date for daily data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--master-dir",
        default="sep_dataset",
        help="Directory where master Parquets live (with '_2.parquet' suffix)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/sharadar_daily",
        help="Directory where daily CSVs are stored"
    )
    parser.add_argument(
        "--output-dir",
        default="sep_dataset",
        help="Directory to write the date-stamped Parquets"
    )
    parser.add_argument(
        "--update-gold",
        action="store_true",
        help="Overwrite the gold master Parquet after merging"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for table in TABLES:
        try:
            merge_table(
                table=table,
                master_dir=args.master_dir,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                date=args.date,
                update_gold=args.update_gold
            )
        except Exception as e:
            logging.error(f"Failed to merge {table} for date {args.date}: {e}")
