#!/usr/bin/env python3
# ===============================
# merge_daily_download.py
# ===============================
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
    """
    master_path = os.path.join(master_dir, f"SHARADAR_{table}_2.parquet")
    daily_csv = os.path.join(data_dir, f"SHARADAR_{table}_{date}.csv")
    output_path = os.path.join(output_dir, f"SHARADAR_{table}_{date}.parquet")

    logging.debug(f"Loading master {table} from {master_path}")
    master_df = pd.read_parquet(master_path)
    logging.debug(f"Master {table} loaded: {len(master_df)} rows")

    logging.debug(f"Reading daily {table} CSV from {daily_csv}")
    cols = pd.read_csv(daily_csv, nrows=0).columns.tolist()
    parse_dates = ['date'] if 'date' in cols else None
    daily_df = pd.read_csv(daily_csv, parse_dates=parse_dates)
    # enforce datetime dtype if present
    if 'date' in daily_df.columns:
        daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    logging.debug(f"Daily {table} loaded: {len(daily_df)} rows, cols={daily_df.columns.tolist()}")

    logging.debug(f"Concatenating master and daily for {table}")
    combined = pd.concat([master_df, daily_df], ignore_index=True)
    logging.debug(f"After concat: {len(combined)} rows")

    # Deduplicate combined DataFrame
    if table == "SEP":
        before = len(combined)
        combined = combined.drop_duplicates(subset=['ticker', 'date'])
        logging.debug(f"Dropped {before - len(combined)} duplicates on ['ticker','date']")
    else:
        before = len(combined)
        combined = combined.drop_duplicates()
        logging.debug(f"Dropped {before - len(combined)} full-row duplicates")

    logging.debug(f"Sorting combined {table}")
    combined = combined.sort_values(combined.columns.tolist())

    logging.debug(f"Writing merged {table} to {output_path}")
    combined.to_parquet(output_path, index=False)
    logging.info(f"Wrote merged {table} file: {output_path}")

    # Overwrite gold master if requested
    if update_gold:
        logging.info(f"Overwriting gold master for {table} at {master_path}")
        combined.to_parquet(master_path, index=False)
        logging.info(f"Gold master updated: {master_path}")


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
        help="Directory where master Parquets live"
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
        help="Also overwrite the master `_2.parquet` files with the merged data"
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
