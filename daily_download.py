#!/usr/bin/env python3
# =====================
# daily_download.py
# =====================
import requests
import argparse
import logging
import os
import sys
import pandas as pd
import pandas_market_calendars as mcal
import merge_daily_download  # assumes merge_daily_download.py in the same directory

# Tables to download and merge
TABLES = ["SEP", "ACTIONS", "METRICS"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )


def download_file(url: str, dest_path: str):
    """
    Downloads a file in streaming mode and writes it to dest_path.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logging.info(f'Download complete: {dest_path}')


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Download SHARADAR tables for a given date, with optional merge into Parquet"
    )
    parser.add_argument(
        "date",
        help="Date to fetch (YYYY-MM-DD), e.g. 2025-05-12"
    )
    parser.add_argument(
        "--data-dir",
        default="data/sharadar_daily",
        help="Directory to store downloaded CSVs"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the downloaded tables into date-stamped Parquet snapshots"
    )
    parser.add_argument(
        "--master-dir",
        default="sep_dataset",
        help="Directory where master Parquets live (for merge)"
    )
    parser.add_argument(
        "--output-dir",
        default="sep_dataset",
        help="Directory to write the merged Parquets"
    )
    args = parser.parse_args()

    date_str = args.date
    # Check if trading day
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=date_str, end_date=date_str)
    if schedule.empty:
        logging.info(f"{date_str} is not a NYSE trading day; skipping download.")
        sys.exit(0)

    os.makedirs(args.data_dir, exist_ok=True)

    # Download CSVs
    API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
    API_KEY = "sMukN5Vun_5JyM7HzHr6"
    for table in TABLES:
        url = f"{API_BASE}/{table}.csv?date={date_str}&api_key={API_KEY}"
        output_file = os.path.join(args.data_dir, f"SHARADAR_{table}_{date_str}.csv")
        download_file(url, output_file)

    # Merge into Parquet snapshots if requested
    if args.merge:
        logging.info("Merging daily tables into Parquet snapshots")
        os.makedirs(args.output_dir, exist_ok=True)
        for table in TABLES:
            try:
                merge_daily_download.merge_table(
                    table=table,
                    master_dir=args.master_dir,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    date=date_str,
                    update_gold=False
                )
            except Exception as e:
                logging.error(f"Failed to merge {table} for date {date_str}: {e}")


if __name__ == "__main__":
    main()
