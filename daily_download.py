#!/usr/bin/env python3
# =====================
# daily_download.py
# =====================
import requests
import argparse
import logging
import os
import sys

import pandas_market_calendars as mcal
import merge_daily_download  # assumes merge_daily_download.py is in the same dir

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
        description="Download SHARADAR SEP for a given date, with optional merge into Parquet"
    )
    parser.add_argument(
        "--date",
        required=True,
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
        help="Merge the downloaded SEP into a date-stamped Parquet snapshot"
    )
    parser.add_argument(
        "--master-dir",
        default="sep_dataset",
        help="Directory where master Parquet lives (for merge)"
    )
    parser.add_argument(
        "--output-dir",
        default="sep_dataset",
        help="Directory to write the merged Parquet"
    )
    args = parser.parse_args()

    date_str = args.date

    # 1) Check if it’s an NYSE trading day
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=date_str, end_date=date_str)
    if sched.empty:
        logging.info(f"{date_str} is not a trading day; skipping.")
        sys.exit(0)

    os.makedirs(args.data_dir, exist_ok=True)

    # 2) Download SEP CSV
    API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
    API_KEY  = "qux3577g2au_ELYoZfCy"
    url = f"{API_BASE}/SEP.csv?date={date_str}&api_key={API_KEY}"
    dest_csv = os.path.join(args.data_dir, f"SHARADAR_SEP_{date_str}.csv")
    download_file(url, dest_csv)

    # 3) Merge into Parquet snapshot if requested
    if args.merge:
        logging.info("Merging SEP into Parquet snapshot")
        os.makedirs(args.output_dir, exist_ok=True)
        merge_daily_download.merge_table(
            table="SEP",
            master_dir=args.master_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            date=date_str,
            update_gold=False
        )

if __name__ == "__main__":
    main()
