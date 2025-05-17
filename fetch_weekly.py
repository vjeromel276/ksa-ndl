#!/usr/bin/env python3
# fetch_weekly.py
# Script to download weekly slices of SHARADAR tables from Nasdaq Data Link,
# handling rate limits, date filtering, and saving incremental CSVs per week.

import argparse
import os
import time
import pandas as pd
import nasdaqdatalink
from nasdaqdatalink import ApiConfig

# Default SHARADAR dataset and tables
default_tables = ["SEP", "ACTIONS", "INDICATORS", "METRICS", "TICKERS", "CALENDAR"]
dataset = "SHARADAR"
# Rate-limit pause to avoid 429s
RATE_LIMIT_SLEEP = 2  # seconds


def fetch_table(dataset: str, table: str, start_date: str, end_date: str, out_dir: str):
    """
    Fetch rows for `table` in `dataset` between start_date and end_date (inclusive)
    using datatable filters: date.gte & date.lte. Saves CSV to out_dir/table_start_end.csv
    """
    print(f"[INFO] Fetching {dataset}/{table} from {start_date} to {end_date}")
    try:
        for table in default_tables:
            # Fetch data from Nasdaq Data Link
            df = nasdaqdatalink.get_table(
                f"{dataset}/{table}",
                date={'gte': start_date, 'lte': end_date},
                paginate=True
            )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {table}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    filename = f"{table}_{start_date}_{end_date}.csv"
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved {table} rows={len(df)} to {path}")

    # backoff to respect rate limits
    time.sleep(RATE_LIMIT_SLEEP)


def main():
    p = argparse.ArgumentParser("Fetch weekly SHARADAR datatables via Nasdaq Data Link")
    p.add_argument("--api-key", required=True, help="Your Nasdaq Data Link API key")
    p.add_argument("--out-dir", required=True, help="Directory to save weekly CSVs")
    p.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD) inclusive")
    p.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD) inclusive")
    p.add_argument(
        "--tables", nargs='+', default=default_tables,
        help="List of SHARADAR tables to fetch (default: all)"
    )
    args = p.parse_args()

    # configure client
    ApiConfig.api_key = args.api_key
    ApiConfig.use_retries = True
    ApiConfig.number_of_retries = 5
    ApiConfig.retry_backoff_factor = 0.5
    ApiConfig.max_wait_between_retries = 8
    ApiConfig.retry_status_codes = [429, 500, 502, 503, 504]

    # iterate tables
    for table in args.tables:
        try:
            fetch_table("SHARADAR", table, args.start_date, args.end_date, args.out_dir)
        except Exception as exc:
            print(f"[ERROR] {table}: {exc}\nRetrying after backoff...")
            time.sleep(RATE_LIMIT_SLEEP * 3)
            fetch_table("SHARADAR", table, args.start_date, args.end_date, args.out_dir)

if __name__ == '__main__':
    main()
