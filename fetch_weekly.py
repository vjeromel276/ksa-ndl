#!/usr/bin/env python3
# fetch_weekly.py
# Script to download weekly slices of SHARADAR tables from Nasdaq Data Link in small batches,
# handling rate limits (429) and saving CSVs or Parquets per week.

import argparse
import os
import time
import nasdaqdatalink
import pandas as pd

# Default SHARADAR dataset and tables
DEFAULT_DATASET = "SHARADAR"
DEFAULT_TABLES = ["ACTIONS", "INDICATORS", "METRICS", "SEP", "TICKERS", "CALENDAR"]

# Sleep time between requests to avoid 429 errors
RATE_LIMIT_SLEEP = 2  # seconds


def fetch_table(dataset: str, table: str, start_date: str, end_date: str, api_key: str, out_dir: str):
    """
    Fetch a single SHARADAR table from Nasdaq Data Link for the given date range.
    Saves output to out_dir/table_start_end.csv
    """
    print(f"[INFO] Fetching {dataset}/{table} from {start_date} to {end_date}")
    df = nasdaqdatalink.get(
        f"{dataset}/{table}",
        start_date=start_date,
        end_date=end_date,
        paginate=True,
        api_key=api_key
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{table}_{start_date}_{end_date}.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {table} to {out_path}")
    time.sleep(RATE_LIMIT_SLEEP)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weekly SHARADAR tables via Nasdaq Data Link API"
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Nasdaq Data Link API key"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to save downloaded CSVs"
    )
    parser.add_argument(
        "--start-date", required=True,
        help="Start date (YYYY-MM-DD) for first week"
    )
    parser.add_argument(
        "--end-date", required=True,
        help="End date (YYYY-MM-DD) exclusive of last week"
    )
    parser.add_argument(
        "--tables", nargs='+', default=DEFAULT_TABLES,
        help="List of SHARADAR tables to fetch"
    )
    args = parser.parse_args()

    # Split into weekly intervals
    week_starts = pd.date_range(start=args.start_date, end=args.end_date, freq='7D').strftime('%Y-%m-%d')
    week_ends = week_starts[1:].tolist() + [args.end_date]

    for start, end in zip(week_starts, week_ends):
        for table in args.tables:
            try:
                fetch_table(DEFAULT_DATASET, table, start, end, args.api_key, args.out_dir)
            except nasdaqdatalink.NasdaqDataLinkError as e:
                print(f"[ERROR] {table} {start}->{end}: {e}")
                print("Retrying after backoff...")
                time.sleep(RATE_LIMIT_SLEEP * 3)
                fetch_table(DEFAULT_DATASET, table, start, end, args.api_key, args.out_dir)

if __name__ == '__main__':
    main()
