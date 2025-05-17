#!/usr/bin/env python3

import requests
import argparse

def download_file(url: str, dest_path: str):
    """
    Downloads a file in streaming mode and writes it to dest_path.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise on bad HTTP status
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f'Download complete: {dest_path}')

def main():
    parser = argparse.ArgumentParser(
        description="Download SHARADAR tables for a given date"
    )
    parser.add_argument(
        "date",
        help="Date to fetch (YYYY-MM-DD), e.g. 2025-05-12"
    )
    args = parser.parse_args()
    date = args.date

    TABLES = ["SEP", "ACTIONS", "METRICS"]
    API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
    API_KEY = "sMukN5Vun_5JyM7HzHr6"

    for table in TABLES:
        url = f"{API_BASE}/{table}.csv?date={date}&api_key={API_KEY}"
        output_file = f"data/sharadar_daily/SHARADAR_{table}_{date}.csv"
        download_file(url, output_file)

if __name__ == "__main__":
    main()
