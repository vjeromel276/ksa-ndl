#!/usr/bin/env python3

import requests




def download_file(url: str, dest_path: str):
    """
    Downloads a file in streaming mode and writes it to dest_path.
    """
    # Stream=True to download in chunks (avoids loading whole file into memory) :contentReference[oaicite:0]{index=0}
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
    print(f'Download complete: {dest_path}')

if __name__ == '__main__':
    TABLES = ["SEP", "ACTIONS", "INDICATORS", "METRICS", "TICKERS", "CALENDAR"]
    API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
    DATE = "2025-05-12"
    API_KEY = "sMukN5Vun_5JyM7HzHr6"
    # dynamically generated URL 
    URL = f"{API_BASE}/{table}.csv?date={DATE}&api_key={API_KEY}"
    OUTPUT_FILE = f"SHARADAR_{table}_2025-05-12.csv"

    download_file(URL, OUTPUT_FILE)
            