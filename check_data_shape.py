#!/usr/bin/env python3
import argparse
import pandas as pd
import pandas.api.types as ptypes

def main():
    p = argparse.ArgumentParser(
        description="Inspect rows, tickers & dates in a SEP‐common Parquet"
    )
    p.add_argument("parquet", help="Path to SHARADAR_SEP_common_<date>.parquet")
    args = p.parse_args()

    # 1) Load the data
    df = pd.read_parquet(args.parquet)

    # 2) Force the 'date' column into actual datetimes
    df['date'] = pd.to_datetime(df['date'], unit='ns', errors='ignore')
    # If that didn't work (e.g. object or string), try generic parse:
    if not ptypes.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='raise')

    # 3) Compute shape metrics
    n_rows    = len(df)
    n_tickers = df['ticker'].nunique()
    n_dates   = df['date'].dt.date.nunique()
    expected  = n_tickers * n_dates
    coverage  = n_rows / expected * 100

    # 4) Print summary
    print(f"Rows × columns: {df.shape}")
    print(f"Unique tickers:  {n_tickers}")
    print(f"Unique dates:    {n_dates}")
    print(f"Expected (t×d):  {n_tickers}×{n_dates} = {expected}")
    print(f"Coverage:        {coverage:.2f}%")

if __name__ == "__main__":
    main()
