#!/usr/bin/env python3
# ingest_weekly.py
# Script to clean and merge weekly SHARADAR CSVs into master Parquet datasets.

import argparse
import os
import glob
import pandas as pd
from models.data import _coerce_sep_dtypes
from core.schema import validate_full_sep


def merge_csvs_to_parquet(csv_pattern: str, parquet_path: str, coerce_fn=None, validate_fn=None, index_cols=None):
    """
    Read all CSVs matching csv_pattern, concat, clean, and merge into the given Parquet file.
    - Skips empty files
    - Removes duplicates based on index_cols
    - Optionally applies coerce_fn and validate_fn
    """
    # Find files
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        print(f"[INFO] No files found for pattern {csv_pattern}")
        return

    dfs = []
    for f in csv_files:
        size = os.path.getsize(f)
        if size == 0:
            print(f"[WARNING] Skipping empty file: {f}")
            continue
        try:
            df_tmp = pd.read_csv(f, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            print(f"[WARNING] No data in file, skipping: {f}")
            continue
        dfs.append(df_tmp)

    if not dfs:
        print(f"[INFO] No valid CSVs to merge for pattern {csv_pattern}")
        return

    new_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Read {len(new_df)} rows from {len(dfs)} files matching {csv_pattern}")

    # Coerce and validate
    if coerce_fn:
        new_df = coerce_fn(new_df)
    if validate_fn:
        validate_fn(new_df)

    # Load or init master
    if os.path.exists(parquet_path):
        master = pd.read_parquet(parquet_path)
        print(f"[INFO] Loaded master '{parquet_path}' with {len(master)} rows")
    else:
        master = pd.DataFrame(columns=new_df.columns)
        print(f"[INFO] Initialized new master DataFrame for '{parquet_path}'")

    # Set index and drop duplicates
    if index_cols:
        new_df = new_df.set_index(index_cols)
        master = master.set_index(index_cols)

    combined = pd.concat([master, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]

    # Reset index if needed
    if index_cols:
        combined = combined.reset_index()

    # Save back to Parquet
    combined.to_parquet(parquet_path, index=False)
    print(f"[INFO] Master updated ({parquet_path}) to {len(combined)} rows")


def main():
    parser = argparse.ArgumentParser(description="Ingest weekly SHARADAR CSVs into master Parquet")
    parser.add_argument(
        "--weekly-dir", required=True,
        help="Directory where weekly CSVs are stored"
    )
    parser.add_argument(
        "--sep-parquet", default="sep_dataset/SHARADAR_SEP.parquet",
        help="Path to master SEP Parquet"
    )
    parser.add_argument(
        "--calendar-parquet", default="data/calendar.parquet",
        help="Path to master calendar Parquet"
    )
    args = parser.parse_args()

    # 1. Ingest SEP
    merge_csvs_to_parquet(
        csv_pattern=os.path.join(args.weekly_dir, "SEP_*.csv"),
        parquet_path=args.sep_parquet,
        coerce_fn=_coerce_sep_dtypes,
        validate_fn=validate_full_sep,
        index_cols=['ticker','date']
    )

    # 2. Ingest Calendar
    merge_csvs_to_parquet(
        csv_pattern=os.path.join(args.weekly_dir, "CALENDAR_*.csv"),
        parquet_path=args.calendar_parquet,
        coerce_fn=None,
        validate_fn=None,
        index_cols=['date']
    )

    # Others (optional): ACTIONS, INDICATORS, METRICS, TICKERS
    # For each, add a merge_csvs_to_parquet call with appropriate coerce/validate and index_cols

if __name__ == '__main__':
    main()
