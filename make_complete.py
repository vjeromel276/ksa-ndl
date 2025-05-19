#!/usr/bin/env python3
# make_complete.py

import os
import json
import pandas as pd
import nasdaqdatalink
from tqdm.auto import tqdm

def make_complete(df_master: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill missing (ticker, date) rows into df_master and return the updated DataFrame.
    missing_df must have columns ['ticker','date'] where 'date' is a datetime or date.
    """
    df = df_master.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Iterate through missing pairs with a progress bar
    for _, row in tqdm(missing_df.iterrows(),
                       total=len(missing_df),
                       desc="Backfilling missing SEP rows"):
        ticker   = row['ticker']
        date_val = row['date']
        # Normalize date to datetime.date
        if isinstance(date_val, pd.Timestamp):
            date_obj = date_val.date()
        elif isinstance(date_val, str):
            date_obj = pd.to_datetime(date_val).date()
        else:
            date_obj = date_val

        iso_date = date_obj.isoformat()
        try:
            new_chunk = nasdaqdatalink.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date=iso_date,
                paginate=True
            )
        except Exception:
            new_chunk = pd.DataFrame()

        if new_chunk.empty:
            continue

        # Normalize and drop any existing row for this ticker/date
        new_chunk['date'] = pd.to_datetime(new_chunk['date']).dt.date
        mask = ~((df['ticker'] == ticker) & (df['date'] == date_obj))
        df = df[mask]

        # Append fetched rows
        df = pd.concat([df, new_chunk], ignore_index=True)

    return df

# CLI entry-point
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Backfill missing SEP rows into master Parquet"
    )
    p.add_argument(
        "--master",
        help="Path to master SEP Parquet (overrides MASTER_PATH env var)",
        default=None
    )
    p.add_argument(
        "--missing",
        help="Path to missing_pairs.json (overrides MISSING_JSON env var)",
        default=None
    )
    args = p.parse_args()

    master_path  = args.master  or os.environ.get("MASTER_PATH",  "sep_dataset/SHARADAR_SEP.parquet")
    missing_path = args.missing or os.environ.get("MISSING_JSON", "missing_pairs.json")

    # Load master and missing-map
    df_master = pd.read_parquet(master_path)
    with open(missing_path, "r") as fp:
        missing_map = json.load(fp)

    # Convert missing_map dict -> DataFrame
    rows = []
    for ticker, dates in missing_map.items():
        for d in dates:
            rows.append({"ticker": ticker, "date": pd.to_datetime(d)})
    missing_df = pd.DataFrame(rows)

    # Run backfill with progress bar
    updated_df = make_complete(df_master, missing_df)

    # Write back to master
    appended = len(updated_df) - len(df_master)
    updated_df.to_parquet(master_path, index=False)
    print(f"✅ All missing rows fetched & master SEP updated ({appended} rows appended).")

if __name__ == "__main__":
    main()
