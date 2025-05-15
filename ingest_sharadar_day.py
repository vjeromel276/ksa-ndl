#!/usr/bin/env python3
"""
ingest_sharadar_day.py

Ingest a single trading day's worth of Sharadar data (SEP & ACTIONS),
either from your local Parquets or, if missing, via the Nasdaq Data Link API,
then append into your master Parquet files with QC checks.
"""

import os
import sys
import argparse
import pandas as pd
import pandas_market_calendars as mcal
import nasdaqdatalink
from datetime import date as _date

from core.schema import validate_sep_df, REQUIRED_COLUMNS   # ← bring in the full‐schema set

# Hard-coded directory where all Sharadar Parquet files live
SEP_DIR = "sep_dataset"


def fetch_via_api(table_name: str, date_str: str) -> pd.DataFrame:
    """Fallback to pull one day's data from the Nasdaq Data Link API."""
    print(f"[INFO] Fetching {table_name} for {date_str} via API...")
    try:
        df = nasdaqdatalink.get_table(f"SHARADAR/{table_name}",
                                      date=date_str, paginate=True)
    except Exception as e:
        print(f"[ERROR] API fetch failed: {e}")
        return pd.DataFrame()
    if df.empty:
        print(f"[WARN] API returned no rows for {table_name} on {date_str}")
    return df


def ingest_table(tgt_date: _date,
                 table_name: str,
                 src_parquet: str,
                 master_parquet: str,
                 date_col: str,
                 key_cols: list):
    src_path    = os.path.join(SEP_DIR, src_parquet)
    master_path = os.path.join(SEP_DIR, master_parquet)

    # --- 1) Pull today's slice from your local source PARQUET (if present) ---
    df_day = pd.DataFrame()
    if os.path.exists(src_path):
        df_src = pd.read_parquet(src_path)
        df_src[date_col] = pd.to_datetime(df_src[date_col]).dt.date
        df_day = df_src[df_src[date_col] == tgt_date]
        if df_day.empty:
            print(f"[WARN] No rows for {src_parquet} on {tgt_date}")
    else:
        print(f"[WARN] Source file missing: {src_parquet}")

    # --- 2) Fallback to API if local returned nothing ---
    if df_day.empty:
        df_day = fetch_via_api(table_name, tgt_date.isoformat())
        if not df_day.empty:
            df_day[date_col] = pd.to_datetime(df_day[date_col]).dt.date

    # nothing to do if still empty
    if df_day.empty:
        return

    # --- 3) QC on today's slice ---
    if df_day.duplicated(subset=key_cols).any():
        dups = df_day[df_day.duplicated(subset=key_cols, keep=False)]
        raise ValueError(f"Duplicate rows in {table_name} for {tgt_date}:\n{dups}")

    # --- 4) Load (and conditionally validate) the existing master, drop today's old rows, then append ---
    if os.path.exists(master_path):
        df_master = pd.read_parquet(master_path)

        # only validate SEP when it already *has* the full schema
        if table_name.upper() == "SEP" and REQUIRED_COLUMNS.issubset(df_master.columns):
            validate_sep_df(df_master)

        df_master[date_col] = pd.to_datetime(df_master[date_col]).dt.date
        # remove any stale rows for tgt_date
        df_master = df_master[df_master[date_col] != tgt_date]
        df_out    = pd.concat([df_master, df_day], ignore_index=True)
    else:
        df_out = df_day

    # Normalize lastupdated if present
    if "lastupdated" in df_out.columns:
        df_out["lastupdated"] = pd.to_datetime(df_out["lastupdated"],
                                               errors="coerce")

    # --- 5) Write it back out ---
    df_out.to_parquet(master_path, index=False)
    print(f"[OK] Appended {len(df_day)} rows to {master_parquet}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Ingest one day's Sharadar SEP & ACTIONS.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = p.parse_args(argv)

    # API key
    api_key = os.environ.get("NASDAQ_API_KEY")
    if not api_key:
        sys.exit("[ERROR] NASDAQ_API_KEY not set in environment")
    nasdaqdatalink.ApiConfig.api_key = api_key

    # Target date
    try:
        tgt_date = pd.to_datetime(args.date).date()
    except Exception:
        sys.exit(f"[ERROR] Bad date: {args.date}")

    # Ensure it's a trading day
    nyse  = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="1998-01-01", end_date=args.date)
    if tgt_date not in sched.index.date:
        sys.exit(f"[ERROR] {tgt_date} is not a NYSE trading day")

    # Ingest SEP & ACTIONS
    ingest_table(
        tgt_date,
        table_name="SEP",
        src_parquet="SHARADAR_SEP_2.parquet",
        master_parquet="SHARADAR_SEP.parquet",
        date_col="date",
        key_cols=["ticker", "date"]
    )
    ingest_table(
        tgt_date,
        table_name="ACTIONS",
        src_parquet="SHARADAR_ACTIONS_2.parquet",
        master_parquet="SHARADAR_ACTIONS.parquet",
        date_col="date",
        key_cols=["ticker", "date", "action", "contraticker"]
    )

    print("[DONE] Ingestion pass complete.")


if __name__ == "__main__":
    main()
