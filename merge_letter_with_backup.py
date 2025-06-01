#!/usr/bin/env python3
import argparse
import os
import shutil
import pandas as pd
from datetime import date

def parse_args():
    p = argparse.ArgumentParser(
        description="Backup SHARADAR_SEP_2.parquet and merge a single‐letter backfill into it"
    )
    p.add_argument(
        "--letter", "-l",
        required=True,
        help="The single letter of the backfill file to merge (A, B, C, etc.)"
    )
    p.add_argument(
        "--date", "-d",
        required=True,
        help="Date suffix of your backfill file (YYYY-MM-DD), e.g. 2025-05-29"
    )
    return p.parse_args()

def main():
    args     = parse_args()
    LETTER   = args.letter.upper()
    SEP_DATE = args.date

    # Paths
    SEP_DIR         = "sep_dataset"
    GOLDEN_SEP      = os.path.join(SEP_DIR, "SHARADAR_SEP_2.parquet")
    BACKFILL_LETTER = f"backfilled_gaps_{SEP_DATE}_{LETTER}.parquet"
    # Backup name uses today's date
    today           = date.today().isoformat()
    BACKUP_SEP      = os.path.join(SEP_DIR, f"SHARADAR_SEP_2_backup_{today}.parquet")

    # 1) Confirm files exist
    if not os.path.isfile(GOLDEN_SEP):
        raise FileNotFoundError(f"Golden SEP not found at: {GOLDEN_SEP}")
    if not os.path.isfile(BACKFILL_LETTER):
        raise FileNotFoundError(f"Backfill file for letter '{LETTER}' not found: {BACKFILL_LETTER}")

    # 2) Back up the current golden SEP
    print(f"Backing up {GOLDEN_SEP} → {BACKUP_SEP} …")
    shutil.copyfile(GOLDEN_SEP, BACKUP_SEP)

    # 3) Load the backup (to perform merge on the backed‐up copy)
    sep = pd.read_parquet(BACKUP_SEP)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 4) Load the single‐letter backfill
    bf = pd.read_parquet(BACKFILL_LETTER)
    if "Date" not in bf.columns:
        raise ValueError(f"Expected a 'Date' column in {BACKFILL_LETTER}, but not found.")
    bf = bf.rename(columns={"Date": "date"})
    bf["date"] = pd.to_datetime(bf["date"], errors="coerce").dt.date

    # 5) Concatenate backup + backfill
    combined = pd.concat([sep, bf], ignore_index=True)

    # 6) Drop duplicates on (ticker, date), keeping the first (backup) row if it existed
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="first")

    # 7) Sort and overwrite the original golden SEP
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(GOLDEN_SEP, index=False)

    print(f"Merge complete. Overwrote {GOLDEN_SEP} with merged data ({len(combined):,} rows).")
    print(f"Backup retained at: {BACKUP_SEP}")

if __name__ == "__main__":
    main()
