#!/usr/bin/env python3
import pandas as pd
import logging
from datetime import date

# CONFIG
TICKER_LIST    = "filtered_tickers_2025-05-29.csv"
FILTERED_SEP   = "sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet"
OUT_MISSING    = f"tickers_missing_all_{date.today().isoformat()}.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # 1) load expected tickers
    expected = pd.read_csv(TICKER_LIST, header=None, names=["ticker"])["ticker"].astype(str)
    expected_set = set(expected)
    logging.info(f"Loaded {len(expected_set):,} expected tickers")

    # 2) load filtered SEP and get actual tickers present
    sep = pd.read_parquet(FILTERED_SEP)
    actual_set = set(sep["ticker"].unique())
    logging.info(f"Found {len(actual_set):,} tickers in the Parquet")

    # 3) compute missing
    missing = sorted(expected_set - actual_set)
    logging.info(f"{len(missing):,} tickers have NO rows in the dataset")

    # 4) write them out
    if missing:
        pd.Series(missing, name="ticker").to_csv(OUT_MISSING, index=False)
        logging.info(f"Wrote list of missing tickers to {OUT_MISSING}")
    else:
        logging.info("✅ All expected tickers are present!")

if __name__ == "__main__":
    main()
