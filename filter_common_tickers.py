#!/usr/bin/env python3
# ========================
# filter_common_tickers.py
# ========================
import argparse
import pandas as pd
from datetime import datetime
import logging
import os

TICKER_META_PATH = "sep_dataset/SHARADAR_TICKERS_2.parquet"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

def get_valid_common_tickers(meta_df: pd.DataFrame) -> set[str]:
    meta_df = meta_df.drop_duplicates(subset="ticker").copy()
    meta_df["ticker"] = meta_df["ticker"].astype(str).str.strip()
    meta_df["category"] = meta_df["category"].astype(str)

    contains_common = meta_df["category"].str.contains("Common Stock", case=False, na=False)
    excludes = ~meta_df["category"].str.contains("ADR|Warrant|Primary|Secondary|ETF|REIT", case=False, na=False)
    is_not_bad_suffix = ~meta_df["ticker"].str.endswith(tuple("UWRQ"))
    is_sane_format = meta_df["ticker"].str.match(r"^[A-Z]{1,5}(\.[A-Z])?$")

    final_mask = contains_common & excludes & is_not_bad_suffix & is_sane_format
    filtered = meta_df.loc[final_mask, "ticker"]

    logging.info(f"✅ Ticker filtering: {len(filtered):,} valid tickers selected "
                 f"(from {len(meta_df):,} total)")
    return set(filtered)

def filter_sep_by_tickers(sep_df: pd.DataFrame, ticker_set: set[str]) -> pd.DataFrame:
    before = len(sep_df)
    unique_before = sep_df["ticker"].nunique()

    filtered = sep_df[sep_df["ticker"].isin(ticker_set)].copy()

    after = len(filtered)
    unique_after = filtered["ticker"].nunique()

    logging.info(f"✅ SEP filtered: {before:,} → {after:,} rows "
                 f"({unique_before:,} → {unique_after:,} tickers)")
    return filtered


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Filter raw SEP to valid common stock tickers")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    date = args.date
    raw_sep = f"sep_dataset/SHARADAR_SEP_{date}.parquet"
    common_sep = f"sep_dataset/SHARADAR_SEP_common_{date}.parquet"

    if not os.path.exists(raw_sep):
        raise FileNotFoundError(f"Missing input: {raw_sep}")
    if not os.path.exists(TICKER_META_PATH):
        raise FileNotFoundError(f"Missing metadata: {TICKER_META_PATH}")

    logging.info(f"🚀 Running common stock filter for {date}")
    sep_df = pd.read_parquet(raw_sep)
    meta_df = pd.read_parquet(TICKER_META_PATH)

    valid_tickers = get_valid_common_tickers(meta_df)
    filtered_df = filter_sep_by_tickers(sep_df, valid_tickers)

    filtered_df.to_parquet(common_sep)
    logging.info(f"🎯 Saved cherry-filtered SEP: {common_sep} ({len(filtered_df):,} rows)")

if __name__ == "__main__":
    main()
