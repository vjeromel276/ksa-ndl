#!/usr/bin/env python3
# ====================================================
# filter_common_with_history.py
# ====================================================
import argparse
import logging
import os
from datetime import datetime

import pandas as pd

# Paths / defaults
TICKER_META_PATH = "sep_dataset/SHARADAR_TICKERS_2.parquet"
DEFAULT_PRICE_THRESHOLD = 5         # minimum close price
DEFAULT_DOLLAR_VOL = 1_000_000      # minimum avg daily dollar‐volume
DEFAULT_WINDOW_DAYS = 252           # minimum trading‐day history

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

def get_valid_common_tickers(meta_df: pd.DataFrame) -> pd.Index:
    """Whitelist genuine common-stock tickers and exclude problematic symbols."""
    df = meta_df.drop_duplicates("ticker").copy()
    df["category"] = df["category"].astype(str)
    df["ticker"] = df["ticker"].fillna("").astype(str)
    mask = (
        df["category"].str.contains("Common Stock", case=False, na=False)
        & ~df["category"].str.contains("ADR|Warrant|ETF|REIT|Primary|Secondary",
                                       case=False, na=False)
        & ~df["ticker"].str.match(r".*[Q\d\.]$")
    )
    valid = df.loc[mask, "ticker"]
    logging.info(f"Ticker whitelist: {len(valid):,} genuine common stocks")
    return valid

def filter_price(sep: pd.DataFrame, price_thr: float) -> pd.DataFrame:
    before = sep.shape[0]
    sep = sep[sep["close"] >= price_thr]
    logging.info(f"Price ≥ ${price_thr}: {before:,} → {len(sep):,} rows")
    return sep

def filter_dollar_volume(sep: pd.DataFrame, dollar_thr: float) -> pd.DataFrame:
    sep = sep.copy()
    sep["dollar_vol"] = sep["close"] * sep["volume"]
    avg_dv = sep.groupby("ticker")["dollar_vol"].mean()
    keep = avg_dv[avg_dv >= dollar_thr].index
    before = sep.shape[0]
    sep = sep[sep["ticker"].isin(keep)]
    logging.info(f"Avg $-vol ≥ ${dollar_thr:,.0f}: kept {len(keep):,} tickers; "
                 f"{before:,} → {len(sep):,} rows")
    return sep.drop(columns="dollar_vol")

def filter_history_window(sep: pd.DataFrame, window: int) -> pd.DataFrame:
    # count trading days per ticker up to as_of_date
    counts = sep.groupby("ticker")["date"].nunique()
    keep = counts[counts >= window].index
    logging.info(f"History ≥ {window} days: {len(keep):,} tickers")
    return sep[sep["ticker"].isin(keep)]

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter SEP to common stocks with price, liquidity, and history requirements"
    )
    p.add_argument("--date", required=True,
                   help="As-of date (YYYY-MM-DD) whose SEP snapshot to filter")
    p.add_argument("--price-threshold", type=float, default=DEFAULT_PRICE_THRESHOLD,
                   help="Minimum close price")
    p.add_argument("--dollar-vol-threshold", type=float, default=DEFAULT_DOLLAR_VOL,
                   help="Minimum average daily dollar-volume")
    p.add_argument("--history-window", type=int, default=DEFAULT_WINDOW_DAYS,
                   help="Minimum trading days of history per ticker")
    return p.parse_args()

def main():
    setup_logging()
    args = parse_args()

    # validate date
    try:
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"Invalid --date {args.date}, must be YYYY-MM-DD")

    raw_path = f"sep_dataset/SHARADAR_SEP_{args.date}.parquet"
    out_path = f"sep_dataset/SHARADAR_SEP_filtered_{args.date}.parquet"

    for path in (raw_path, TICKER_META_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"Missing input file: {path}")

    logging.info(f"🚀 Filtering SEP snapshot for {args.date}")
    sep = pd.read_parquet(raw_path)
    meta = pd.read_parquet(TICKER_META_PATH)

    # 1) restrict to common-stock tickers
    whitelist = get_valid_common_tickers(meta)
    before = sep.shape[0]
    sep = sep[sep["ticker"].isin(whitelist)]
    logging.info(f"Ticker filter: {before:,} → {len(sep):,} rows")

    # 2) price floor
    sep = filter_price(sep, args.price_threshold)

    # 3) liquidity floor
    sep = filter_dollar_volume(sep, args.dollar_vol_threshold)

    # 4) history window
    sep = filter_history_window(sep, args.history_window)

    # 5) save
    sep.to_parquet(out_path)
    logging.info(f"🎯 Saved filtered SEP: {out_path} "
                 f"({len(sep):,} rows; {sep['ticker'].nunique():,} tickers)")

if __name__ == "__main__":
    main()
