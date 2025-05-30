#!/usr/bin/env python3
import pandas as pd
import logging
from datetime import datetime

# ———————————————————
# CONFIGURE LOGGING
# ———————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    # ————————————————
    # 1) LOAD TICKER METADATA
    # ————————————————
    meta = pd.read_parquet("sep_dataset/SHARADAR_TICKERS_2.parquet")
    logging.info(f"Loaded metadata for {len(meta):,} tickers")

    # parse the listing / delisting dates
    for col in ("firstpricedate", "lastpricedate"):
        meta[col] = pd.to_datetime(meta[col], errors="coerce").dt.date

    # ————————————————
    # 2) LOAD SEP PRICE DATA
    # ————————————————
    sep = pd.read_parquet("sep_dataset/SHARADAR_SEP_2.parquet")
    logging.info(f"Loaded SEP data with {len(sep):,} rows across {sep['ticker'].nunique():,} tickers")

    # ensure your SEP date column is datetime.date
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # ————————————————
    # 3) COMPUTE SEP RANGES PER TICKER
    # ————————————————
    sep_ranges = (
        sep
        .groupby("ticker")["date"]
        .agg(sep_min="min", sep_max="max")
        .reset_index()
    )
    logging.info(f"Computed SEP date ranges for {len(sep_ranges):,} tickers")

    # ————————————————
    # 4) MERGE & COMPARE
    # ————————————————
    df = (
        meta[["ticker","firstpricedate","lastpricedate"]]
        .merge(sep_ranges, on="ticker", how="inner")
    )
    logging.info(f"Merged metadata with SEP ranges: {len(df):,} tickers to check")

    # flags for missing coverage
    df["missing_early"] = df["sep_min"] > df["firstpricedate"]
    # only check late if ticker has been delisted
    df["missing_late"] = df["lastpricedate"].notna() & (df["sep_max"] < df["lastpricedate"])

    # ————————————————
    # 5) SUMMARIES
    # ————————————————
    total = len(df)
    early = df["missing_early"].sum()
    late  = df["missing_late"].sum()
    logging.info(f"{early:,}/{total:,} tickers start AFTER their listing date")
    logging.info(f"{late:,}/{total:,} tickers end BEFORE their delisting date")

    # list a few examples of each
    if early:
        logging.info("Examples of missing_early:")
        print(df.loc[df["missing_early"], ["ticker","firstpricedate","sep_min"]].head(5).to_string(index=False))
    if late:
        logging.info("Examples of missing_late:")
        print(df.loc[df["missing_late"], ["ticker","sep_max","lastpricedate"]].head(5).to_string(index=False))

    # ————————————————
    # 6) OUTPUT FULL REPORT
    # ————————————————
    out_name = f"ticker_range_check_{datetime.today().date().isoformat()}.csv"
    df.to_csv(out_name, index=False)
    logging.info(f"Wrote full range report to '{out_name}'")

if __name__ == "__main__":
    main()
