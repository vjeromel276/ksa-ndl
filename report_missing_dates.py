#!/usr/bin/env python3
import pandas as pd
import logging
from datetime import datetime

# ————————————————
# CONFIGURE LOGGING
# ————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    # ————————————————
    # 1) LOAD UNIVERSE
    # ————————————————
    logging.info("Loading ticker universe from 'ticker_universe_clean.csv'")
    tickers = pd.read_csv("ticker_universe_clean.csv")["ticker"].astype(str).tolist()
    logging.info(f"→ {len(tickers)} tickers loaded")

    # ————————————————
    # 2) LOAD SEP_2 DATA
    # ————————————————
    logging.info("Reading SEP_2.parquet")
    df = pd.read_parquet("sep_dataset/SHARADAR_SEP_2.parquet")
    logging.info(f"→ SEP_2 contains {len(df):,} rows across {df['ticker'].nunique():,} tickers")

    # ————————————————
    # 3) FILTER TO CLEAN UNIVERSE
    # ————————————————
    df = df[df["ticker"].isin(tickers)]
    logging.info(f"After filter: {len(df):,} rows across {df['ticker'].nunique():,} tickers")

    # ————————————————
    # 4) NORMALIZE DATE COLUMN
    # ————————————————
    logging.info("Converting `date` column to datetime")
    df["date"] = pd.to_datetime(df["date"])
    logging.info(f"→ Date range: {df['date'].min().date()} through {df['date'].max().date()}")

    # ————————————————
    # 5) BUILD MASTER CALENDAR
    # ————————————————
    start, end = df["date"].min(), df["date"].max()
    master_dates = pd.bdate_range(start=start, end=end)
    logging.info(f"Master business‐day calendar contains {len(master_dates):,} dates")

    # ————————————————
    # 6) COMPUTE MISSING DATES PER TICKER
    # ————————————————
    missing_rows = []
    for ticker, group in df.groupby("ticker"):
        available = pd.DatetimeIndex(group["date"].unique())
        missing = master_dates.difference(available)
        logging.info(
            f"{ticker:6s}: {len(available):5,} available dates → "
            f"{len(missing):5,} missing"
        )
        for dt in missing:
            missing_rows.append({
                "ticker":   ticker,
                "missing_date": dt.date().isoformat()
            })

    # ————————————————
    # 7) OUTPUT REPORT
    # ————————————————
    report_df = pd.DataFrame(missing_rows)
    output_name = f"missing_dates_report_{datetime.today().date().isoformat()}.csv"
    report_df.to_csv(output_name, index=False)
    logging.info(f"Written {len(report_df):,} rows to '{output_name}'")

if __name__ == "__main__":
    main()
