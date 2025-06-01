#!/usr/bin/env python3
import pandas as pd
import pandas_market_calendars as mcal
import logging
from datetime import date

# ———————————————
# CONFIG
# ———————————————
SEP_PATH   = "sep_dataset/SHARADAR_SEP_2.parquet"
OUT_CSV    = f"missing_days_per_ticker_{date.today().isoformat()}.csv"

# ———————————————
# LOGGING
# ———————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    # 1) load filtered SEP
    logging.info(f"Loading filtered SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # determine overall date bounds for calendar
    overall_start = sep["date"].min()
    overall_end   = sep["date"].max()
    logging.info(f"Data covers {overall_start} → {overall_end}")

    # 2) build NYSE calendar between those dates
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(
        start_date=overall_start.isoformat(),
        end_date  =overall_end.isoformat()
    )
    trading_days_all = schedule.index.date

    # 3) compute missing‐day counts per ticker
    results = []
    tickers = sep["ticker"].unique()
    logging.info(f"Found {len(tickers):,} tickers to check")

    for tk in tickers:
        dates = sep.loc[sep["ticker"] == tk, "date"].unique()
        dates.sort()
        if len(dates) < 2:
            # trivial case—nothing to compare
            missing = len(dates)  # or 0?
        else:
            first, last = dates[0], dates[-1]
            # expected trading days for this ticker
            mask = (trading_days_all >= first) & (trading_days_all <= last)
            expected = set(trading_days_all[mask])
            actual   = set(dates)
            missing  = len(expected - actual)

        results.append({"ticker": tk, "missing_count": missing})

    # 4) assemble and save
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("missing_count", ascending=False)
    df_out.to_csv(OUT_CSV, index=False)
    logging.info(f"Wrote missing‐day counts to {OUT_CSV}")

if __name__ == "__main__":
    main()
