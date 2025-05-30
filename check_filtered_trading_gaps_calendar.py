#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import pandas_market_calendars as mcal
from datetime import date

# ———————————————
# CONFIG
# ———————————————
DATA_START = date(1997, 12, 31)
SEP_PATH   = "sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet"

# ———————————————
# LOGGING
# ———————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_intervals(dts):
    """
    Given a sorted list of date objects, returns list of (start,end) for contiguous runs.
    """
    runs = []
    start = prev = dts[0]
    for curr in dts[1:]:
        if (curr - prev).days > 1:
            runs.append((start, prev))
            start = curr
        prev = curr
    runs.append((start, prev))
    return runs

def main():
    # 1) load filtered SEP
    logging.info(f"Loading filtered SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 2) pre‐filter tickers exactly as you did before
    grp = sep.groupby("ticker").agg(
        trading_days=("date","nunique"),
        avg_price    =("close","mean"),
        # now compute dollar‐volume
        avg_dollar_vol=("close", lambda x: (x * sep.loc[x.index, "volume"]).mean())
    )
    sel = grp[
        (grp.trading_days   >= 252  ) &
        (grp.avg_price      >= 5    ) &
        (grp.avg_dollar_vol >= 1e6  )
    ]
    tickers = sel.index.to_list()
    logging.info(f"{len(tickers):,} tickers pass your filters")

    # 3) build the NYSE trading calendar once
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(
        start_date=DATA_START.isoformat(),
        end_date=date.today().isoformat()
    )
    trading_days_all = schedule.index.date

    # 4) for each ticker, compare actual dates to expected trading days
    gaps = []
    for tk in tickers:
        actual = sorted(d for d in sep.loc[sep["ticker"] == tk, "date"].unique() if d >= DATA_START)
        if len(actual) < 2:
            continue

        # expected days for this ticker
        expected = [d for d in trading_days_all if actual[0] <= d <= actual[-1]]

        # missing trading days
        missing = sorted(set(expected) - set(actual))
        if not missing:
            continue

        # group missing into contiguous intervals
        for start, end in group_intervals(missing):
            gap_len = (end - start).days + 1
            gaps.append({
                "ticker":     tk,
                "gap_start":  start,
                "gap_end":    end,
                "gap_length": gap_len
            })

    # 5) output
    gaps_df = pd.DataFrame(gaps)
    out_name = f"filtered_trading_gaps_calendar_{date.today().isoformat()}.csv"
    gaps_df.to_csv(out_name, index=False)
    logging.info(f"Wrote {len(gaps_df):,} gap records to {out_name}")

if __name__ == "__main__":
    main()
