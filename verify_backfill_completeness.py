#!/usr/bin/env python3
import pandas as pd
import pandas_market_calendars as mcal
import logging
from datetime import datetime, date

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_START     = date(1997, 12, 31)
SEP_PATH       = "sep_dataset/SHARADAR_SEP_2.parquet"
BACKFILL_A     = "backfilled_gaps_2025-05-29_B.parquet"
OUTPUT_CSV     = "backfill_B_completeness.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_missing_intervals(sorted_dates):
    """
    Given a sorted list of missing date objects, return
    a list of contiguous (start, end) tuples.
    """
    if not sorted_dates:
        return []
    intervals = []
    start = prev = sorted_dates[0]
    for curr in sorted_dates[1:]:
        if (curr - prev).days > 1:
            intervals.append((start, prev))
            start = curr
        prev = curr
    intervals.append((start, prev))
    return intervals

def main():
    # 1) Load the list of tickers we just backfilled (A‐letter)
    logging.info(f"Loading backfill tickers from {BACKFILL_A}")
    bf = pd.read_parquet(BACKFILL_A)
    backfill_tickers = sorted(bf["ticker"].unique())
    logging.info(f"→ {len(backfill_tickers)} tickers to verify")

    # 2) Load the updated SEP dataset
    logging.info(f"Loading merged SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 3) For each backfill ticker, find its effective first & last date in SEP
    #    (Note: if a ticker had data before 1997-12-31, we only expect data from 1997-12-31 onward.)
    #    We’ll gather “first_seen” and “last_seen” from SEP itself for each ticker.
    first_last = {}
    for tk in backfill_tickers:
        dates = sep.loc[sep["ticker"] == tk, "date"]
        if dates.empty:
            # shouldn't happen since we already verified presence
            logging.warning(f"{tk} has no rows in SEP even after merge!")
            continue
        first_seen = min(dates)
        last_seen  = max(dates)
        # Enforce DATA_START
        effective_start = first_seen if first_seen >= DATA_START else DATA_START
        effective_end   = last_seen
        first_last[tk] = (effective_start, effective_end)

    # 4) Build NYSE calendar from DATA_START to today
    overall_start = DATA_START
    overall_end   = datetime.today().date()
    logging.info(f"Building NYSE calendar from {overall_start} to {overall_end}")
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(
        start_date=overall_start.isoformat(),
        end_date=overall_end.isoformat()
    )
    trading_days_all = set(schedule.index.date)

    # 5) For each backfill ticker, compute missing intervals
    records = []
    for tk in backfill_tickers:
        start, end = first_last.get(tk, (None, None))
        if start is None:
            # no data in SEP
            records.append({
                "ticker":          tk,
                "complete?":       False,
                "missing_periods":"NO SEP ROWS",
                "total_expected":  0,
                "total_actual":    0
            })
            continue

        # Only consider expected days between start and end
        expected = {d for d in trading_days_all if start <= d <= end}

        # Actual days in SEP between start-end
        actual_dates = set(sep.loc[(sep["ticker"] == tk) & 
                                   (sep["date"] >= start) & 
                                   (sep["date"] <= end),
                                   "date"])

        missing = sorted(expected - actual_dates)
        if missing:
            intervals = group_missing_intervals(missing)
            missing_str = "; ".join(
                f"{s.isoformat()}→{e.isoformat()}" for s, e in intervals
            )
            total_expected = len(expected)
            total_actual   = len(actual_dates)
            complete = False
        else:
            missing_str = ""
            total_expected = len(expected)
            total_actual   = len(actual_dates)
            complete = True

        records.append({
            "ticker":          tk,
            "effective_start": start,
            "effective_end":   end,
            "total_expected":  total_expected,
            "total_actual":    total_actual,
            "complete?":       complete,
            "missing_periods": missing_str
        })

    # 6) Build DataFrame & write out
    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values(
        ["complete?", "ticker"],
        ascending=[False, True]
    ).reset_index(drop=True)

    df_out.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Wrote backfill‐A completeness report to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
