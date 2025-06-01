#!/usr/bin/env python3
import pandas as pd
import pandas_market_calendars as mcal
import logging
from datetime import datetime, date

# -------------------------------
# CONFIGURATION
# -------------------------------

# The earliest date your SEP feed actually contains is 1997-12-31
DATA_START = date(1997, 12, 31)

TICKER_CSV    = "common_mid_large_primary_caps.csv"
GOLDEN_SEP    = "sep_dataset/SHARADAR_SEP_2.parquet"
OUTPUT_CSV    = "ticker_history_completeness.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_missing_intervals(sorted_missing_dates):
    """
    Given a sorted list of date objects representing missing days,
    return a list of (start, end) tuples for contiguous missing intervals.
    """
    if not sorted_missing_dates:
        return []
    intervals = []
    start = prev = sorted_missing_dates[0]
    for curr in sorted_missing_dates[1:]:
        if (curr - prev).days > 1:
            intervals.append((start, prev))
            start = curr
        prev = curr
    intervals.append((start, prev))
    return intervals

def main():
    # 1) Load the tickers to check, then drop duplicates
    logging.info(f"Loading tickers from {TICKER_CSV}")
    meta = pd.read_csv(
        TICKER_CSV,
        parse_dates=["firstpricedate", "lastpricedate"]
    )
    meta["firstpricedate"] = meta["firstpricedate"].dt.date
    meta["lastpricedate"]  = meta["lastpricedate"].dt.date

    # Drop any duplicate ticker rows (keep the first occurrence)
    meta = meta.drop_duplicates(subset=["ticker"])
    logging.info(f"→ {len(meta)} unique tickers after deduplication")

    # 2) Load SEP and extract actual trading dates
    logging.info(f"Loading SEP from {GOLDEN_SEP}")
    sep = pd.read_parquet(GOLDEN_SEP)
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 3) Build NYSE calendar from DATA_START to today
    overall_end = datetime.today().date()
    logging.info(f"Building NYSE calendar from {DATA_START} to {overall_end}")
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(
        start_date=DATA_START.isoformat(),
        end_date=overall_end.isoformat()
    )
    trading_days_all = schedule.index.date

    # 4) For each ticker, compare actual vs expected (starting no earlier than DATA_START)
    records = []
    logging.info(f"Checking {len(meta)} tickers one by one")
    for _, row in meta.iterrows():
        tk = row["ticker"]
        first_raw = row["firstpricedate"]
        last_raw  = row["lastpricedate"] if not pd.isna(row["lastpricedate"]) else overall_end

        # Use DATA_START if the official firstpricedate is earlier
        start = first_raw if first_raw >= DATA_START else DATA_START
        end   = last_raw if last_raw <= overall_end else overall_end

        # Expected trading days for this ticker between start and end
        mask = (trading_days_all >= start) & (trading_days_all <= end)
        expected_days = set(trading_days_all[mask])

        # Actual days from SEP, clipped to [start, end]
        actual_days = set(sep.loc[sep["ticker"] == tk, "date"])
        actual_days = {d for d in actual_days if start <= d <= end}

        # Identify missing days
        missing_days = sorted(expected_days - actual_days)
        if missing_days:
            intervals = group_missing_intervals(missing_days)
            missing_str = "; ".join(
                f"{s.isoformat()}→{e.isoformat()}" for s, e in intervals
            )
            total_expected = len(expected_days)
            total_actual   = len(actual_days)
            complete = False
        else:
            missing_str = ""
            total_expected = len(expected_days)
            total_actual   = len(actual_days)
            complete = True

        records.append({
            "ticker":           tk,
            "firstpricedate":   first_raw,
            "lastpricedate":    last_raw if not pd.isna(row["lastpricedate"]) else "",
            "effective_start":  start,
            "effective_end":    end,
            "total_expected":   total_expected,
            "total_actual":     total_actual,
            "complete?":        complete,
            "missing_periods":  missing_str
        })

    # 5) Build DataFrame & write out
    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values(
        ["complete?", "ticker"],
        ascending=[False, True]
    ).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Wrote completeness report to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
