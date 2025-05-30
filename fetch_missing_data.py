#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

# ————————————————
# CONFIGURE LOGGING
# ————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_intervals(dates):
    """
    Given a sorted list of date objects, returns a list of (start, end) tuples
    for contiguous runs—i.e. gaps >1 day start a new interval.
    """
    intervals = []
    start = prev = dates[0]
    for curr in dates[1:]:
        if (curr - prev).days > 1:
            intervals.append((start, prev))
            start = curr
        prev = curr
    intervals.append((start, prev))
    return intervals

def main():
    csv_path = "missing_dates_report_2025-05-29.csv"
    logging.info(f"Loading missing report from '{csv_path}'")
    missing = pd.read_csv(csv_path, parse_dates=['missing_date'])
    logging.info(f"→ Loaded {len(missing):,} rows across {missing['ticker'].nunique():,} tickers")

    all_chunks = []
    for ticker, grp in missing.groupby('ticker'):
        dates = sorted(grp['missing_date'].dt.date.unique())
        intervals = group_intervals(dates)
        logging.info(f"{ticker}: {len(intervals)} missing interval(s) detected")

        for start, end in intervals:
            fetch_end = end + timedelta(days=1)  # yfinance end is exclusive
            logging.info(f"{ticker}: fetching {start} → {end}")
            try:
                hist = yf.Ticker(ticker).history(
                    start=start.isoformat(),
                    end=fetch_end.isoformat(),
                    auto_adjust=False,
                    actions=False
                )
            except Exception as e:
                logging.error(f"{ticker}: fetch error – {e!r}, skipping this interval")
                continue

            if hist.empty:
                logging.warning(f"{ticker}: no data returned for {start}→{end}")
                continue

            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce').dt.date
            if hist['Date'].isnull().all():
                logging.warning(f"{ticker}: 'Date' not datetime-like; skipping interval")
                continue

            sel = hist[hist['Date'].isin(dates)]
            logging.info(f"→ {len(sel):,} matched rows in {start}→{end}")

            if sel.empty:
                continue

            sel = sel[['Date','Open','High','Low','Close','Volume','Adj Close']].copy()
            sel.insert(0, 'ticker', ticker)
            all_chunks.append(sel)

    # assemble final DF
    if all_chunks:
        out_df = pd.concat(all_chunks, ignore_index=True)
    else:
        out_df = pd.DataFrame(columns=[
            'ticker','Date','Open','High','Low','Close','Volume','Adj Close'
        ])

    logging.info(f"Compiled final DataFrame with {len(out_df):,} rows")
    out_name = f"missing_data_fetched_{datetime.today().date().isoformat()}.csv"
    out_df.to_csv(out_name, index=False)
    logging.info(f"Wrote output to '{out_name}'")

if __name__ == "__main__":
    main()
