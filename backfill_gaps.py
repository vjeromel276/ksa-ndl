#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import logging
import time
from datetime import timedelta, date

# ———————————————
# CONFIG
# ———————————————
SEP_PATH    = "sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet"
GAPS_CSV    = "filtered_trading_gaps_calendar_2025-05-29.csv"
OUT_CSV     = f"backfilled_gaps_{date.today().isoformat()}.csv"
OUT_PARQ    = f"backfilled_gaps_{date.today().isoformat()}.parquet"

# pause in seconds between API calls
THROTTLE_SEC = 3.0

# ———————————————
# LOGGING
# ———————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def fetch_interval(ticker, start, end):
    """
    Fetch history for ticker from start to end (inclusive).
    Returns DataFrame with Date, O/H/L/C/Vol/Adj Close.
    """
    yf_start = start.isoformat()
    yf_end   = (end + timedelta(days=1)).isoformat()  # exclusive
    df = yf.Ticker(ticker).history(
        start=yf_start,
        end=yf_end,
        auto_adjust=False,
        actions=False
    ).reset_index()

    if df.empty:
        logging.warning(f"{ticker}: no data for {start} → {end}")
        return pd.DataFrame()

    df['Date'] = df['Date'].dt.date
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    df = df.loc[mask, ['Date','Open','High','Low','Close','Volume','Adj Close']]
    df.insert(0, 'ticker', ticker)
    return df

def main():
    # 1) load your filtered SEP (for context)
    logging.info(f"Loading filtered SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep['date'] = pd.to_datetime(sep['date']).dt.date

    # 2) load gap intervals
    logging.info(f"Loading gap list from {GAPS_CSV}")
    gaps = pd.read_csv(GAPS_CSV, parse_dates=['gap_start','gap_end'])
    logging.info(f"→ {len(gaps):,} gap intervals to backfill")

    # 3) fetch each interval, throttling in between
    fetched = []
    for idx, row in gaps.iterrows():
        tk    = row['ticker']
        start = row['gap_start'].date() if hasattr(row['gap_start'], 'date') else row['gap_start']
        end   = row['gap_end'].date()   if hasattr(row['gap_end'],   'date') else row['gap_end']
        logging.info(f"{tk}: fetching {start} → {end}")
        try:
            df = fetch_interval(tk, start, end)
            if not df.empty:
                fetched.append(df)
        except Exception as e:
            logging.error(f"{tk}: error fetching {start}→{end} → {e!r}")
        logging.debug(f"Sleeping for {THROTTLE_SEC}s to avoid rate limits…")
        time.sleep(THROTTLE_SEC)

    # 4) assemble and dedupe
    if not fetched:
        logging.error("No data fetched; exiting.")
        return
    backfilled = pd.concat(fetched, ignore_index=True)
    backfilled.drop_duplicates(subset=['ticker','Date'], inplace=True)
    logging.info(f"Fetched {len(backfilled):,} backfill rows total")

    # 5) write outputs
    backfilled.to_csv(OUT_CSV, index=False)
    logging.info(f"Wrote flat CSV → {OUT_CSV}")
    backfilled.to_parquet(OUT_PARQ, index=False)
    logging.info(f"Wrote Parquet     → {OUT_PARQ}")
     # 6) REPORT TICKERS STILL MISSING
    fetched_tks   = set(backfilled['ticker'].unique())
    all_gap_tks   = set(gaps['ticker'].unique())
    still_missing = sorted(all_gap_tks - fetched_tks)

    if still_missing:
        logging.warning(f"{len(still_missing):,} tickers still missing data: {still_missing[:10]} …")
        # write them out
        pd.Series(still_missing, name='ticker')\
          .to_csv(f"still_missing_tickers_{date.today().isoformat()}.csv", index=False)
        logging.info("Wrote list of tickers still missing to " +
                     f"still_missing_tickers_{date.today().isoformat()}.csv")
    else:
        logging.info("✅ All tickers fully backfilled!")

if __name__ == "__main__":
    main()
