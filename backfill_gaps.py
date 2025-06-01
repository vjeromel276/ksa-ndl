#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import logging
import time
from datetime import timedelta, date

# ———————————————
# CONFIG
# ———————————————
SEP_PATH     = "sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet"
GAPS_CSV     = "filtered_trading_gaps_calendar_2025-05-29.csv"
META_PATH    = "sep_dataset/SHARADAR_TICKERS_2.parquet"
OUT_CSV      = f"backfilled_gaps_{date.today().isoformat()}.csv"
OUT_PARQ     = f"backfilled_gaps_{date.today().isoformat()}.parquet"
STILL_MISS_CSV = f"still_missing_tickers_{date.today().isoformat()}.csv"

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
    # 1) Load filtered SEP (for context)
    logging.info(f"Loading filtered SEP from {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep['date'] = pd.to_datetime(sep['date']).dt.date

    # 2) Load gap intervals
    logging.info(f"Loading gap list from {GAPS_CSV}")
    gaps = pd.read_csv(GAPS_CSV, parse_dates=['gap_start','gap_end'])
    logging.info(f"→ {len(gaps):,} gap intervals to backfill")

    # 3) Load ticker metadata for IPO/delist dates
    logging.info(f"Loading ticker metadata from {META_PATH}")
    meta = pd.read_parquet(META_PATH)
    meta['firstpricedate'] = pd.to_datetime(meta['firstpricedate'], errors='coerce').dt.date
    meta['lastpricedate']  = pd.to_datetime(meta['lastpricedate'],  errors='coerce').dt.date

    # 4) Fetch each interval, skipping out‐of‐bounds and throttling
    fetched = []
    for idx, row in gaps.iterrows():
        tk    = row['ticker']
        start = row['gap_start'].date() if hasattr(row['gap_start'], 'date') else row['gap_start']
        end   = row['gap_end'].date()   if hasattr(row['gap_end'],   'date') else row['gap_end']

        # lookup first/last for this ticker
        rec = meta.loc[meta['ticker'] == tk]
        if rec.empty:
            logging.warning(f"{tk}: no metadata found, skipping interval {start}→{end}")
            continue
        first = rec['firstpricedate'].iat[0]
        last  = rec['lastpricedate'].iat[0]

        # skip before IPO
        if start < first:
            logging.warning(f"{tk}: gap {start}→{end} starts before IPO ({first}), skipping")
            continue
        # skip after delist (if known)
        if pd.notna(last) and end > last:
            logging.warning(f"{tk}: gap {start}→{end} ends after delist ({last}), skipping")
            continue

        logging.info(f"{tk}: fetching {start} → {end}")
        try:
            df = fetch_interval(tk, start, end)
            if not df.empty:
                fetched.append(df)
        except Exception as e:
            logging.error(f"{tk}: error fetching {start}→{end} → {e!r}")

        logging.debug(f"Sleeping for {THROTTLE_SEC}s to avoid rate limits…")
        time.sleep(THROTTLE_SEC)

    # 5) Assemble and dedupe
    if not fetched:
        logging.error("No data fetched; exiting.")
        return
    backfilled = pd.concat(fetched, ignore_index=True)
    backfilled.drop_duplicates(subset=['ticker','Date'], inplace=True)
    logging.info(f"Fetched {len(backfilled):,} backfill rows total")

    # 6) Write outputs
    backfilled.to_csv(OUT_CSV, index=False)
    logging.info(f"Wrote flat CSV → {OUT_CSV}")
    backfilled.to_parquet(OUT_PARQ, index=False)
    logging.info(f"Wrote Parquet     → {OUT_PARQ}")

    # 7) Report tickers still missing
    fetched_tks   = set(backfilled['ticker'].unique())
    all_gap_tks   = set(gaps['ticker'].unique())
    still_missing = sorted(all_gap_tks - fetched_tks)

    if still_missing:
        logging.warning(f"{len(still_missing):,} tickers still missing data: {still_missing[:10]} …")
        pd.Series(still_missing, name='ticker').to_csv(STILL_MISS_CSV, index=False)
        logging.info(f"Wrote list of tickers still missing to {STILL_MISS_CSV}")
    else:
        logging.info("✅ All tickers fully backfilled!")

if __name__ == "__main__":
    main()
