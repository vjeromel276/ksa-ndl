#!/usr/bin/env python3
import argparse
import os
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# ———————————————
# ARG PARSING
# ———————————————
parser = argparse.ArgumentParser(
    description="Backfill SEP gaps for tickers by first-letter, with resume support"
)
parser.add_argument(
    "--letter", "-l",
    required=True,
    help="First letter of tickers to process (A to Z)"
)
parser.add_argument(
    "--date", "-d",
    required=True,
    help="Date suffix of your filtered SEP (e.g. 2025-05-29)"
)
parser.add_argument(
    "--throttle", "-t",
    type=float,
    default=1.0,
    help="Seconds to sleep between API calls (default 1.0)"
)
args = parser.parse_args()
LETTER     = args.letter.upper()
SEP_DATE    = args.date
THROTTLE_SEC= args.throttle

# ———————————————
# PATHS
# ———————————————
SEP_PATH      = f"sep_dataset/SHARADAR_SEP_filtered_{SEP_DATE}.parquet"
GAPS_CSV      = f"filtered_trading_gaps_calendar_{SEP_DATE}.csv"
META_PATH     = "sep_dataset/SHARADAR_TICKERS_2.parquet"
OUT_CSV       = f"backfilled_gaps_{SEP_DATE}_{LETTER}.csv"
OUT_PARQ      = f"backfilled_gaps_{SEP_DATE}_{LETTER}.parquet"
STILL_MISS_CSV= f"still_missing_tickers_{SEP_DATE}_{LETTER}.csv"

# ———————————————
# LOGGING
# ———————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def fetch_interval(ticker, start, end):
    yf_start = start.isoformat()
    yf_end   = (end + timedelta(days=1)).isoformat()
    df = yf.Ticker(ticker).history(
        start=yf_start, end=yf_end,
        auto_adjust=False, actions=False
    ).reset_index()
    if df.empty:
        logging.warning(f"{ticker}: no data for {start}→{end}")
        return pd.DataFrame()
    df['Date'] = df['Date'].dt.date
    df = df[(df['Date']>=start)&(df['Date']<=end)]
    df = df[['Date','Open','High','Low','Close','Volume','Adj Close']]
    df.insert(0,'ticker',ticker)
    return df

def main():
    # Load inputs
    logging.info(f"Loading SEP filtered data: {SEP_PATH}")
    sep = pd.read_parquet(SEP_PATH)
    sep['date'] = pd.to_datetime(sep['date'],errors='coerce').dt.date

    logging.info(f"Loading gap intervals: {GAPS_CSV}")
    gaps = pd.read_csv(GAPS_CSV,parse_dates=['gap_start','gap_end'])
    # filter to our letter
    gaps = gaps[gaps['ticker'].str.startswith(LETTER)]
    logging.info(f"→ {len(gaps):,} total gaps for letter '{LETTER}'")

    # Load metadata for IPO/delist
    meta = pd.read_parquet(META_PATH)
    meta['firstpricedate'] = pd.to_datetime(meta['firstpricedate'],errors='coerce').dt.date
    meta['lastpricedate']  = pd.to_datetime(meta['lastpricedate'], errors='coerce').dt.date

    # Resume support: skip already‐fetched tickers
    fetched_tks = set()
    if os.path.exists(OUT_PARQ):
        logging.info(f"Resuming from existing {OUT_PARQ}")
        prev = pd.read_parquet(OUT_PARQ)
        fetched_tks = set(prev['ticker'].unique())
        logging.info(f"→ already fetched {len(fetched_tks)} tickers")
    # filter out gaps for those tickers
    gaps = gaps[~gaps['ticker'].isin(fetched_tks)]

    fetched = []
    for _, row in gaps.iterrows():
        tk    = row['ticker']
        start = row['gap_start'].date()
        end   = row['gap_end'].date()

        # IPO/delist guards
        rec = meta[meta['ticker']==tk]
        if rec.empty:
            logging.warning(f"{tk}: no metadata, skipping {start}→{end}")
            continue
        first, last = rec['firstpricedate'].iat[0], rec['lastpricedate'].iat[0]
        if start < first:
            logging.warning(f"{tk}: start {start}<IPO {first}, skipping")
            continue
        if pd.notna(last) and end>last:
            logging.warning(f"{tk}: end {end}>delist {last}, skipping")
            continue

        # fetch
        logging.info(f"{tk}: fetching {start}→{end}")
        try:
            df = fetch_interval(tk, start, end)
            if not df.empty:
                fetched.append(df)
                fetched_tks.add(tk)
        except Exception as e:
            logging.error(f"{tk}: error {e!r}")
        logging.debug(f"Sleeping {THROTTLE_SEC}s…")
        time.sleep(THROTTLE_SEC)

    # assemble
    if fetched:
        out_df = pd.concat(fetched,ignore_index=True).drop_duplicates(['ticker','Date'])
        # if resuming, append to prior file
        if os.path.exists(OUT_PARQ):
            prev = pd.read_parquet(OUT_PARQ)
            out_df = pd.concat([prev, out_df],ignore_index=True).drop_duplicates(['ticker','Date'])
        out_df.to_parquet(OUT_PARQ, index=False)
        out_df.to_csv(OUT_CSV, index=False)
        logging.info(f"Wrote {len(out_df):,} rows to {OUT_PARQ} / {OUT_CSV}")
    else:
        logging.info("No new data fetched this run.")

    # report still missing
    all_gap_tks = set(pd.read_csv(GAPS_CSV)['ticker'].str.startswith(LETTER)
                      .pipe(lambda m: pd.read_csv(GAPS_CSV)[m]['ticker']))
    still = sorted(all_gap_tks - fetched_tks)
    if still:
        pd.Series(still,name='ticker').to_csv(STILL_MISS_CSV,index=False)
        logging.warning(f"{len(still)} tickers still missing → {STILL_MISS_CSV}")
    else:
        logging.info("✅ All tickers for this letter fully backfilled!")

if __name__ == "__main__":
    main()
