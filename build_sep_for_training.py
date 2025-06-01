#!/usr/bin/env python3
import argparse
import logging
import pandas as pd

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a SEP Parquet for training from completeness CSV + golden SEP"
    )
    p.add_argument(
        "--completeness", "-c",
        required=True,
        help="Path to ticker_history_completeness.csv"
    )
    p.add_argument(
        "--min-rows", "-m",
        type=int,
        required=True,
        help="Minimum number of SEP rows required (e.g. 252)"
    )
    p.add_argument(
        "--sep-parquet", "-s",
        required=True,
        help="Path to SHARADAR_SEP_2.parquet (golden SEP)"
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write filtered SEP Parquet (e.g. sep_training_universe.parquet)"
    )
    return p.parse_args()

def main():
    opts = parse_args()

    # 1) Load completeness CSV and filter tickers
    logging.info(f"Loading completeness data from {opts.completeness}")
    comp = pd.read_csv(
        opts.completeness,
        parse_dates=["firstpricedate","lastpricedate","effective_start","effective_end"]
    )
    comp["total_actual"] = comp["total_actual"].astype(int)

    filtered = comp[comp["total_actual"] >= opts.min_rows].copy()
    tickers_to_keep = set(filtered["ticker"])
    logging.info(f"→ {len(tickers_to_keep):,} tickers pass the ≥{opts.min_rows}-row filter")

    # 2) Load golden SEP Parquet
    logging.info(f"Loading SEP data from {opts.sep_parquet}")
    sep = pd.read_parquet(f"sep_dataset/{opts.sep_parquet}")
    # Ensure date is datetime.date
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 3) Filter SEP to only the selected tickers
    logging.info("Filtering SEP to selected tickers…")
    sep_filt = sep[sep["ticker"].isin(tickers_to_keep)].copy()
    n_rows, n_tks = sep_filt.shape[0], sep_filt["ticker"].nunique()
    logging.info(f"→ Retained {n_rows:,} rows across {n_tks:,} tickers")

    # 4) Subset to OCHLV & Adj Close
    #    Ensure these columns exist in your SEP; adjust names if needed.
    keep_cols = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj"]
    missing_cols = [c for c in keep_cols if c not in sep_filt.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in SEP: {missing_cols}")
    sep_train = sep_filt[keep_cols]

    # 5) Write out to Parquet
    logging.info(f"Writing filtered SEP to {opts.output}")
    sep_train.to_parquet(opts.output, index=False)
    logging.info(f"✅ Wrote {len(sep_train):,} rows to {opts.output}")

if __name__ == "__main__":
    main()
