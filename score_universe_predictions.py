#!/usr/bin/env python3
"""
score_universe_predictions.py

Given a SEP snapshot (filtered), a predictions Parquet (all_preds_<date>.parquet),
and a horizon (e.g. 5d), compute actual future returns for each ticker and score:

- actual_return  = (close_{t+h} / close_t) - 1
- actual_sign    = 1 if actual_return > 0 else 0
- correct        = (signal_long == actual_sign)
- return_error   = pred_return - actual_return

Outputs a CSV (and optional Parquet) with one row per ticker:
[ticker, prob_long, pred_return, actual_return, actual_sign, correct, return_error]
"""
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(
        description="Score a universe of horizon-based predictions against actual SEP data"
    )
    p.add_argument(
        "--sep", 
        required=True, 
        help="Filtered SEP Parquet (e.g., sep_dataset/SHARADAR_SEP_filtered_2025-05-29.parquet)"
    )
    p.add_argument(
        "--predictions", 
        required=True, 
        help="Predictions Parquet (all_preds_<date>.parquet) containing at least: "
             "ticker, prob_long, pred_<horizon>_return, signal_long"
    )
    p.add_argument(
        "--horizon", 
        choices=["1d","5d","10d","30d"], 
        default="5d",
        help="Forecast horizon (e.g. 5d). The script will shift close by this many trading days."
    )
    p.add_argument(
        "--date", 
        required=True, 
        help="As-of date of the predictions (YYYY-MM-DD)"
    )
    p.add_argument(
        "--output", 
        default=None, 
        help="Path for the output CSV (defaults to scored_<date>_<horizon>.csv)"
    )
    return p.parse_args()

def main():
    opts = parse_args()
    # parse as-of date
    asof = pd.to_datetime(opts.date).date()

    # derive integer horizon N (strip trailing 'd' and convert to int)
    try:
        N = int(opts.horizon.rstrip("d"))
    except ValueError:
        logger.error("Unable to parse integer from horizon '%s'", opts.horizon)
        return

    # 1) Load SEP data and keep only necessary columns
    logger.info("Loading SEP from %s", opts.sep)
    sep = pd.read_parquet(opts.sep, columns=["ticker","date","close"])
    # ensure date is datetime.date
    sep["date"] = pd.to_datetime(sep["date"], errors="coerce").dt.date

    # 2) Sort & shift each ticker's close by N rows to get future_close_N
    sep = sep.sort_values(["ticker","date"])
    # groupby ticker → create future_close column
    sep["future_close"] = (
        sep
        .groupby("ticker")["close"]
        .shift(-N)
    )

    # 3) Extract rows where date == asof
    df_asof = sep[sep["date"] == asof].copy()
    if df_asof.empty:
        logger.error("No SEP rows found for as-of date %s. Check your SEP file.", asof)
        return
    # keep only ticker, as-of close, and future_close
    df_asof = df_asof[["ticker","close","future_close"]].rename(
        columns={"close":"close_asof"}
    )
    # compute actual_return and actual_sign
    df_asof["actual_return"] = (
        df_asof["future_close"] / df_asof["close_asof"] - 1.0
    )
    df_asof["actual_sign"] = (df_asof["actual_return"] > 0).astype(int)

    # 4) Load predictions Parquet
    logger.info("Loading predictions from %s", opts.predictions)
    preds = pd.read_parquet(opts.predictions)
    # Expect at least: ticker, prob_long, signal_long, and pred_<horizon>_return
    ret_col = f"pred_{opts.horizon}_return"
    required_cols = {"ticker","prob_long","signal_long", ret_col}
    missing_cols = required_cols - set(preds.columns)
    if missing_cols:
        logger.error(
            "Predictions file is missing required columns: %s",
            sorted(missing_cols)
        )
        return

    # 5) Merge predictions with actuals on ticker
    df_merge = pd.merge(
        preds,
        df_asof[["ticker","close_asof","future_close","actual_return","actual_sign"]],
        on="ticker",
        how="left"
    )

    # 6) Compute scoring metrics
    #   - correct: did signal_long == actual_sign?
    #   - return_error: pred_return - actual_return
    df_merge["correct"] = (
        df_merge["signal_long"].fillna(0).astype(int) 
        == df_merge["actual_sign"].fillna(-1).astype(int)
    )
    df_merge["return_error"] = (
        df_merge[ret_col].astype(float) 
        - df_merge["actual_return"].astype(float)
    )

    # 7) Assemble final DataFrame
    out_cols = [
        "ticker",
        "prob_long",
        "signal_long",
        ret_col,
        "actual_return",
        "actual_sign",
        "correct",
        "return_error"
    ]
    df_out = df_merge.loc[:, out_cols].copy()

    # 8) Write to CSV (and Parquet if you like)
    default_out = f"scored_{opts.date}_{opts.horizon}.csv"
    out_path = opts.output or default_out
    logger.info("Writing %d rows to %s", len(df_out), out_path)
    df_out.to_csv(out_path, index=False)

    # Also write a Parquet alongside (optional, change or remove if you don’t want it)
    pq_path = out_path.replace(".csv", ".parquet")
    try:
        df_out.to_parquet(pq_path, index=False)
        logger.info("Also wrote Parquet → %s", pq_path)
    except Exception as e:
        logger.warning("Could not write Parquet: %s", e)

    logger.info("Scoring complete.")


if __name__ == "__main__":
    main()
