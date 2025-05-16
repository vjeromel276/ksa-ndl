#!/usr/bin/env python3
import os
import sys
import argparse

import pandas as pd
import pandas_market_calendars as mcal

from core.schema import validate_volume_df

# ── PARAMETERS ────────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD = 0.99     # require ≥99% trading‐day coverage
VOLUME_THRESHOLD   = 100_000  # require ≥100k avg daily volume

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compute per‐ticker coverage and volume metrics"
    )
    p.add_argument(
        "--sep-master",
        type=str,
        default=os.environ.get("SEP_MASTER", "sep_dataset/SHARADAR_SEP.parquet"),
        help="Path to master SEP dataset (prices)"
    )
    p.add_argument(
        "--meta-table",
        type=str,
        default=os.environ.get("TICKERS_META", "sep_dataset/SHARADAR_TICKERS_2.parquet"),
        help="Path to ticker metadata table"
    )
    p.add_argument(
        "--cov-thresh",
        type=float,
        default=COVERAGE_THRESHOLD,
        help="Coverage threshold (0.0–1.0)"
    )
    p.add_argument(
        "--vol-thresh",
        type=int,
        default=VOLUME_THRESHOLD,
        help="Volume threshold (int)"
    )
    p.add_argument(
        "--out-coverage",
        type=str,
        default="ticker_coverage.csv",
        help="Output CSV for coverage metrics"
    )
    p.add_argument(
        "--out-vol",
        type=str,
        default="ticker_coverage_with_volume.csv",
        help="Output CSV for coverage+volume metrics"
    )
    p.add_argument(
        "--out-universe",
        type=str,
        default="ticker_universe_clean.csv",
        help="Output CSV for the clean ticker universe"
    )
    opts = p.parse_args(argv or [])

    # 1) Load the full SEP table, then immediately slice to just (ticker, date, volume)
    full_sep = pd.read_parquet(opts.sep_master)
    df       = full_sep[["ticker", "date", "volume"]].copy()

    # ── COERCE to the exact dtypes our volume‐slice schema expects ────────────────
    df["ticker"] = df["ticker"].astype("category")
    df["date"]   = pd.to_datetime(df["date"])
    df["volume"] = df["volume"].astype("float64")

    # 2) Validate that slice
    validate_volume_df(df)
    print(f"[INFO] Loaded & validated (ticker, date, volume) slice: {opts.sep_master} ({df.shape[0]} rows)")

    # 3) Now convert date→python date for calendar logic
    df["date"] = df["date"].dt.date

    # 4) Load metadata for listing/delisting dates and filtering to common‐stock
    meta = pd.read_parquet(
        opts.meta_table,
        columns=["ticker", "exchange", "category", "firstpricedate", "lastpricedate"]
    )
    is_common     = meta["category"].str.contains("Common Stock", case=False, na=False)
    valid_tickers = meta.loc[is_common, "ticker"]
    df = df[df["ticker"].isin(valid_tickers)]
    print(f"[INFO] Restricted to {df['ticker'].nunique()} common‐stock tickers")

    # 5) Parse listing/delisting dates
    meta = meta.rename(columns={
        "firstpricedate": "listed",
        "lastpricedate":  "delisted"
    })
    meta["listed"]   = pd.to_datetime(meta["listed"],   errors="coerce").dt.date
    meta["delisted"] = pd.to_datetime(meta["delisted"], errors="coerce").dt.date

    # 6) Build full NYSE calendar over our SEP date range
    global_start = df["date"].min()
    global_end   = df["date"].max()
    nyse         = mcal.get_calendar("NYSE")
    sched        = nyse.schedule(
        start_date=global_start.isoformat(),
        end_date=global_end.isoformat()
    )
    all_days     = sorted(sched.index.date)

    # 7) Compute coverage per ticker
    records = []
    for ticker, sub in df.groupby("ticker", observed=False):
        row = meta[meta["ticker"] == ticker]
        listed   = (row["listed"].iat[0]
                    if not row.empty and pd.notna(row["listed"].iat[0])
                    else global_start)
        delisted = (row["delisted"].iat[0]
                    if not row.empty and pd.notna(row["delisted"].iat[0])
                    else global_end)

        win_start     = max(listed, global_start)
        win_end       = min(delisted, global_end)
        expected_days = [d for d in all_days if win_start <= d <= win_end]
        have_days     = [d for d in sub["date"]    if win_start <= d <= win_end]

        coverage = len(have_days) / len(expected_days) if expected_days else 0.0
        records.append({
            "ticker":        ticker,
            "listed":        listed,
            "delisted":      delisted,
            "win_start":     win_start,
            "win_end":       win_end,
            "have_days":     len(have_days),
            "expected_days": len(expected_days),
            "coverage":      coverage
        })

    cov_df = pd.DataFrame(records).sort_values("coverage", ascending=False)
    cov_df.to_csv(opts.out_coverage, index=False)
    print(f"[INFO] Wrote coverage metrics → {opts.out_coverage}")

    # 8) Compute & merge avg volume
    avg_vol = df.groupby("ticker", observed=False)["volume"].mean().reset_index(name="avg_volume")
    cov_vol = cov_df.merge(avg_vol, on="ticker", how="left")
    cov_vol.to_csv(opts.out_vol, index=False)
    print(f"[INFO] Wrote coverage+volume metrics → {opts.out_vol}")

    # 9) Apply universe filters
    clean = cov_vol[
        (cov_vol["coverage"] >= opts.cov_thresh) &
        (cov_vol["avg_volume"] >= opts.vol_thresh)
    ]
    clean.to_csv(opts.out_universe, index=False)
    print(f"[INFO] Wrote clean ticker universe → {opts.out_universe}")

    # 10) Summary
    n_all   = len(cov_vol)
    n_keep  = len(clean)
    print("\n[SUMMARY]")
    print(f"  ≥{int(opts.cov_thresh*100)}% coverage : {cov_vol[cov_vol.coverage>=opts.cov_thresh].shape[0]}/{n_all}")
    print(f"  ≥{opts.vol_thresh:,} avg vol :    {cov_vol[cov_vol.avg_volume>=opts.vol_thresh].shape[0]}/{n_all}")
    print(f"  Final universe : {n_keep}/{n_all} tickers")

if __name__ == "__main__":
    main()
