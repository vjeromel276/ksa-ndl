#!/usr/bin/env python3
import os
import sys
import argparse

import pandas as pd
import pandas_market_calendars as mcal

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

    # 1) Load master SEP (only ticker, date, volume needed here)
    df = pd.read_parquet(
        opts.sep_master,
        columns=["ticker", "date", "volume"]
    )
    print(f"[INFO] Loaded {opts.sep_master} ({df.shape[0]} rows)")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 2) Load metadata for listing/delisting dates and filtering to common stock
    meta = pd.read_parquet(
        opts.meta_table,
        columns=["ticker", "exchange", "category", "firstpricedate", "lastpricedate"]
    )
    # Filter to common‐stock only
    is_common = meta["category"].str.contains("Common Stock", case=False, na=False)
    valid_tickers = meta.loc[is_common, "ticker"]
    df = df[df["ticker"].isin(valid_tickers)]
    print(f"[INFO] Restricted to {df['ticker'].nunique()} common‐stock tickers")

    # Parse list/delist dates
    meta = meta.rename(columns={
        "firstpricedate": "listed",
        "lastpricedate":  "delisted"
    })
    meta["listed"]   = pd.to_datetime(meta["listed"],   errors="coerce").dt.date
    meta["delisted"] = pd.to_datetime(meta["delisted"], errors="coerce").dt.date

    # 3) Build full NYSE calendar
    global_start = df["date"].min()
    global_end   = df["date"].max()
    nyse         = mcal.get_calendar("NYSE")
    sched        = nyse.schedule(
        start_date=global_start.isoformat(),
        end_date=global_end.isoformat()
    )
    all_days     = sorted(sched.index.date)

    # 4) Compute coverage per ticker
    records = []
    for ticker, sub in df.groupby("ticker"):
        row = meta[meta["ticker"] == ticker]
        listed   = row["listed"].iat[0]   if not row.empty and pd.notna(row["listed"].iat[0]) else global_start
        delisted = row["delisted"].iat[0] if not row.empty and pd.notna(row["delisted"].iat[0]) else global_end

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

    # 5) Average volume & merge
    avg_vol = df.groupby("ticker")["volume"].mean().reset_index(name="avg_volume")
    cov_vol = cov_df.merge(avg_vol, on="ticker", how="left")
    cov_vol.to_csv(opts.out_vol, index=False)
    print(f"[INFO] Wrote coverage+volume metrics → {opts.out_vol}")

    # 6) Apply universe filters
    clean = cov_vol[
        (cov_vol["coverage"] >= opts.cov_thresh) &
        (cov_vol["avg_volume"] >= opts.vol_thresh)
    ]
    clean.to_csv(opts.out_universe, index=False)
    print(f"[INFO] Wrote clean ticker universe → {opts.out_universe}")

    # 7) Summary
    n_all   = len(cov_vol)
    n_keep  = len(clean)
    print("\n[SUMMARY]")
    print(f"  ≥{int(opts.cov_thresh*100)}% coverage : {cov_vol[cov_vol.coverage>=opts.cov_thresh].shape[0]}/{n_all}")
    print(f"  ≥{opts.vol_thresh:,} avg vol :    {cov_vol[cov_vol.avg_volume>=opts.vol_thresh].shape[0]}/{n_all}")
    print(f"  Final universe : {n_keep}/{n_all} tickers")

if __name__ == "__main__":
    main()
