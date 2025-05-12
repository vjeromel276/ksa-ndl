#!/usr/bin/env python3
import os
import sys
import argparse

import pandas as pd
import pandas_market_calendars as mcal

def main(args=None):
    # ── ARGPARSE ─────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser(
        description="Compute per-ticker coverage and volume metrics"
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
        default=0.99,
        help="Coverage threshold (0.0-1.0)"
    )
    p.add_argument(
        "--vol-thresh",
        type=int,
        default=100_000,
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
    opts = p.parse_args(args or [])

    SEP_MASTER       = opts.sep_master
    TICKERS_META     = opts.meta_table
    COVERAGE_THRESH  = opts.cov_thresh
    VOLUME_THRESH    = opts.vol_thresh
    OUT_COV          = opts.out_coverage
    OUT_VOL          = opts.out_vol
    OUT_UNIV         = opts.out_universe

    # 1) Load master SEP and parse dates
    df = pd.read_parquet(SEP_MASTER, columns=["ticker","date","volume"])
    print(f"[INFO] Loaded {SEP_MASTER} ({df.shape[0]} rows)")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"[INFO] Parsed date column ({df['date'].min()} to {df['date'].max()})")

    # 2) Load ticker metadata for filtering
    meta = pd.read_parquet(
        TICKERS_META,
        columns=["ticker","exchange","category"]
    )
    print(f"[INFO] Unique exchanges: {meta['exchange'].unique()}")
    print(f"[INFO] Unique categories: {pd.unique(meta['category'])[:10]}")

    # 3) Filter to common-stock tickers
    is_common = meta["category"].str.contains("Common Stock", case=False, na=False)
    valid = meta.loc[is_common, "ticker"].unique()
    print(f"[INFO] Common Stock tickers: {len(valid)} of {meta.shape[0]}")
    if len(valid) == 0:
        sys.exit("[ERROR] No Common Stock tickers found. Check your metadata filters.")

    before = df.shape[0]
    df = df[df["ticker"].isin(valid)]
    after  = df.shape[0]
    print(f"[INFO] Restricted SEP from {before}→{after} rows for common stocks")

    # 4) Load price-date bounds
    meta_dates = pd.read_parquet(
        TICKERS_META,
        columns=["ticker","firstpricedate","lastpricedate"]
    ).rename(columns={
        "firstpricedate": "listed",
        "lastpricedate":  "delisted"
    })
    meta_dates["listed"]   = pd.to_datetime(meta_dates["listed"],   errors="coerce").dt.date
    meta_dates["delisted"] = pd.to_datetime(meta_dates["delisted"], errors="coerce").dt.date

    # 5) Build NYSE calendar
    nyse         = mcal.get_calendar("NYSE")
    global_start = df["date"].min()
    global_end   = df["date"].max()
    sched        = nyse.schedule(
        start_date=global_start.isoformat(),
        end_date=global_end.isoformat()
    )
    all_days     = sorted(sched.index.date)

    # 6) Compute coverage per ticker
    cov = []
    for ticker, sub in df.groupby("ticker"):
        row = meta_dates[meta_dates["ticker"] == ticker]
        listed   = row["listed"].iat[0]   if not row.empty and pd.notna(row["listed"].iat[0]) else global_start
        delisted = row["delisted"].iat[0] if not row.empty and pd.notna(row["delisted"].iat[0]) else global_end

        win_start     = max(listed, global_start)
        win_end       = min(delisted, global_end)
        expected_days = [d for d in all_days if win_start <= d <= win_end]
        have_days     = [d for d in sub["date"] if win_start <= d <= win_end]
        coverage      = len(have_days) / len(expected_days) if expected_days else 0.0

        cov.append({
            "ticker":        ticker,
            "listed":        listed,
            "delisted":      delisted,
            "win_start":     win_start,
            "win_end":       win_end,
            "have_days":     len(have_days),
            "expected_days": len(expected_days),
            "coverage":      coverage
        })

    cov_df = pd.DataFrame(cov).sort_values("coverage", ascending=False)

    # 7) Export coverage metrics
    cov_df.to_csv(OUT_COV, index=False)
    print(f"[INFO] Wrote {OUT_COV} ({len(cov_df)} rows)")

    # 8) Compute & merge average volume
    avg_vol = df.groupby("ticker")["volume"].mean().reset_index(name="avg_volume")
    cov_vol = cov_df.merge(avg_vol, on="ticker", how="left")
    cov_vol.to_csv(OUT_VOL, index=False)
    print(f"[INFO] Wrote {OUT_VOL} ({len(cov_vol)} rows)")

    # 9) Filter by coverage & volume thresholds
    clean = cov_vol[
        (cov_vol["coverage"] >= COVERAGE_THRESH) &
        (cov_vol["avg_volume"] >= VOLUME_THRESH)
    ].copy()
    clean.to_csv(OUT_UNIV, index=False)
    print(f"[INFO] Wrote {OUT_UNIV} ({len(clean)} tickers)")

    # 10) Summary
    print("\nFilter summary:")
    print(f"  ≥{int(COVERAGE_THRESH*100)}% coverage:  {cov_vol[cov_vol.coverage>=COVERAGE_THRESH].shape[0]}")
    print(f"  ≥{VOLUME_THRESH:,} avg volume: {cov_vol[cov_vol.avg_volume>=VOLUME_THRESH].shape[0]}")
    print(f"  Both criteria: {len(clean)} tickers")
    print(f"  Filtered out : {len(cov_vol) - len(clean)} tickers")
    print(f"  Total tickers : {len(cov_vol)}")
    print(f"  Clean coverage: {len(clean) / len(cov_vol):.2%}")
    print(f"  Clean avg vol : {clean['avg_volume'].mean():,.0f}")

if __name__ == "__main__":
    main(sys.argv[1:])
