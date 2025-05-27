#!/usr/bin/env python3
"""
validate_with_meta.py

Vectorized validation of per-ticker feature coverage against metadata.

Usage:
    python validate_with_meta.py
"""
import pandas as pd
import numpy as np

def main():
    # 1) Load SEP calendar (fully filtered) and Features (per-ticker)
    sep = pd.read_parquet(
        "sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet",
        columns=["ticker", "date"]
    )
    sep["date"] = pd.to_datetime(sep["date"])

    features = pd.read_parquet(
        "sep_dataset/features_per_ticker_2025-05-23.parquet"
    )
    features["date"] = pd.to_datetime(features["date"])

    # 2) Load & collapse metadata to one row per ticker
    meta = pd.read_csv(
        "original_files/SHARADAR_TICKERS_2.csv",
        usecols=["ticker", "firstpricedate", "lastpricedate"],
        parse_dates=["firstpricedate", "lastpricedate"]
    )
    meta_unique = (
        meta
        .groupby("ticker", as_index=True)
        .agg({
            "firstpricedate": "min",
            "lastpricedate":  "max"
        })
    )

    # 3) Build global trading calendar
    calendar = np.sort(sep["date"].unique())

    # 4) Compute possible_dates per ticker via searchsorted
    fpd = meta_unique["firstpricedate"].values.astype("datetime64[ns]")
    lpd = meta_unique["lastpricedate"].values.astype("datetime64[ns]")
    left_idxs  = np.searchsorted(calendar, fpd, side="left")
    right_idxs = np.searchsorted(calendar, lpd, side="right")
    meta_unique["possible_dates"] = (right_idxs - left_idxs).astype(int)

    # 5) Compute actual_dates from features
    feat_counts = (
        features
        .groupby("ticker")["date"]
        .nunique()
        .rename("actual_dates")
    )

    # 6) Join metadata with actual counts
    report = meta_unique.join(feat_counts, how="inner")

    # 7) Date-alignment flag
    report["date_aligned"] = report["possible_dates"] == report["actual_dates"]

    # 8) Compute non-null feature-cell counts per ticker
    feat_cols = [c for c in features.columns if c not in ("ticker", "date")]
    nonnull_per_feature = (
        features[feat_cols]
        .notna()
        .groupby(features["ticker"])
        .sum()
    )
    report["nonnull_cells"] = nonnull_per_feature.sum(axis=1).astype(int)

    # 9) Compute possible_cells and coverage_pct
    n_feats = len(feat_cols)
    report["possible_cells"] = report["possible_dates"] * n_feats
    report["coverage_pct"] = (
        report["nonnull_cells"] / report["possible_cells"] * 100
    )

    # 10) Print summaries
    total = len(report)
    aligned = report["date_aligned"].sum()
    not_aligned = total - aligned

    print("\n=== DATE ALIGNMENT SUMMARY ===")
    print(f"Tickers total:        {total}")
    print(f"Fully aligned:        {aligned}")
    print(f"Not aligned:          {not_aligned}")
    if not_aligned:
        print("\nSample of mis-aligned tickers:")
        print(report[~report.date_aligned]
              [["possible_dates","actual_dates"]]
              .head(10))

    print("\n=== FEATURE COVERAGE SUMMARY ===")
    perfect = (report.coverage_pct >= 99.9).sum()
    below100 = (report.coverage_pct < 100).sum()
    print(f"Perfect ≥99.9%:       {perfect}")
    print(f"Below 100%:           {below100}")
    print("\nBottom 10 tickers by coverage:")
    print(report.sort_values("coverage_pct")
          [["coverage_pct","possible_cells","nonnull_cells"]]
          .head(10))

if __name__ == "__main__":
    main()
