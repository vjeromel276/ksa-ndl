#!/usr/bin/env python3
"""
Vectorized validation of per‐ticker feature coverage against metadata.
"""
import pandas as pd
import numpy as np

# 1) Load data
sep = pd.read_parquet(
    "sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet",
    columns=["ticker","date"]
)
sep["date"] = pd.to_datetime(sep["date"])

features = pd.read_parquet(
    "sep_dataset/features_per_ticker_2025-05-23.parquet"
)
features["date"] = pd.to_datetime(features["date"])

meta = pd.read_csv(
    "original_files/SHARADAR_TICKERS_2.csv",
    usecols=["ticker","firstpricedate","lastpricedate"],
    parse_dates=["firstpricedate","lastpricedate"]
).set_index("ticker")

# 2) Build global trading calendar
calendar = np.sort(sep["date"].unique())

# 3) Compute possible_dates per ticker via searchsorted
fpd = meta["firstpricedate"].values
lpd = meta["lastpricedate"].values
left_idxs  = np.searchsorted(calendar, fpd, side="left")
right_idxs = np.searchsorted(calendar, lpd, side="right")
meta["possible_dates"] = right_idxs - left_idxs

# 4) Compute actual_dates from features
feat_counts = (
    features
    .groupby("ticker")["date"]
    .nunique()
    .rename("actual_dates")
)
# Align meta to only tickers we generated features for
report = meta.join(feat_counts, how="inner")

# 5) Date‐alignment flag
report["date_aligned"] = report["possible_dates"] == report["actual_dates"]

# 6) Feature coverage: non‐null cells per ticker
feat_cols = [c for c in features.columns if c not in ("ticker","date")]
nonnull_per_feature = (
    features[feat_cols]
    .notna()
    .groupby(features["ticker"])
    .sum()
)
report["nonnull_cells"] = nonnull_per_feature.sum(axis=1).astype(int)

# 7) Possible cells = possible_dates × number of features
n_feats = len(feat_cols)
report["possible_cells"] = report["possible_dates"] * n_feats

# 8) Coverage percentage
report["coverage_pct"] = report["nonnull_cells"] / report["possible_cells"] * 100

# 9) Summaries
print("\n=== DATE ALIGNMENT SUMMARY ===")
print(f"Tickers total:        {len(report)}")
print(f"Fully aligned:        {report.date_aligned.sum()}")
print(f"Not aligned:          {len(report) - report.date_aligned.sum()}")
print(report[~report.date_aligned]
      [["possible_dates","actual_dates"]]
      .head(10), "\n")

print("=== FEATURE COVERAGE SUMMARY ===")
print(f"Perfect ≥99.9%:       {(report.coverage_pct >= 99.9).sum()}")
print(f"Below 100%:           {(report.coverage_pct < 100).sum()}")
print(report.sort_values("coverage_pct").head(10)
      [["coverage_pct","possible_cells","nonnull_cells"]])
