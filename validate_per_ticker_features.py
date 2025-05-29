#!/usr/bin/env python3
import pandas as pd

# 1) Load SEP (fully filtered) and Features (per‐ticker)
sep = pd.read_parquet(
    "sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-27.parquet",
    columns=["ticker","date"]
)
sep["date"] = pd.to_datetime(sep["date"])

features = pd.read_parquet(
    "sep_dataset/features_per_ticker_2025-05-27.parquet"
)
features["date"] = pd.to_datetime(features["date"])

# 2) Build date‐sets per ticker
sep_dates   = sep.groupby("ticker")["date"].apply(set)
feat_dates  = features.groupby("ticker")["date"].apply(set)

# 3) Check date alignment per ticker
rows = []
for ticker, sd in sep_dates.items():
    fd = feat_dates.get(ticker, set())
    missing = sd - fd
    extra   = fd - sd
    rows.append({
        "ticker": ticker,
        "sep_count":   len(sd),
        "feat_count":  len(fd),
        "missing":     len(missing),
        "extra":       len(extra),
        "aligned":     len(missing)==0 and len(extra)==0
    })
align_df = pd.DataFrame(rows)
print("\n=== DATE ALIGNMENT SUMMARY ===")
print("Tickers total: ", len(align_df))
print("Fully aligned:", align_df.aligned.sum())
print("Not aligned:  ", len(align_df) - align_df.aligned.sum())
print(align_df[["ticker","sep_count","feat_count","missing","extra","aligned"]]
      .query("not aligned"))
# (optionally write to CSV)
# align_df.to_csv("date_alignment_report.csv", index=False)

# 4) Compute overall feature coverage per ticker
feat_cols = [c for c in features.columns if c not in ("ticker","date")]
cov_rows = []
for ticker, grp in features.groupby("ticker"):
    total_days = len(sep_dates[ticker])
    possible   = total_days * len(feat_cols)
    nonnull    = grp[feat_cols].notna().sum().sum()
    cov_rows.append({
        "ticker":        ticker,
        "possible_cells":possible,
        "nonnull_cells": nonnull,
        "coverage_pct":  nonnull/possible * 100
    })
cov_df = pd.DataFrame(cov_rows)
print("\n=== FEATURE COVERAGE SUMMARY ===")
print("Tickers total:         ", len(cov_df))
print("Perfect coverage (>99.9%):", (cov_df.coverage_pct >= 99.9).sum())
print("Below 100%:            ", (cov_df.coverage_pct < 100).sum())
print(cov_df.sort_values("coverage_pct").head(10))
# (optionally write to CSV)
# cov_df.to_csv("feature_coverage_report.csv", index=False)
