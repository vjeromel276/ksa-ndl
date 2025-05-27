#!/usr/bin/env python3
import pandas as pd

# 1) Load your per‐ticker feature Parquet
features = pd.read_parquet(
    "sep_dataset/features_per_ticker_2025-05-23.parquet"
)

# 2) Filter for OPB
opb = features[features["ticker"] == "OPB"]

# 3) Quick sanity‐checks
print("OPB shape:", opb.shape)
print("First few rows:\n", opb.head())
print("Missing values per column:\n", opb.isna().sum().sort_values(ascending=False))

# 4) (Optional) Dump to CSV for deeper exploration
opb.to_csv("OPB_features.csv", index=False)
