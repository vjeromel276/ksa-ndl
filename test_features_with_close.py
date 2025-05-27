#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1) Load the features Parquet
feats = pd.read_parquet("sep_dataset/features_2025-05-23.parquet")

# 2) Ensure its 'date' column is datetime64
feats["date"] = pd.to_datetime(feats["date"])

# 3) Load the fully-filtered SEP snapshot just for 'close'
sep_close = pd.read_parquet(
    "sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet",
    columns=["ticker", "date", "close"]
)

# 4) Ensure its 'date' is also datetime64
sep_close["date"] = pd.to_datetime(sep_close["date"])

# 5) Merge so we restore the 'close' price
df = feats.merge(sep_close, on=["ticker", "date"], how="left")
assert "close" in df.columns, "Missing 'close' after merge!"

# 6) Check missing values
print("=== Missing values per column ===")
print(df.isna().sum().sort_values(ascending=False).head(20))
print()

# 7) Basic summary statistics
print("=== Summary statistics ===")
print(df.describe().T)
print()

# 8) Create next-day return & binary target
df = df.sort_values(["ticker", "date"])
df["next_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["target"] = (df["next_return"] > 0).astype(int)

# 9) Correlation with target
corr = df.corr()["target"].abs().sort_values(ascending=False)
print("=== Top features by |correlation| with target ===")
print(corr.head(11))
print()

# 10) Time-series cross-validated accuracy
X = df.drop(columns=["ticker", "date", "next_return", "target"])
y = df["target"].fillna(0).astype(int)

tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(n_estimators=50, max_depth=5,
                               random_state=42, n_jobs=-1)
scores = cross_val_score(model, X.fillna(0), y, cv=tscv, scoring="accuracy")

print("=== TimeSeries CV Accuracy Scores ===")
print(scores)
print("Mean accuracy:", scores.mean())
