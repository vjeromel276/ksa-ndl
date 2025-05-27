#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1) Load your feature set
feats = pd.read_parquet("sep_dataset/features_2025-05-23.parquet")
feats["date"] = pd.to_datetime(feats["date"])

# 2) Load the SEP snapshot (close price)
sep_close = pd.read_parquet(
    "sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet",
    columns=["ticker", "date", "close"]
)
sep_close["date"] = pd.to_datetime(sep_close["date"])

# 3) Merge to restore 'close'
df = feats.merge(sep_close, on=["ticker", "date"], how="left")

# 4) Compute next-day return & binary target
df = df.sort_values(["ticker", "date"])
df["next_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["target"] = (df["next_return"] > 0).astype(int)

# 5) Report missing‐value counts
print("=== Missing values per column ===")
print(df.isna().sum().sort_values(ascending=False).head(20))
print()

# 6) Summary statistics (only numeric cols)
print("=== Summary statistics (numeric) ===")
print(df.select_dtypes(include=["number"]).describe().T)
print()

# 7) Correlation with target (numeric only)
numeric = df.select_dtypes(include=["number"]).columns.tolist()
numeric.remove("target")      # exclude the target from itself
corr = df[numeric + ["target"]].corr()["target"].abs().sort_values(ascending=False)
print("=== Top 10 features by |correlation| with target ===")
print(corr.head(11))
print()

# 8) Quick time-series CV accuracy
X = df[numeric].fillna(0)
y = df["target"]

tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
print("=== TimeSeries CV Accuracy Scores ===")
print(scores)
print("Mean accuracy:", scores.mean())
