#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    # 1) Load your FEATURE Parquet
    #    Make sure this path matches your pipeline's feature files
    feat_path = "sep_dataset/features_per_ticker_2025-05-30.parquet"
    logger.info("Loading feature DataFrame from %s", feat_path)
    df = pd.read_parquet(feat_path)

    # 2) Data-quality checks
    logger.info("=== Missing values per column ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(20))
    print()

    logger.info("=== Basic summary statistics ===")
    print(df.describe().T)
    print()

    # 3) Build a simple “next-day” return target
    #    Assumes your feature DF includes columns at least: ['ticker','date','close']
    #    If not, adjust to whatever “close” column you have in that feature DF.
    df = df.sort_values(["ticker", "date"])
    df["next_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df["target"] = (df["next_return"] > 0).astype(int)

    # 4) Compute correlation of numeric features vs. target
    #    We should drop any non-numeric columns before calling .corr()
    numeric_df = df.select_dtypes(include=[np.number, "float64", "int64"])
    corr_series = numeric_df.corr()["target"].abs().sort_values(ascending=False)
    logger.info("=== Top 10 features by correlation with target ===")
    print(corr_series.head(11))  # includes 'target' itself at top
    print()

    # 5) Prepare X and y for TimeSeriesSplit CV
    #    Drop: ticker (object), date (datetime), next_return (float), target (int)
    drop_cols = ["ticker", "date", "next_return", "target"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["target"].fillna(0).astype(int)

    # 6) TimeSeries CV with RandomForest
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    logger.info("=== Running TimeSeriesSplit CV ===")
    # Fill any remaining NaN’s with 0 (or use another imputation strategy)
    X_filled = X.fillna(0)
    scores = cross_val_score(model, X_filled, y, cv=tscv, scoring="accuracy")
    logger.info("=== TimeSeries CV Accuracy Scores ===")
    print(scores)
    print("Mean accuracy:", scores.mean())


if __name__ == "__main__":
    main()
