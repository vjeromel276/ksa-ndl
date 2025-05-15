#!/usr/bin/env python3
"""
train.py

Train & evaluate a baseline forecasting model using rolling time-series CV.

Usage:
  python models/train.py \
    --sep-master sep_dataset/SHARADAR_SEP.parquet \
    --train-window 252 \
    --test-window 21 \
    --step 21 \
    --mode classify
"""

import argparse
import numpy as np
import pandas as pd

from core.schema      import validate_sep_df
from models.data      import load_features, load_targets
from models.cv        import TimeSeriesSplitter
from models.baseline  import train_baseline_classification, train_baseline_regression  # :contentReference[oaicite:1]{index=1}
from models.metrics   import return_accuracy, regression_mse, regression_mae

def main():
    p = argparse.ArgumentParser(description="Train & CV a baseline quant model")
    p.add_argument(
        "--sep-master", type=str,
        default="sep_dataset/SHARADAR_SEP.parquet",
        help="Master SEP Parquet (prices) to load"
    )
    p.add_argument(
        "--train-window", type=int, default=252,
        help="Number of days for each training window"
    )
    p.add_argument(
        "--test-window", type=int, default=21,
        help="Number of days for each test window"
    )
    p.add_argument(
        "--step", type=int, default=21,
        help="Step in days to advance each fold"
    )
    p.add_argument(
        "--mode", choices=["classify","regress"], default="classify",
        help="Train classification (direction) or regression (returns)"
    )
    args = p.parse_args()

    # 1) Load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    validate_sep_df(sep)
    print(f"[INFO] Loaded SEP: {sep.shape[0]} rows")

    # 2) Build features X and targets y
    X = load_features(sep)
    y = load_targets(sep)
    print(f"[INFO] Features: {X.shape}, Targets: {y.shape}")

    # 3) Setup time-series splitter
    tss = TimeSeriesSplitter(
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step
    )

    scores = []
    fold = 0
    # We need a DataFrame to split on — use X (it has 'date' level) 
    # so that train/test retains the same rows in both X and y.
    X_df = X.reset_index()  # columns: ticker, date, ...
    for train_idx, test_idx in tss.split(X_df, date_col="date"):
        fold += 1
        X_tr = X_df.iloc[train_idx].set_index(["ticker","date"])
        X_te = X_df.iloc[test_idx].set_index(["ticker","date"])
        y_tr = y.loc[X_tr.index]
        y_te = y.loc[X_te.index]

        # 4) Train
        if args.mode == "classify":
            model = train_baseline_classification(X_tr, y_tr["dir_5d"])
        else:
            model = train_baseline_regression(X_tr, y_tr["return_5d"])

        # 5) Predict & evaluate
        preds = model.predict(X_te)
        if args.mode == "classify":
            score = return_accuracy(y_te["dir_5d"], preds)
            metric_name = "Accuracy"
        else:
            score = regression_mae(y_te["return_5d"], preds)
            metric_name = "MAE"

        print(f"[Fold {fold}] {metric_name} = {score:.4f}")
        scores.append(score)

    # 6) Summary
    mean_score = np.mean(scores) if scores else float("nan")
    print(f"\nCV folds: {fold}, Mean {metric_name}: {mean_score:.4f}")

if __name__ == "__main__":
    main()
