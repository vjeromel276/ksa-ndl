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

from core.schema          import validate_sep_df
from models.data          import load_features, load_targets
from models.cv            import TimeSeriesSplitter
from models.baseline      import train_and_evaluate

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
    splitter = TimeSeriesSplitter(
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step
    )

    # 4) Train & evaluate
    scores = train_and_evaluate(X, y, splitter, mode=args.mode)

    # 5) Summary
    metric_name = "Accuracy" if args.mode == "classify" else "MAE"
    mean_score = np.mean(scores) if scores else float("nan")
    print(f"\nCV folds: {len(scores)}, Mean {metric_name}: {mean_score:.4f}")

if __name__ == "__main__":
    main()
