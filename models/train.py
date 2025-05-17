# models/train.py
#!/usr/bin/env python3
"""
Train & evaluate a baseline forecasting model using rolling time-series CV.
"""
import argparse
import numpy as np
import pandas as pd

from core.schema      import validate_full_sep
from models.data      import load_features, load_targets, _coerce_sep_dtypes
from models.cv        import TimeSeriesSplitter
from models.baseline  import train_and_evaluate


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
    p.add_argument(
        "--backend", choices=["dummy","xgb","torch"], default="dummy",
        help="Which backend to use for the baseline model"
    )
    p.add_argument(
        "--device", choices=["cpu","gpu"], default="cpu",
        help="Where to run the model (gpu → XGBoost gpu_hist / future PyTorch)"
    )
    p.add_argument(
        "--target-col", type=str, default="dir_5d",
        help="Which target column from load_targets to train on (e.g., dir_5d, dir_10d)"
    )
    args = p.parse_args()

    # 1) Load & coerce dtypes + validate full‐SEP schema
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    print(f"[INFO] Loaded & schema‐checked SEP: {sep.shape[0]} rows")

    # 2) Build features & select target
    X = load_features(sep)
    y_df = load_targets(sep)
    if args.target_col not in y_df.columns:
        raise ValueError(f"Unknown target column: {args.target_col}. Available: {list(y_df.columns)}")
    y = y_df[args.target_col]
    print(f"[INFO] Features: {X.shape}, Targets (1D): {y.shape}, using '{args.target_col}'")

    # 3) Setup time-series splitter
    splitter = TimeSeriesSplitter(
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step
    )

    # 4) Train & evaluate
    scores = train_and_evaluate(
        X, y, splitter,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    # 5) Summary
    metric_name = "Accuracy" if args.mode == "classify" else "MAE"
    print(f"\nCV folds: {len(scores)}, Mean {metric_name}: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
