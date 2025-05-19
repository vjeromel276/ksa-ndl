# train_and_save_models.py
# Console script to train final 5-day direction classifier and return regressor,
# then save them to disk for later use.

#!/usr/bin/env python3
# Premise: Handle root cause of errors (NaNs, dtype issues) while preserving all valid data

import argparse
import joblib
import pandas as pd
import numpy as np

from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.baseline import train_baseline_classification, train_baseline_regression


def main():
    parser = argparse.ArgumentParser(
        description="Train and save 5-day direction classifier and return regressor."
    )
    parser.add_argument(
        "--sep-master", type=str, default="sep_dataset/SHARADAR_SEP_common_v1.parquet",
        help="Path to master SEP Parquet file"
    )
    parser.add_argument(
        "--backend", choices=["dummy","xgb","torch"], default="xgb",
        help="Backend for classifier"
    )
    parser.add_argument(
        "--device", choices=["cpu","gpu"], default="gpu",
        help="Device for training"
    )
    parser.add_argument(
        "--target-col", type=str, default="dir_5d",
        help="Target column for classification (e.g., dir_5d)"
    )
    parser.add_argument(
        "--clf-out", type=str, required=True,
        help="Output path for saved classifier model (joblib)"
    )
    parser.add_argument(
        "--reg-out", type=str, required=True,
        help="Output path for saved regressor model (joblib)"
    )
    args = parser.parse_args()

    # 1) Load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    print(f"[INFO] Loaded SEP: {sep.shape[0]} rows")

    # 2) Build features
    X = load_features(sep)
    # Convert all feature columns to numeric floats
    try:
        X = X.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error converting features to numeric: {e}")
    # Drop rows with NaNs in features (leading/trailing due to rolling windows)
    initial_shape = X.shape
    X = X.dropna()
    print(f"[INFO] Dropped feature NaNs: {initial_shape} -> {X.shape}")

    # 3) Load and align targets
    y_df = load_targets(sep)
    if args.target_col not in y_df.columns:
        raise ValueError(f"Unknown classification target: {args.target_col}")
    if 'return_5d' not in y_df.columns:
        raise ValueError("Return target 'return_5d' not found in targets")

    y_class = y_df[args.target_col].reindex(X.index)
    y_reg = y_df['return_5d'].reindex(X.index)
    # Drop any rows where targets are NaN
    mask = y_class.notna() & y_reg.notna()
    X_clean = X.loc[mask]
    y_class_clean = y_class.loc[mask]
    y_reg_clean = y_reg.loc[mask]
    print(f"[INFO] After dropping target NaNs: Features: {X_clean.shape}, Classification: {y_class_clean.shape}, Regression: {y_reg_clean.shape}")

    # 4) Train classifier
    print("[INFO] Training 5-day direction classifier...")
    clf = train_baseline_classification(
        X_clean, y_class_clean,
        backend=args.backend,
        device=args.device,
        num_classes=len(pd.unique(y_class_clean))
    )
    joblib.dump(clf, args.clf_out)
    print(f"[INFO] Classifier saved to {args.clf_out}")

    # 5) Train regressor
    print("[INFO] Training 5-day return regressor...")
    reg = train_baseline_regression(
        X_clean, y_reg_clean,
        backend=args.backend,
        device=args.device
    )
    joblib.dump(reg, args.reg_out)
    print(f"[INFO] Regressor saved to {args.reg_out}")

    print("[INFO] Training and save complete.")


if __name__ == "__main__":
    main()
