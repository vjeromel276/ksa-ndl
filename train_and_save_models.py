# train_and_save_models.py
# Console script to train final direction classifier and return regressor,
# then save them to disk for later use.

#!/usr/bin/env python3
# Premise: Handle root cause of errors (NaNs, dtype issues) while preserving all valid data

import argparse
import logging
import sys
import joblib
import pandas as pd
import numpy as np

from models.cherry_picker import get_valid_tickers_for_horizon
from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.baseline import train_baseline_classification, train_baseline_regression

# ——————————————————————————————————————————————————————————————————
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train and save direction classifier and return regressor."
    )
    parser.add_argument(
        "--sep-master", type=str,
        default="sep_dataset/SHARADAR_SEP_common_v1.parquet",
        help="Path to master SEP Parquet file"
    )
    parser.add_argument(
        "--universe-csv", type=str, required=True,
        help="Path to ticker_universe_clean_<date>.csv"
    )
    parser.add_argument(
        "--horizon", choices=["1d","5d","10d","30d"], default="5d",
        help="Prediction horizon: cherry-pick tickers with sufficient history"
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

    # 1) Load, coerce dtypes, and validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded & validated SEP: {sep.shape[0]} rows")

    # 2) Cherry-pick tickers for this horizon
    logger.info(f"Selecting tickers for {args.horizon} prediction…")
    tickers = get_valid_tickers_for_horizon(
        universe_csv=args.universe_csv,
        horizon=args.horizon
    )
    if not tickers:
        logger.error(f"No tickers meet the requirement for horizon '{args.horizon}' – aborting")
        sys.exit(1)
    logger.info(f"{len(tickers)} tickers passed the {args.horizon} filter")

    sep = sep[sep["ticker"].isin(tickers)]
    logger.info(f"After cherry-pick filter: SEP shape = {sep.shape}")

    # 3) Build features
    X = load_features(sep)
    logger.info(f"Raw features shape: {X.shape}")
    try:
        X = X.astype(np.float32)
    except Exception:
        logger.exception("Failed to convert features to float32")
        raise
    before_shape = X.shape
    X = X.dropna()
    logger.info(f"Dropped feature NaNs: {before_shape} -> {X.shape}")

    # 4) Load and align targets
    y_df = load_targets(sep)
    if args.target_col not in y_df.columns:
        logger.error(f"Unknown classification target: {args.target_col}")
        sys.exit(1)
    if 'return_5d' not in y_df.columns:
        logger.error("Return target 'return_5d' not found in targets")
        sys.exit(1)

    y_class = y_df[args.target_col].reindex(X.index)
    y_reg   = y_df['return_5d'].reindex(X.index)
    mask = y_class.notna() & y_reg.notna()
    X_clean = X.loc[mask]
    y_class_clean = y_class.loc[mask]
    y_reg_clean   = y_reg.loc[mask]
    logger.info(
        f"After dropping target NaNs: Features={X_clean.shape}, "
        f"Classification={y_class_clean.shape}, Regression={y_reg_clean.shape}"
    )

    # 5) Train classifier
    logger.info("Training classifier...")
    clf = train_baseline_classification(
        X_clean, y_class_clean,
        backend=args.backend,
        device=args.device,
        num_classes=int(pd.unique(y_class_clean).size)
    )
    joblib.dump(clf, args.clf_out)
    logger.info(f"Classifier saved to {args.clf_out}")

    # 6) Train regressor
    logger.info("Training regressor...")
    reg = train_baseline_regression(
        X_clean, y_reg_clean,
        backend=args.backend,
        device=args.device
    )
    joblib.dump(reg, args.reg_out)
    logger.info(f"Regressor saved to {args.reg_out}")

    logger.info("Training and save complete.")


if __name__ == "__main__":
    main()
