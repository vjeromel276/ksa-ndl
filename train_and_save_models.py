#!/usr/bin/env python3
# train_and_save_models.py
# Console script to train direction classifier and return regressor,
# now with GPU‐accelerated XGBoost when --device gpu is specified.

import argparse
import logging
import sys
import os
import re
from datetime import datetime

import joblib
import pandas as pd
import numpy as np

from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.baseline import train_baseline_classification, train_baseline_regression

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save direction classifier and return regressor."
    )
    parser.add_argument(
        "--sep-master", required=True,
        help="Path to master SEP Parquet (e.g., SHARADAR_SEP_fully_filtered_YYYY-MM-DD.parquet)"
    )
    parser.add_argument(
        "--date", default=None,
        help="Date for naming outputs (YYYY-MM-DD). Inferred from --sep-master if omitted."
    )
    parser.add_argument(
        "--horizon", choices=["1d","5d","10d","30d"], default="5d",
        help="Prediction horizon"
    )
    parser.add_argument(
        "--backend", choices=["dummy","xgb","torch"], default="xgb",
        help="Classifier backend"
    )
    parser.add_argument(
        "--device", choices=["cpu","gpu"], default="gpu",
        help="Device for training (only matters for xgb/torch backends)"
    )
    return parser.parse_args()

def infer_date_from_sep(path: str) -> str:
    """
    Extracts YYYY-MM-DD from a SEP filename.
    """
    fname = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    if not m:
        raise ValueError(f"Could not infer date from SEP filename: {path}")
    return m.group(1)

def main():
    args = parse_args()

    # Determine date string
    date_str = args.date or infer_date_from_sep(args.sep_master)
    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error("Date must be in YYYY-MM-DD format: %s", date_str)
        sys.exit(1)
    logger.info(f"Using date: {date_str}")

    # Output filenames
    clf_out = f"models/dir_{args.horizon}_clf_{date_str}.joblib"
    reg_out = f"models/return_{args.horizon}_reg_{date_str}.joblib"

    # Load and validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP: {sep.shape[0]:,} rows, {sep['ticker'].nunique():,} tickers")

    # Feature engineering
    X = load_features(sep)
    logger.info(f"Raw features shape: {X.shape}")
    X = X.astype(np.float32)
    before = X.shape
    X = X.dropna()
    logger.info(f"Dropped NaNs: {before} -> {X.shape}")

    # Load and align targets
    y_df = load_targets(sep)
    target_col = f"dir_{args.horizon}"
    return_col = f"return_{args.horizon}"
    if target_col not in y_df.columns or return_col not in y_df.columns:
        logger.error(f"Targets {target_col} or {return_col} missing in y_df")
        sys.exit(1)

    y_class = y_df[target_col].reindex(X.index)
    y_reg   = y_df[return_col].reindex(X.index)
    mask    = y_class.notna() & y_reg.notna()

    X_clean = X.loc[mask]
    y_class = y_class.loc[mask]
    y_reg   = y_reg.loc[mask]
    logger.info(
        f"Clean data: Features={X_clean.shape}, Class targets={y_class.shape}, Reg targets={y_reg.shape}"
    )

    # Train classifier (will use GPU if --device gpu)
    logger.info("Training classifier...")
    clf = train_baseline_classification(
        X_clean, y_class,
        backend=args.backend,
        device=args.device,
        num_classes=int(pd.unique(y_class).size)
    )
    joblib.dump(clf, clf_out)
    logger.info(f"Classifier saved to {clf_out}")

    # Train regressor (will use GPU if --device gpu)
    logger.info("Training regressor...")
    reg = train_baseline_regression(
        X_clean, y_reg,
        backend=args.backend,
        device=args.device
    )
    joblib.dump(reg, reg_out)
    logger.info(f"Regressor saved to {reg_out}")

    logger.info("🎉 Training and saving complete.")

if __name__ == "__main__":
    main()
