#!/usr/bin/env python3
# train_and_save_models.py
# —————————————————————
# Train direction classifier & return regressor,
# with optional precomputed features input, dropping dupes
# and incomplete columns when --features is used.

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

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train & save classifier + regressor (with optional precomputed features)"
    )
    p.add_argument(
        "--sep-master", required=True,
        help="Path to master SEP Parquet (e.g. SHARADAR_SEP_clean_universe_YYYY-MM-DD.parquet)"
    )
    p.add_argument(
        "--features", default=None,
        help="OPTIONAL path to precomputed features Parquet (must have ticker,date,index)"
    )
    p.add_argument(
        "--date", default=None,
        help="YYYY-MM-DD for naming outputs (inferred from --sep-master if omitted)"
    )
    p.add_argument(
        "--horizon", choices=["1d","5d","10d","30d"], default="5d",
        help="Prediction horizon"
    )
    p.add_argument(
        "--backend", choices=["dummy","xgb","torch"], default="xgb",
        help="Model backend"
    )
    p.add_argument(
        "--device", choices=["cpu","gpu"], default="gpu",
        help="Device for training (only for xgb/torch)"
    )
    return p.parse_args()


def infer_date_from_sep(path: str) -> str:
    fname = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    if not m:
        raise ValueError(f"Could not infer date from SEP filename: {path}")
    return m.group(1)


def main():
    args = parse_args()

    # — determine date_str for outputs —
    date_str = args.date or infer_date_from_sep(args.sep_master)
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error("Date must be YYYY-MM-DD, got %s", date_str)
        sys.exit(1)
    logger.info(f"Using date: {date_str}")

    clf_out = f"models/dir_{args.horizon}_clf_{date_str}.joblib"
    reg_out = f"models/return_{args.horizon}_reg_{date_str}.joblib"

    # — load & validate SEP —
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    n_rows, n_tickers = sep.shape[0], sep["ticker"].nunique()
    logger.info(f"Loaded SEP: {n_rows:,} rows, {n_tickers:,} tickers")

    # — features: build or load —
    if args.features:
        # 1) load precomputed
        logger.info(f"Loading precomputed features from {args.features}")
        X = pd.read_parquet(args.features)
        # drop any dupes (ticker/date)
        before_dupes = X.shape[0]
        X = X.drop_duplicates(subset=["ticker","date"])
        dropped_dupes = before_dupes - X.shape[0]
        if dropped_dupes:
            logger.info(f"Dropped {dropped_dupes:,} duplicate rows from features")
        # normalize & index
        X["date"] = pd.to_datetime(X["date"])
        X.set_index(["ticker","date"], inplace=True)
        # drop any columns with missing values
        mask_complete = X.notna().all(axis=0)
        incomplete = X.columns[~mask_complete].tolist()
        if incomplete:
            logger.info(f"Dropping {len(incomplete)} incomplete feature columns: {incomplete}")
            X = X.loc[:, mask_complete]
        logger.info(f"Precomputed features shape (cleaned): {X.shape}")
        X_final = X.astype(np.float32)

    else:
        # 2) build on the fly & dropna rows
        X = load_features(sep)
        logger.info(f"Raw features shape: {X.shape}")
        X = X.astype(np.float32)
        before = X.shape
        X = X.dropna()
        logger.info(f"Dropped NaNs: {before} -> {X.shape}")
        X_final = X

    # — load & align targets —
    y_df = load_targets(sep)
    cls_col = f"dir_{args.horizon}"
    ret_col = f"return_{args.horizon}"
    if cls_col not in y_df.columns or ret_col not in y_df.columns:
        logger.error(f"Missing target columns {cls_col} or {ret_col}")
        sys.exit(1)

    y_class = y_df[cls_col].reindex(X_final.index)
    y_reg   = y_df[ret_col].reindex(X_final.index)
    mask    = y_class.notna() & y_reg.notna()

    X_clean = X_final.loc[mask]
    y_class = y_class.loc[mask]
    y_reg   = y_reg.loc[mask]
    logger.info(
        f"Clean data: Features={X_clean.shape}, Class targets={y_class.shape}, Reg targets={y_reg.shape}"
    )

    # — train classifier —
    logger.info("Training classifier...")
    clf = train_baseline_classification(
        X_clean, y_class,
        backend=args.backend,
        device=args.device,
        num_classes=int(pd.unique(y_class).size)
    )
    joblib.dump(clf, clf_out)
    logger.info(f"Classifier saved to {clf_out}")

    # — train regressor —
    logger.info("Training regressor...")
    reg = train_baseline_regression(
        X_clean, y_reg,
        backend=args.backend,
        device=args.device
    )
    joblib.dump(reg, reg_out)
    logger.info(f"Regressor saved to {reg_out}")

    logger.info("🎉 Training complete.")

if __name__ == "__main__":
    main()
