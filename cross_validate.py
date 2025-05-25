#!/usr/bin/env python3
# cross_validate.py

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.baseline import train_baseline_classification, train_baseline_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(
        description="Time-series cross-validation of direction classifier & return regressor"
    )
    p.add_argument(
        "--sep-master", required=True,
        help="Path to fully-filtered SEP Parquet (e.g. sep_dataset/SHARADAR_SEP_fully_filtered_YYYY-MM-DD.parquet)"
    )
    p.add_argument(
        "--horizon", choices=["1d","5d","10d","30d"], default="5d",
        help="Prediction horizon"
    )
    p.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of time-series folds"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP: {sep.shape[0]:,} rows")

    # 2) build features & targets
    X = load_features(sep).astype(np.float32)
    y_df = load_targets(sep)
    dir_col = f"dir_{args.horizon}"
    ret_col = f"return_{args.horizon}"

    # align and drop missing
    mask = y_df[dir_col].notna() & y_df[ret_col].notna()
    X, y_dir, y_ret = X.loc[mask], y_df[dir_col].loc[mask], y_df[ret_col].loc[mask]
    logger.info(f"Prepared data: X={X.shape}, dir={y_dir.shape}, ret={y_ret.shape}")

    # 3) sort by date (necessary for TimeSeriesSplit)
    X = X.sort_index()
    y_dir = y_dir.sort_index()
    y_ret = y_ret.sort_index()

    # 4) time-series cross validator
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_dir, y_test_dir = y_dir.iloc[train_idx], y_dir.iloc[test_idx]
        y_train_ret, y_test_ret = y_ret.iloc[train_idx], y_ret.iloc[test_idx]

        logger.info(f"Fold {fold}: train={X_train.shape}, test={X_test.shape}")

        # 5a) classifier
        clf = train_baseline_classification(
            X_train, y_train_dir,
            backend="xgb", device="cpu",
            num_classes=int(pd.unique(y_train_dir).size)
        )
        y_pred_clf = clf.predict(X_test)
        acc_clf = accuracy_score(y_test_dir, y_pred_clf)

        # 5b) regressor
        reg = train_baseline_regression(
            X_train, y_train_ret,
            backend="xgb", device="cpu"
        )
        y_pred_ret = reg.predict(X_test)
        # directional accuracy from regressor
        dir_from_reg = (y_pred_ret > 0).astype(int)
        acc_reg_dir = accuracy_score(y_test_dir, dir_from_reg)
        # simulate a simple long(+1)/short(–1) strategy
        strat_ret = y_test_ret * (2*dir_from_reg - 1)
        cum_ret = strat_ret.sum()

        logger.info(
            f"Fold {fold} → clf_acc={acc_clf:.2%}, reg_dir_acc={acc_reg_dir:.2%}, P&L={cum_ret:.2%}"
        )

        results.append({
            "fold": fold,
            "clf_accuracy": acc_clf,
            "reg_dir_accuracy": acc_reg_dir,
            "cumulative_return": cum_ret
        })

    # 6) report
    df = pd.DataFrame(results)
    print("\n=== Cross-Validation Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
