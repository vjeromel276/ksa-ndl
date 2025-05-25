#!/usr/bin/env python3
# cross_validate_models.py
"""
Time-series cross-validation for your direction classifier and return regressor.
Splits on dates to avoid look-ahead bias, trains on an expanding window, and
reports out-of-sample accuracy and RMSE for each fold.
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import joblib
from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.cherry_picker import get_valid_tickers_for_horizon
from models.baseline import train_baseline_classification, train_baseline_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Time-series CV for direction classifier & return regressor"
    )
    p.add_argument("--sep-master", required=True,
                   help="Path to fully-filtered SEP Parquet (e.g. SHARADAR_SEP_fully_filtered_YYYY-MM-DD.parquet)")
    p.add_argument("--universe-csv", required=True,
                   help="Path to ticker_universe_clean_<date>.csv")
    p.add_argument("--horizon", choices=["1d","5d","10d","30d"], default="5d",
                   help="Prediction horizon")
    p.add_argument("--splits", type=int, default=5,
                   help="Number of time-series splits (default: 5)")
    p.add_argument("--backend", choices=["dummy","xgb","torch"], default="xgb",
                   help="Which backend to train with")
    p.add_argument("--device", choices=["cpu","gpu"], default="cpu",
                   help="Train on CPU or GPU")
    return p.parse_args()


def main():
    args = parse_args()
    # 1) Load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP: {sep.shape[0]:,} rows")

    # 2) Cherry-pick universe for this horizon
    tickers = get_valid_tickers_for_horizon(
        universe_csv=args.universe_csv,
        horizon=args.horizon
    )
    sep = sep[sep["ticker"].isin(tickers)]
    logger.info(f"After cherry-pick: {sep['ticker'].nunique():,} tickers, {sep.shape[0]:,} rows")

    # 3) Build features & targets
    X = load_features(sep).astype(np.float32)
    y_df = load_targets(sep)
    dir_col = f"dir_{args.horizon}"
    ret_col = f"return_{args.horizon}"
    y_dir = y_df[dir_col].reindex(X.index)
    y_ret = y_df[ret_col].reindex(X.index)

    mask = y_dir.notna() & y_ret.notna()
    X, y_dir, y_ret = X[mask], y_dir[mask], y_ret[mask]
    logger.info(f"Cleaned: {X.shape[0]:,} samples")

    # 4) Prepare time-series splits on dates
    dates = sorted(X.index.get_level_values("date").unique())
    tscv = TimeSeriesSplit(n_splits=args.splits)

    fold_stats = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(dates), 1):
        train_dates = [dates[i] for i in train_idx]
        test_dates  = [dates[i] for i in test_idx]

        train_mask = X.index.get_level_values("date").isin(train_dates)
        test_mask  = X.index.get_level_values("date").isin(test_dates)

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr_dir, y_te_dir = y_dir[train_mask], y_dir[test_mask]
        y_tr_ret, y_te_ret = y_ret[train_mask], y_ret[test_mask]

        logger.info(f"[Fold {fold}] Train on {len(train_dates)} dates → {X_tr.shape[0]:,} rows; "
                    f"Test on {len(test_dates)} dates → {X_te.shape[0]:,} rows")

        # 5a) Train classifier
        clf = train_baseline_classification(
            X_tr, y_tr_dir,
            backend=args.backend,
            device=args.device,
            num_classes=int(pd.unique(y_tr_dir).size)
        )
        # predict direction
        #– if your baseline returns an sklearn-style .predict_proba:
        try:
            y_pred_prob = clf.predict_proba(X_te)[:, 1]
            y_pred_dir  = (y_pred_prob >= 0.5).astype(int)
        except AttributeError:
            # fallback to XGBoost raw booster
            from xgboost import DMatrix
            dmat_clf = DMatrix(X_te.values, feature_names=X_te.columns.tolist())
            y_pred_prob = clf.get_booster().predict(dmat_clf)
            y_pred_dir  = (y_pred_prob >= 0.5).astype(int)

        acc = accuracy_score(y_te_dir, y_pred_dir)

        # 5b) Train regressor
        reg = train_baseline_regression(
            X_tr, y_tr_ret,
            backend=args.backend,
            device=args.device
        )
        # predict returns
        try:
            y_pred_ret = reg.predict(X_te)
        except AttributeError:
            dmat_reg = DMatrix(X_te.values, feature_names=X_te.columns.tolist())
            y_pred_ret = reg.get_booster().predict(dmat_reg)
        mse  = mean_squared_error(y_te_ret, y_pred_ret)
        rmse = np.sqrt(mse)

        fold_stats.append({
            "fold": fold,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "dir_acc": acc,
            "ret_rmse": rmse
        })
        logger.info(f"[Fold {fold}] Dir Acc = {acc:.3%}, Ret RMSE = {rmse:.4f}")

    # 6) Summarize
    df_stats = pd.DataFrame(fold_stats)
    summary = {
        "mean_dir_acc": df_stats["dir_acc"].mean(),
        "std_dir_acc":  df_stats["dir_acc"].std(),
        "mean_ret_rmse":df_stats["ret_rmse"].mean(),
        "std_ret_rmse": df_stats["ret_rmse"].std(),
    }
    logger.info("=== Cross-Validation Summary ===")
    for k, v in summary.items():
        if "acc" in k:
            logger.info(f"{k}: {v:.3%}")
        else:
            logger.info(f"{k}: {v:.4f}")

    # 7) Save fold-level metrics
    out_csv = f"cv_results_{args.horizon}_{datetime.today().date()}.csv"
    df_stats.to_csv(out_csv, index=False)
    logger.info(f"Detailed fold metrics → {out_csv}")


if __name__ == "__main__":
    main()
