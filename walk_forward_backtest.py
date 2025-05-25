#!/usr/bin/env python3
# walk_forward_backtest.py
"""
Walk-forward backtest for your direction‐classifier and return‐regressor.
Uses an expanding 252-day training window, tests on the next 21 days,
then rolls forward by 21 days.  Outputs detailed fold metrics + summary.
"""

import argparse
import logging
from datetime import date
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import numpy as np
from core.schema import validate_full_sep
from models.data import load_features, load_targets, _coerce_sep_dtypes
from models.cherry_picker import get_valid_tickers_for_horizon
from models.baseline import train_baseline_classification, train_baseline_regression

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(
        description="Walk-forward backtest (252d train → 21d test → step=21d)"
    )
    p.add_argument("--sep-master",   required=True,
                   help="Fully-filtered SEP Parquet (e.g. SHARADAR_SEP_fully_filtered_YYYY-MM-DD.parquet)")
    p.add_argument("--universe-csv", required=True,
                   help="Ticker universe CSV (e.g. ticker_universe_clean_YYYY-MM-DD.csv)")
    p.add_argument("--horizon",      choices=["1d","5d","10d","30d"], default="5d",
                   help="Prediction horizon")
    p.add_argument("--train-window", type=int, default=252,
                   help="Training window size in trading days")
    p.add_argument("--test-window",  type=int, default=21,
                   help="Test window size in trading days")
    p.add_argument("--step-size",    type=int, default=21,
                   help="Rolling step size in trading days")
    p.add_argument("--backend",      choices=["dummy","xgb","torch"], default="xgb",
                   help="Which model backend to train with")
    p.add_argument("--device",       choices=["cpu","gpu"], default="cpu",
                   help="Train on CPU or GPU")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Probability threshold to convert p_up→signal")
    p.add_argument("--out-detail",   default="wf_folds.csv",
                   help="Output CSV path for per-fold metrics")
    p.add_argument("--out-summary",  default="wf_summary.csv",
                   help="Output CSV path for aggregated summary")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP: {sep.shape[0]:,} rows")

    # 2) Cherry-pick universe for this horizon
    tickers = get_valid_tickers_for_horizon(args.universe_csv, args.horizon)
    sep = sep[sep["ticker"].isin(tickers)]
    logger.info(f"After cherry-pick: {sep['ticker'].nunique():,} tickers, {sep.shape[0]:,} rows")

    # 3) Build features & targets
    X = load_features(sep).astype(np.float32)
    y_df = load_targets(sep)
    dir_col = f"dir_{args.horizon}"
    ret_col = f"return_{args.horizon}"
    y_dir = y_df[dir_col].reindex(X.index)
    y_ret = y_df[ret_col].reindex(X.index)

    # 4) Drop any rows with missing targets
    mask = y_dir.notna() & y_ret.notna()
    X, y_dir, y_ret = X[mask], y_dir[mask], y_ret[mask]
    logger.info(f"Prepared data: X={X.shape}, dir=(…){y_dir.shape}, ret=(…){y_ret.shape}")

    # 5) Determine unique sorted dates (our time axis)
    dates = sorted(X.index.get_level_values("date").unique())
    n_dates = len(dates)
    tw, vw, step = args.train_window, args.test_window, args.step_size

    fold_stats = []
    fold = 0

    # 6) Rolling windows
    for start in range(0, n_dates - tw - vw + 1, step):
        fold += 1
        train_dates = dates[start : start + tw]
        test_dates  = dates[start + tw : start + tw + vw]

        # Masks
        tr_mask = X.index.get_level_values("date").isin(train_dates)
        te_mask = X.index.get_level_values("date").isin(test_dates)

        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr_dir, y_te_dir = y_dir[tr_mask], y_dir[te_mask]
        y_tr_ret, y_te_ret = y_ret[tr_mask], y_ret[te_mask]

        logger.info(
            f"[Fold {fold}] Train on {len(train_dates)} dates → {X_tr.shape[0]:,} rows; "
            f"Test on {len(test_dates)} dates → {X_te.shape[0]:,} rows"
        )

        # 7a) Train classifier
        clf = train_baseline_classification(
            X_tr, y_tr_dir,
            backend=args.backend,
            device=args.device,
            num_classes=int(pd.unique(y_tr_dir).size)
        )
        # 7b) Train regressor
        reg = train_baseline_regression(
            X_tr, y_tr_ret,
            backend=args.backend,
            device=args.device
        )

        # 8) Make predictions on test
        try:
            # sklearn-style
            p_up = clf.predict_proba(X_te)[:, 1]
        except AttributeError:
            from xgboost import DMatrix
            dmat = DMatrix(X_te.values, feature_names=X_te.columns.tolist())
            p_up = clf.get_booster().predict(dmat)

        try:
            y_pred_ret = reg.predict(X_te)
        except AttributeError:
            from xgboost import DMatrix
            dmat = DMatrix(X_te.values, feature_names=X_te.columns.tolist())
            y_pred_ret = reg.get_booster().predict(dmat)

        # 9) Signals & P&L
        pred_dir = (p_up >= args.threshold).astype(int)
        # long(+1)/short(-1)
        strat_ret = y_te_ret.values * (2 * pred_dir - 1)

        # 10) Metrics
        acc   = accuracy_score(y_te_dir, pred_dir)
        mse  = mean_squared_error(y_te_ret, y_pred_ret)
        rmse = np.sqrt(mse)
        cum_r = strat_ret.sum()
        sharpe = (
            strat_ret.mean() / strat_ret.std() * np.sqrt(252 / vw)
            if strat_ret.std() > 0 else np.nan
        )

        fold_stats.append({
            "fold": fold,
            "train_start": train_dates[0],
            "train_end":   train_dates[-1],
            "test_start":  test_dates[0],
            "test_end":    test_dates[-1],
            "dir_acc":     acc,
            "ret_rmse":    rmse,
            "cum_return":  cum_r,
            "sharpe":      sharpe
        })

        logger.info(
            f"[Fold {fold}] Acc={acc:.2%}, RMSE={rmse:.4f}, "
            f"CumR={cum_r:.2%}, Sharpe≈{sharpe:.2f}"
        )

    # 11) Save detailed fold metrics
    df_folds = pd.DataFrame(fold_stats)
    df_folds.to_csv(args.out_detail, index=False)
    logger.info(f"Detailed fold metrics → {args.out_detail}")

    # 12) Summarize across folds
    summary = {
        "mean_dir_acc":    df_folds["dir_acc"].mean(),
        "std_dir_acc":     df_folds["dir_acc"].std(),
        "mean_ret_rmse":   df_folds["ret_rmse"].mean(),
        "std_ret_rmse":    df_folds["ret_rmse"].std(),
        "mean_cum_return": df_folds["cum_return"].mean(),
        "std_cum_return":  df_folds["cum_return"].std(),
        "mean_sharpe":     df_folds["sharpe"].mean(),
        "std_sharpe":      df_folds["sharpe"].std(),
    }
    df_summary = pd.DataFrame(summary.items(), columns=["metric","value"])
    df_summary.to_csv(args.out_summary, index=False)
    logger.info(f"Summary metrics → {args.out_summary}")

if __name__ == "__main__":
    main()
