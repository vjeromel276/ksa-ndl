#!/usr/bin/env python3
# walk_forward_backtest.py
"""
Walk-forward backtest for your direction‐classifier and return‐regressor.
Uses an expanding training window, tests on the next block, then rolls forward.
Outputs detailed fold metrics + summary.
"""

import argparse
import logging
import pandas as pd
import numpy as np
import cupy as cp

from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import DMatrix

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
                   help="Fully-filtered SEP Parquet")
    p.add_argument("--universe-csv", required=True,
                   help="Ticker universe CSV")
    p.add_argument("--horizon",      choices=["1d","5d","10d","30d"], default="5d",
                   help="Prediction horizon")
    p.add_argument("--train-window", type=int, default=252,
                   help="Training window size (trading days)")
    p.add_argument("--test-window",  type=int, default=21,
                   help="Test window size (trading days)")
    p.add_argument("--step-size",    type=int, default=21,
                   help="Rolling step size (trading days)")
    p.add_argument("--purge-days",   type=int, default=None,
                   help="Trading days before test_start to purge (default = horizon-1)")
    p.add_argument("--embargo-days", type=int, default=0,
                   help="Trading days after test_end to embargo (default=0)")
    p.add_argument("--backend",      choices=["dummy","xgb","torch"], default="xgb",
                   help="Model backend")
    p.add_argument("--device",       choices=["cpu","gpu"], default="cpu",
                   help="Train & predict on CPU or GPU")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Probability threshold for p_up → signal")
    p.add_argument("--out-detail",   default="wf_folds.csv",
                   help="CSV path for per-fold metrics")
    p.add_argument("--out-summary",  default="wf_summary.csv",
                   help="CSV path for aggregated summary")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load & validate SEP
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP: {sep.shape[0]:,} rows")

    # 2) Cherry-pick universe
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

    # 4) Drop rows with missing targets
    mask = y_dir.notna() & y_ret.notna()
    X, y_dir, y_ret = X[mask], y_dir[mask], y_ret[mask]
    logger.info(f"Prepared data: X={X.shape}, dir={y_dir.shape}, ret={y_ret.shape}")

    # 5) Time axis
    dates = sorted(X.index.get_level_values("date").unique())
    n_dates = len(dates)
    tw, vw, step = args.train_window, args.test_window, args.step_size

    # Build a map date → trading‐day index
    date_positions = {d: i for i, d in enumerate(dates)}

    fold_stats = []
    fold = 0

    # 6) Rolling windows
    for start in range(0, n_dates - tw - vw + 1, step):
        fold += 1
        train_block = dates[start : start + tw]
        test_block  = dates[start + tw : start + tw + vw]

        tr = X.index.get_level_values("date").isin(train_block)
        te = X.index.get_level_values("date").isin(test_block)

        X_tr, X_te = X[tr], X[te]
        y_tr_dir, y_te_dir = y_dir[tr], y_dir[te]
        y_tr_ret, y_te_ret = y_ret[tr], y_ret[te]

        logger.info(f"[Fold {fold}] Train={X_tr.shape[0]:,} rows; Test={X_te.shape[0]:,} rows")

        # — Trading‐day purge & embargo in one go ——
        # compute each training‐row’s position in the full dates[] array
        tr_dates = X_tr.index.get_level_values("date")
        tr_pos   = np.array([ date_positions[d] for d in tr_dates ])

        # how many trading days to drop before test start?
        horizon_days   = int(args.horizon.rstrip("d"))
        purge_days     = args.purge_days if args.purge_days is not None else (horizon_days - 1)
        test_start_pos = start + tw
        cutoff_pos     = test_start_pos - purge_days

        # purge: keep everything before cutoff_pos
        keep_mask = (tr_pos < cutoff_pos)

        # embargo: also drop rows that would land within embargo_days after test_end
        if args.embargo_days > 0:
            test_end_pos   = start + tw + vw - 1
            embargo_cutoff = test_end_pos + args.embargo_days
            # keep rows either before purge OR _after_ the embargo window
            keep_mask = (tr_pos < cutoff_pos) | (tr_pos > embargo_cutoff)

        # debug how many survived
        logger.debug(
            f"[Fold {fold}] dates in train: {len(tr_pos):,}; "
            f"purge <{cutoff_pos}; embargo >{embargo_cutoff if args.embargo_days>0 else 'n/a'} → "
            f"kept {keep_mask.sum():,}"
        )

        # apply once, on the ORIGINAL X_tr
        X_tr, y_tr_dir, y_tr_ret = (
            X_tr[keep_mask],
            y_tr_dir[keep_mask],
            y_tr_ret[keep_mask],
        )
        logger.debug(f" After purge+embargo: {X_tr.shape[0]:,} rows")

        if X_tr.shape[0] == 0:
            logger.warning(f"[Fold {fold}] no training data after purge/embargo → skipping")
            continue

        # 7) Train & predict
        clf = train_baseline_classification(
            X_tr, y_tr_dir,
            backend=args.backend, device=args.device,
            num_classes=int(pd.unique(y_tr_dir).size)
        )
        reg = train_baseline_regression(
            X_tr, y_tr_ret,
            backend=args.backend, device=args.device
        )
        # convert your sklearn wrappers into raw Boosters
        # bst_clf = clf.get_booster()
        # bst_reg = reg.get_booster()

        # # push them onto the GPU
        # for bst in (bst_clf, bst_reg):
        #     bst.set_param({
        #         "tree_method":   "hist",
        #         "device":     "cuda",
        #         "gpu_id":        0
        #     })

        # # now do your DMatrix→.predict exactly as you already have
        # logger.info("→ clf   type: %s, methods: %s",
        #     type(clf), sorted([m for m in dir(clf) if not m.startswith("_")]))
        # logger.info("→ reg   type: %s, methods: %s",
        #     type(reg), sorted([m for m in dir(reg) if not m.startswith("_")]))

        # ensure GPU predictor if desired
        if args.backend=="xgb" and args.device=="gpu":
            for b in (clf.get_booster(), reg.get_booster()):
                # logger.debug(f"Setting booster {b} to use GPU predictor")
                b.set_param({
                    "tree_method":   "hist",
                    "device":     "cuda"
                })

        # 8) Predict p_up
        if args.backend=="xgb":
            arr = cp.asarray(X_te.values) if args.device=="gpu" else X_te.values
            p_up = clf.get_booster().predict(DMatrix(arr, feature_names=X_te.columns.tolist()))
            y_pred_ret = reg.get_booster().predict(DMatrix(arr, feature_names=X_te.columns.tolist()))
        else:
            p_up = clf.predict_proba(X_te)[:,1]
            y_pred_ret = reg.predict(X_te)

        pred_dir = (p_up >= args.threshold).astype(int)
        strat_ret = y_te_ret.values * (2*pred_dir - 1)

        # 9) Fold metrics
        acc    = accuracy_score(y_te_dir, pred_dir)
        rmse   = np.sqrt(mean_squared_error(y_te_ret, y_pred_ret))
        cum_r  = strat_ret.sum()
        sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252/vw)
                  if strat_ret.std()>0 else np.nan)

        fold_stats.append({
            "fold":       fold,
            "train_start": train_block[0],
            "train_end":   train_block[-1],
            "test_start":  test_block[0],
            "test_end":    test_block[-1],
            "dir_acc":    acc,
            "ret_rmse":   rmse,
            "cum_return": cum_r,
            "sharpe":     sharpe
        })

        logger.info(f"[Fold {fold}] Acc={acc:.2%}, RMSE={rmse:.4f}, "
                    f"CumR={cum_r:.2%}, Sharpe≈{sharpe:.2f}")

    # 10) Save results
    df_folds = pd.DataFrame(fold_stats)
    df_folds.to_csv(args.out_detail, index=False)
    logger.info(f"Saved folds → {args.out_detail}")

    df_summary = pd.DataFrame([
        ("mean_dir_acc",    df_folds["dir_acc"].mean()),
        ("std_dir_acc",     df_folds["dir_acc"].std()),
        ("mean_ret_rmse",   df_folds["ret_rmse"].mean()),
        ("std_ret_rmse",    df_folds["ret_rmse"].std()),
        ("mean_cum_return", df_folds["cum_return"].mean()),
        ("std_cum_return",  df_folds["cum_return"].std()),
        ("mean_sharpe",     df_folds["sharpe"].mean()),
        ("std_sharpe",      df_folds["sharpe"].std()),
    ], columns=["metric","value"])
    df_summary.to_csv(args.out_summary, index=False)
    logger.info(f"Saved summary → {args.out_summary}")


if __name__ == "__main__":
    main()
