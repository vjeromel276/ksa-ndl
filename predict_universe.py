#!/usr/bin/env python3
"""
predict_universe.py

Score a universe of tickers at a given date/horizon using your
precomputed features and trained XGBoost models.
"""
import argparse
import logging

import joblib
import pandas as pd
import xgboost as xgb

from models.data import _coerce_sep_dtypes

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict cross-sectional signals on a SEP universe snapshot"
    )
    p.add_argument("--sep-master",      required=True, help="Filtered SEP Parquet")
    p.add_argument("--ticker-universe", required=True, help="CSV of tickers to keep")
    p.add_argument("--features",        required=True, help="Precomputed features Parquet")
    p.add_argument("--date",            required=True, help="As-of date (YYYY-MM-DD)")
    p.add_argument("--horizon",         choices=["1d","5d","10d","30d"], default="5d")
    p.add_argument("--threshold",       type=float, default=0.6, help="Prob threshold")
    p.add_argument("--output",          default=None, help="Output Parquet path")
    return p.parse_args()


def main():
    opts = parse_args()

    # parse as-of date
    asof = pd.to_datetime(opts.date)

    # 1) load SEP and restrict to universe
    logger.info("Loading SEP from %s", opts.sep_master)
    sep = pd.read_parquet(opts.sep_master)
    sep = _coerce_sep_dtypes(sep)

    keep = pd.read_csv(opts.ticker_universe, usecols=["ticker"])["ticker"]
    before = sep.shape[0]
    sep = sep[sep["ticker"].isin(keep)]
    logger.info("Filtered SEP: %d → %d rows", before, sep.shape[0])

    # 2) load features
    logger.info("Loading features from %s", opts.features)
    feats = pd.read_parquet(opts.features)
    feats = feats.drop_duplicates(subset=["ticker","date"])
    feats["date"] = pd.to_datetime(feats["date"])
    feats.set_index(["ticker","date"], inplace=True)

    # 3) slice to as-of date
    X_date = feats.xs(asof, level="date", drop_level=False)
    logger.info("Features for %s: %d tickers × %d cols",
                opts.date, *X_date.shape)

    # 4) load models
    clf_path = f"models/dir_{opts.horizon}_clf_{opts.date}.joblib"
    reg_path = f"models/return_{opts.horizon}_reg_{opts.date}.joblib"
    logger.info("Loading classifier %s", clf_path)
    clf = joblib.load(clf_path)
    logger.info("Loading regressor %s", reg_path)
    reg = joblib.load(reg_path)

    # 5) align feature names
    expected = clf.get_booster().feature_names
    missing  = set(expected) - set(X_date.columns)
    if missing:
        logger.error("Missing features: %s", sorted(missing))
        raise RuntimeError("Feature mismatch")
    X_date = X_date.loc[:, expected]

    # 6) classification & regression
    dmat  = xgb.DMatrix(X_date.values, feature_names=expected)
    probs = clf.get_booster().predict(dmat)

    # <-- THE FIX: do not feed a DMatrix into reg.predict() -->
    rets  = reg.predict(X_date.values)

    # 7) assemble results
    out = pd.DataFrame({
        "ticker":                   X_date.index.get_level_values("ticker"),
        "prob_long":                probs,
        f"pred_{opts.horizon}_return": rets
    })
    out["signal_long"] = (out["prob_long"] >= opts.threshold).astype(int)

    # 8) write
    out_fn = opts.output or f"predictions_universe_{opts.date}.parquet"
    logger.info("Writing %d rows to %s", len(out), out_fn)
    out.to_parquet(out_fn, index=False)

    logger.info("Done.")

if __name__ == "__main__":
    main()
