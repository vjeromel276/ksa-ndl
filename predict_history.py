#!/usr/bin/env python3
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import cupy as cp
from xgboost import DMatrix

from core.schema import validate_full_sep
from models.data import load_features, _coerce_sep_dtypes

# ——— Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(
        description="Batch‐predict across full history for a given horizon"
    )
    p.add_argument("--sep-master", required=True,
                   help="Path to fully‐filtered SEP parquet (e.g. SHARADAR_SEP_fully_filtered_2025-05-23.parquet)")
    p.add_argument("--horizon",  type=int,   default=5,
                   help="Prediction horizon in trading days")
    p.add_argument("--clf-model", required=True,
                   help="Path to dir_<horizon>_clf_<date>.joblib")
    p.add_argument("--reg-model", required=True,
                   help="Path to return_<horizon>_reg_<date>.joblib")
    p.add_argument("--threshold", type=float, default=None,
                   help="If set, will also emit a 'signal' column at this probability cutoff")
    p.add_argument("--output", required=True,
                   help="Where to write CSV of all predictions")
    args = p.parse_args()

    # 1) Load SEP and validate
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)
    logger.info(f"Loaded SEP with {sep['ticker'].nunique():,} tickers, {len(sep):,} rows")

    # 2) Build features for *every* date
    X = load_features(sep).astype(np.float32)
    # index is MultiIndex [ticker, date]
    all_dates = sorted(X.index.get_level_values("date").unique())
    usable_dates = all_dates[:-args.horizon]
    logger.info(f"Will predict on {len(usable_dates)} dates "
                f"(dropping last {args.horizon} days of history)")

    # 3) Load your models once
    clf = joblib.load(args.clf_model)
    reg = joblib.load(args.reg_model)
    logger.info(f"Loaded classifier {args.clf_model} and regressor {args.reg_model}")

    # 4) Loop over dates, predict and accumulate
    records = []
    for dt in usable_dates:
        X_dt = X.xs(dt, level="date")
        X_vals = X_dt.values  # NumPy array for sklearn fallbacks

        # classification probabilities
        if hasattr(clf, "get_booster"):
            gpu = cp.asarray(X_vals)
            dmat = DMatrix(gpu, feature_names=X_dt.columns.tolist())
            probs = clf.get_booster().predict(dmat)
        else:
            probs = clf.predict_proba(X_vals)[:, 1]

        # regression predictions
        if hasattr(reg, "get_booster"):
            # reuse dmat if you like, else reconstruct:
            gpu = cp.asarray(X_vals)
            dmat = DMatrix(gpu, feature_names=X_dt.columns.tolist())
            rets = reg.get_booster().predict(dmat)
        else:
            rets = reg.predict(X_vals)

        for i, ticker in enumerate(X_dt.index):
            rec = {
                "ticker":      ticker,
                "date":        dt,
                "p_up":        float(probs[i]),
                "pred_return": float(rets[i])
            }
            if args.threshold is not None:
                if rec["p_up"] >= args.threshold:
                    rec["signal"] = "up"
                elif (1 - rec["p_up"]) >= args.threshold:
                    rec["signal"] = "down"
                else:
                    rec["signal"] = "no_signal"
            records.append(rec)

    # 5) Write out
    out = pd.DataFrame.from_records(records)
    out.to_csv(args.output, index=False)
    logger.info(f"All‐history predictions saved → {args.output}")

if __name__=="__main__":
    main()
