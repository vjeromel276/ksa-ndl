#!/usr/bin/env python3
import argparse
import joblib
import logging
import pandas as pd
import numpy as np
import cupy as cp
from xgboost import DMatrix
from core.schema import validate_full_sep
from models.data import load_features, _coerce_sep_dtypes

# Configure logger
def configure_logging():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = configure_logging()

def main():
    parser = argparse.ArgumentParser(
        description="Batch-universe and single-ticker prediction for 5-day horizon"
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="As-of date for prediction (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Optional: specific ticker to predict (e.g. AAPL); if omitted, runs on full universe"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Probability threshold for classifying direction"
    )
    args = parser.parse_args()

    date = args.date
    # 1) Load & validate SEP for given date
    sep_path = f"sep_dataset/SHARADAR_SEP_common_{date}.parquet"
    logger.debug(f"Loading SEP data from {sep_path}")
    sep = pd.read_parquet(sep_path)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)

    # 2) Build features
    logger.debug("Generating feature matrix")
    X = load_features(sep).astype(np.float32)

    # 3) Filter to as-of-date
    as_of = pd.to_datetime(date)
    try:
        X_date = X.xs(as_of, level="date")
    except KeyError:
        raise KeyError(f"No feature data for date {date}")

    # 4) Optional single-ticker filter
    if args.ticker:
        if args.ticker not in X_date.index:
            raise KeyError(f"Ticker {args.ticker} not found for date {date}")
        X_date = X_date.loc[[args.ticker]]

    # 5) Prepare DMatrix on GPU
    gpu_array = cp.asarray(X_date.values)
    dmat = DMatrix(gpu_array, feature_names=X_date.columns.tolist())

    # 6) Load dated models (YYYY-MM-DD format)
    clf_path = f"models/dir_5d_clf_{date}.joblib"
    reg_path = f"models/return_5d_reg_{date}.joblib"
    logger.debug(f"Loading classification model from {clf_path}")
    clf = joblib.load(clf_path)
    logger.debug(f"Loading regression model from {reg_path}")
    reg = joblib.load(reg_path)

    # 7) Run predictions
    logger.debug("Running predictions")
    probs = clf.get_booster().predict(dmat)
    rets = reg.get_booster().predict(dmat)

    # 8) Assemble results
    records = []
    for idx, ticker in enumerate(X_date.index):
        p_up = float(probs[idx])
        pred_ret = float(rets[idx])
        if p_up >= args.threshold:
            signal = "up"
        elif (1 - p_up) >= args.threshold:
            signal = "down"
        else:
            signal = "no_signal"
        records.append({
            "ticker": ticker,
            "date": date,
            "p_up": p_up,
            "pred_return": pred_ret,
            "signal": signal
        })

    df = pd.DataFrame(records)
    if args.ticker:
        row = df.iloc[0]
        print(f"Ticker: {row.ticker} | Date: {row.date}")
        print(f"P_up: {row.p_up:.4f} | Pred_return: {row.pred_return:.4f} | Signal: {row.signal}")
    else:
        # Save or print CSV
        print(df.to_csv(index=False))
        df.to_csv(f"predictions/predictions_{date}.csv", index=False)
        logger.debug(f"Predictions saved to predictions/predictions_{date}.csv")
    logger.debug("Prediction complete")

if __name__ == "__main__":
    main()
