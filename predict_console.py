#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd
import numpy as np
import logging

from core.schema import validate_full_sep
from models.data import load_features, _coerce_sep_dtypes
import cupy as cp
from xgboost import DMatrix

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(
        description="Predict 5-day direction and return for a given ticker and date on GPU using date-stamped models."
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="As-of date for prediction (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--history-csv", type=str, default=None,
        help="Optional path to CSV file with 252-day history for a single ticker"
    )
    parser.add_argument(
        "--classifier", type=str,
        help="Path to saved classifier model (joblib). Overrides date-stamped default",
    )
    parser.add_argument(
        "--regressor", type=str,
        help="Path to saved regression model (joblib). Overrides date-stamped default",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Probability threshold for classifying direction (e.g., 0.95)"
    )
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="Ticker symbol to predict for"
    )
    args = parser.parse_args()

    # Build file paths based on date
    sep_file = f"sep_dataset/SHARADAR_SEP_common_{args.date}.parquet"
    clf_file = args.classifier or f"models/dir_5d_clf_{args.date}.joblib"
    reg_file = args.regressor or f"models/return_5d_reg_{args.date}.joblib"

    # 1) Load data (full SEP or history CSV)
    if args.history_csv:
        sep = pd.read_csv(args.history_csv, parse_dates=['date'])
        sep = _coerce_sep_dtypes(sep)
        validate_full_sep(sep)
    else:
        sep = pd.read_parquet(sep_file)
        sep = _coerce_sep_dtypes(sep)
        validate_full_sep(sep)

    # 2) Build features and cast to float32
    X = load_features(sep).astype(np.float32)

    # 3) Select the single row for prediction
    idx = (args.ticker, pd.to_datetime(args.date))
    if idx not in X.index:
        raise KeyError(f"Features for {idx} not found in {sep_file}.")
    x_row = X.loc[[idx]]

    # 4) Load models
    clf = joblib.load(clf_file)
    reg = joblib.load(reg_file)

    feature_names = x_row.columns.tolist()

    # 5) GPU-based classification prediction
    data_gpu = cp.asarray(x_row.values)
    dmat = DMatrix(data_gpu, feature_names=feature_names)
    logger.debug("Created DMatrix with feature names: %s", feature_names)
    p_up = float(clf.get_booster().predict(dmat)[0])
    p_down = 1.0 - p_up
    direction = (
        "up" if p_up >= args.threshold else
        "down" if p_down >= args.threshold else
        "no_signal"
    )

    # 6) GPU-based regression prediction
    dmat_reg = DMatrix(cp.asarray(x_row.values), feature_names=feature_names)
    logger.debug("Created Regression DMatrix with feature names: %s", feature_names)
    return_pred = float(reg.get_booster().predict(dmat_reg)[0])

    # 7) Output results
    print(f"Date: {args.date}, Ticker: {args.ticker}")
    print(f"5d_dir: {direction}  (P(up)={p_up:.3f}, threshold={args.threshold})")
    print(f"5d_return: {return_pred:.6f}")

if __name__ == "__main__":
    main()
