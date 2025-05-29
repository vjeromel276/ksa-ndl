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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Predict batch universe for a given date.")
    parser.add_argument("--date", required=True, help="As-of date for prediction (YYYY-MM-DD)")
    parser.add_argument("--sep-master", required=False, help="SEP master file path")
    parser.add_argument("--threshold", type=float, default=0.95, help="Probability threshold for signal")
    args = parser.parse_args()

    date = args.date
    sep_path = args.sep_master or f"sep_dataset/SHARADAR_SEP_clean_{date}.parquet"

    logger.info(f"Loading SEP data from {sep_path}")
    sep = pd.read_parquet(sep_path)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)

    # Feature Engineering
    logger.info("Generating feature matrix")
    X = load_features(sep).astype(np.float32)

    # Filter by date
    as_of = pd.to_datetime(date)
    X_date = X.xs(as_of, level="date", drop_level=False)
    logger.info(f"Features for prediction date {date}: {X_date.shape[0]} tickers")

    if X_date.empty:
        logger.error(f"No data for prediction date {date}")
        return

    # Prepare GPU data
    gpu_array = cp.asarray(X_date.values)
    dmat = DMatrix(gpu_array, feature_names=X_date.columns.tolist())

    # Load models
    clf_path = f"models/dir_5d_clf_{date}.joblib"
    reg_path = f"models/return_5d_reg_{date}.joblib"
    logger.info(f"Loading classifier from {clf_path}")
    clf = joblib.load(clf_path)
    logger.info(f"Loading regressor from {reg_path}")
    reg = joblib.load(reg_path)

    # Predict
    logger.info("Running batch predictions...")
    probs = clf.get_booster().predict(dmat)
    rets = reg.get_booster().predict(dmat)

    # Results DataFrame
    predictions = pd.DataFrame({
        'ticker': X_date.index.get_level_values('ticker'),
        'date': date,
        'p_up': probs,
        'pred_return': rets,
        'signal': np.where(probs >= args.threshold, 'up', np.where(1 - probs >= args.threshold, 'down', 'no_signal'))
    })

    # Save results
    output_path = f"predictions/predictions_{date}.csv"
    predictions.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
