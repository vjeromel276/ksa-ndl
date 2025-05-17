# predict_console.py
# Console application to generate 5-day direction and return predictions using trained models.

#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd

from core.schema import validate_full_sep, _coerce_sep_dtypes
from models.data import load_features


def main():
    parser = argparse.ArgumentParser(
        description="Predict 5-day direction and return for a given ticker and date."
    )
    parser.add_argument(
        "--sep-master", type=str, default="sep_dataset/SHARADAR_SEP.parquet",
        help="Path to master SEP Parquet file"
    )
    parser.add_argument(
        "--classifier", type=str, required=True,
        help="Path to saved classifier model (e.g., .joblib or .pkl)"
    )
    parser.add_argument(
        "--regressor", type=str, required=True,
        help="Path to saved regression model"
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Probability threshold for classifying direction (e.g., 0.95)"
    )
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="Ticker symbol to predict for"
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="As-of date for prediction (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # 1) Load and validate SEP data
    sep = pd.read_parquet(args.sep_master)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)

    # 2) Build features for all (will select one)
    X = load_features(sep)

    # 3) Select the single row for (ticker, date)
    idx = (args.ticker, pd.to_datetime(args.date))
    if idx not in X.index:
        raise KeyError(f"Features for {idx} not found in SEP data.")
    x_input = X.loc[[idx]]  # DataFrame of shape (1, n_features)

    # 4) Load models
    clf = joblib.load(args.classifier)
    reg = joblib.load(args.regressor)

    # 5) Classification: predict probability and threshold
    proba = clf.predict_proba(x_input)[0]
    p_up = float(proba[1])
    direction = (
        "up" if p_up >= args.threshold else
        ("down" if proba[0] >= args.threshold else "no_signal")
    )

    # 6) Regression: predict 5-day return
    return_pred = float(reg.predict(x_input)[0])

    # 7) Output results
    print(f"Ticker: {args.ticker}, Date: {args.date}")
    print(f"5d_dir: {direction}  (P(up)={p_up:.3f}, threshold={args.threshold})")
    print(f"5d_return: {return_pred:.6f}")


if __name__ == "__main__":
    main()
