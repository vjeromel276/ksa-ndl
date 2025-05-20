#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd
import numpy as np
import cupy as cp
from xgboost import DMatrix
from models.data import load_features, _coerce_sep_dtypes
from core.schema import validate_full_sep


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict stocks using date-stamped models."
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="Date stamp for models and SEP (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--sep-path", type=str,
        default="sep_dataset/SHARADAR_SEP_common_{date}.parquet",
        help="Path template for SEP Parquet file."
    )
    parser.add_argument(
        "--clf-path", type=str,
        default="models/dir_5d_clf_{date}.joblib",
        help="Path template for classifier joblib file."
    )
    parser.add_argument(
        "--reg-path", type=str,
        default="models/return_5d_reg_{date}.joblib",
        help="Path template for regressor joblib file."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Probability threshold for generating signals."
    )
    args = parser.parse_args()

    # Parse date and format paths
    as_of = pd.to_datetime(args.date)
    sep_file = args.sep_path.format(date=args.date)
    clf_file = args.clf_path.format(date=args.date)
    reg_file = args.reg_path.format(date=args.date)

    # 1) Load & validate SEP (up to as_of-date)
    sep = pd.read_parquet(sep_file)
    sep = _coerce_sep_dtypes(sep)
    validate_full_sep(sep)

    # 2) Build features, cast to float32
    X = load_features(sep).astype(np.float32)

    # 3) Filter to as_of date
    try:
        X_date = X.loc[X.index.get_level_values("date") == as_of]
    except KeyError:
        # if no MultiIndex, assume simple index
        X_date = X.loc[X.index.get_level_values("date") == as_of]

    # 4) Create DMatrix for XGBoost
    dmat = DMatrix(X_date)

    # 5) Load classifier and predict probabilities
    clf = joblib.load(clf_file)
    probs = clf.get_booster().predict(dmat)

    # 6) Load regressor and predict returns
    reg = joblib.load(reg_file)
    rets = reg.predict(dmat)

    # 7) Assemble results
    results = pd.DataFrame({
        "ticker": X_date.index.get_level_values("ticker"),
        "P_up": probs,
        "pred_return": rets
    })

    # 8) Apply threshold for signals
    th = args.threshold
    results["signal"] = np.where(
        results["P_up"] >= th, "up",
        np.where(1 - results["P_up"] >= th, "down", "no_signal")
    )

    # 9) Display top signals
    top_up = results[results.signal == "up"].sort_values("P_up", ascending=False)
    top_down = results[results.signal == "down"].sort_values("P_up")
    print(f"High-confidence UP signals (as of {args.date}):\n", top_up.head(10))
    print(f"\nHigh-confidence DOWN signals (as of {args.date}):\n", top_down.head(10))


if __name__ == "__main__":
    main()
