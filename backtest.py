#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_predictions(path):
    df = pd.read_csv(path, parse_dates=["date"])
    if not {"ticker","date","p_up"}.issubset(df.columns):
        missing = {"ticker","date","p_up"} - set(df.columns)
        raise ValueError(f"Predictions missing columns: {missing}")
    return df

def load_actuals(path, horizon=5):
    # read full history up to your as-of date
    df = pd.read_parquet(path, columns=["ticker","date","close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"])
    # compute forward return
    df["next_close"]    = df.groupby("ticker")["close"].shift(-horizon)
    df["actual_return"] = (df["next_close"] - df["close"]) / df["close"]
    df["actual_direction"] = (df["actual_return"] > 0).astype(int)
    # drop the last `horizon` days per ticker (no forward close)
    df = df.dropna(subset=["actual_return"])
    return df[["ticker","date","actual_return","actual_direction"]]

def backtest_for_threshold(preds, actuals, thresh):
    # only keep the dates in common
    merged = pd.merge(preds, actuals, on=["ticker","date"], how="inner")
    if merged.empty:
        return None

    # generate signals
    merged["signal"] = np.where(merged["p_up"] >= thresh, "up",
                        np.where((1-merged["p_up"]) >= thresh, "down",
                                 "no_signal"))

    # drop no‐signal rows
    df = merged.loc[merged["signal"] != "no_signal"].copy()
    if df.empty:
        return {"Threshold":thresh, "Trades":0, "Accuracy":np.nan, "CumReturn":0.0}

    # map to +1 / 0
    df["pred_dir"] = df["signal"].map({"up":1,"down":0})
    df["correct"]  = (df["pred_dir"] == df["actual_direction"]).astype(int)
    # long = +1, short = -1
    df["strat_ret"] = df["actual_return"] * (2*df["pred_dir"] - 1)

    return {
        "Threshold"        : thresh,
        "Trades"           : len(df),
        "Accuracy"         : df["correct"].mean(),
        "CumulativeReturn" : df["strat_ret"].sum(),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="CSV of predictions with columns [ticker,date,p_up,…]")
    p.add_argument("--actuals", required=True,
                   help="Parquet of full SEP history (so you can compute forward returns)")
    p.add_argument("--horizon", type=int, default=5,
                   help="Prediction horizon in trading days (must match how preds were built)")
    p.add_argument("--output", required=True, help="CSV path for threshold sweep results")
    args = p.parse_args()

    preds   = load_predictions(args.predictions)
    actuals = load_actuals(args.actuals, horizon=args.horizon)

    # sanity check
    n_pred_dates   = preds["date"].nunique()
    n_actual_dates = actuals["date"].nunique()
    common_dates   = sorted(set(preds["date"]).intersection(actuals["date"]))
    logger.info(f"Predicted dates: {n_pred_dates}, actual‐return dates: {n_actual_dates}, "
                f"common dates: {len(common_dates)} → {common_dates[:3]}…")

    rows = []
    for thresh in [0.5,0.6,0.7,0.8,0.9,0.95]:
        stats = backtest_for_threshold(preds, actuals, thresh)
        if stats is None:
            logger.warning(f"--> no overlap at threshold {thresh:.2f} (no merged rows)")
            stats = {"Threshold":thresh, "Trades":0, "Accuracy":np.nan, "CumulativeReturn":0.0}
        logger.info(f"Thresh={thresh:.2f} → Trades={stats['Trades']}, "
                    f"Acc={stats['Accuracy']:.1%}, P&L={stats['CumulativeReturn']:.1%}")
        rows.append(stats)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    logger.info(f"Threshold sweep results → {args.output}")

if __name__=="__main__":
    main()
