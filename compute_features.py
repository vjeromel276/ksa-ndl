#!/usr/bin/env python3
#=========================
# compute_features.py
#=========================
import os
import sys
import argparse

import pandas as pd

from feature_engineering.factors import build_all_factors

def main(args=None):
    p = argparse.ArgumentParser(
        description="Compute cross‐sectional features for each (ticker, date)"
    )
    p.add_argument("--sep",       required=True,
                   help="Path to SEP Parquet (OHLCV + CloseAdj)")
    p.add_argument("--metrics",   required=True,
                   help="Path to SHARADAR/METRICS Parquet")
    p.add_argument("--indicators",required=True,
                   help="Path to SHARADAR/INDICATORS Parquet")
    p.add_argument("--out",       default="features.parquet",
                   help="Output Parquet for features")
    p.add_argument("--min-date",  help="Earliest date to include (YYYY-MM-DD)")
    p.add_argument("--max-date",  help="Latest date to include (YYYY-MM-DD)")
    opts = p.parse_args(args or [])

    # 1) Load data
    sep = pd.read_parquet(opts.sep, columns=[
        "ticker","date","open","high","low","close","volume","closeadj"
    ])
    sep["date"] = pd.to_datetime(sep["date"])
    metrics = pd.read_parquet(opts.metrics)
    indicators = pd.read_parquet(opts.indicators)

    # 2) Filter date range if requested
    if opts.min_date:
        sep = sep[sep.date >= opts.min_date]
    if opts.max_date:
        sep = sep[sep.date <= opts.max_date]

    # 3) Compute features
    feats = build_all_factors(sep, metrics, indicators)

    # 4) Write out
    feats.to_parquet(opts.out, index=False)
    print(f"[INFO] Wrote {len(feats)} rows × {len(feats.columns)-2} features to {opts.out}")

if __name__ == "__main__":
    main(sys.argv[1:])
