#!/usr/bin/env python3
#=========================
# compute_features.py
#=========================
"""
Compute a massive feature set for each (ticker, date) by stitching together
all available factors, then join them back onto the SEP index so you never
gain or lose rows, and fill any missing values per ticker:

  • liquidity   (e.g. dollar-volume)
  • momentum    (multi-window returns)
  • volatility  (rolling-std)
  • technicals  (MA, MACD, RSI, Bollinger)
  • seasonality (day-of-week, month-turn, etc.)
  • value       (dividend yield + placeholders for PE/PB/…)
  • quality     (ROE, ROA, margins, accruals)

Finally, forward- and backward-fill every feature per ticker so your final
matrix has the same row count as the SEP input, with no holes except the
PE/PB/PS/EV_EBITDA placeholders (which remain NA).
"""
import sys
import argparse
import logging

import pandas as pd

from feature_engineering.factors.liquidity   import build as build_liquidity
from feature_engineering.factors.momentum    import build as build_momentum
from feature_engineering.factors.volatility  import build as build_volatility
from feature_engineering.factors.technicals  import build as build_technicals
from feature_engineering.factors.seasonality import build as build_seasonality
from feature_engineering.factors.value       import build as build_value
from feature_engineering.factors.quality     import build as build_quality

# ——— Logging setup ——————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def parse_windows(window_str: str):
    return [int(w.strip()) for w in window_str.split(",") if w.strip()]

def load_table(path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet table, normalize column names to lowercase,
    coerce 'date' (or 'calendardate') into datetime64[ns].
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "calendardate" in df.columns:
        df["date"] = pd.to_datetime(df["calendardate"])
        df.drop(columns=["calendardate"], inplace=True)
    else:
        raise ValueError(f"No 'date' or 'calendardate' in {path}")

    return df

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compute unified feature matrix—aligned to SEP—with full coverage."
    )
    p.add_argument("--sep",          required=True,
                   help="Filtered SEP Parquet (OHLCV + closeadj)")
    p.add_argument("--metrics",      required=True,
                   help="SHARADAR_METRICS CSV or Parquet")
    p.add_argument("--fundamentals", required=True,
                   help="SHARADAR_SF1 fundamentals CSV or Parquet")
    p.add_argument("--out",     default="features.parquet",
                   help="Output Parquet for the feature matrix")
    p.add_argument("--min-date", help="Earliest SEP date to include")
    p.add_argument("--max-date", help="Latest SEP date to include")
    p.add_argument("--liquidity-windows",  type=parse_windows, default="21,63",
                   help="Liquidity windows, comma-separated")
    p.add_argument("--momentum-windows",   type=parse_windows, default="21,63,126,252",
                   help="Momentum windows, comma-separated")
    p.add_argument("--volatility-windows", type=parse_windows, default="20,50,100",
                   help="Volatility windows, comma-separated")
    opts = p.parse_args(argv or sys.argv[1:])

    # 1) Load SEP snapshot & capture its index
    logging.info("Loading SEP from %s", opts.sep)
    sep = pd.read_parquet(
        opts.sep,
        columns=["ticker","date","open","high","low","close","volume","closeadj"]
    )
    sep["date"] = pd.to_datetime(sep["date"])
    sep_idx = sep.set_index(["ticker","date"]).index
    logging.info("SEP rows: %d", len(sep_idx))

    # 2) Load METRICS & SF1 fundamentals
    logging.info("Loading METRICS from %s", opts.metrics)
    metrics = load_table(opts.metrics)
    logging.info("METRICS rows: %d", len(metrics))

    logging.info("Loading FUNDAMENTALS from %s", opts.fundamentals)
    fundamentals = load_table(opts.fundamentals)
    logging.info("FUNDAMENTALS rows: %d", len(fundamentals))

    # 3) Optional date-filter on SEP (recompute sep_idx)
    if opts.min_date:
        sep = sep[sep.date >= opts.min_date]
        sep_idx = sep.set_index(["ticker","date"]).index
        logging.info("Filtered SEP ≥ %s → %d rows", opts.min_date, len(sep_idx))
    if opts.max_date:
        sep = sep[sep.date <= opts.max_date]
        sep_idx = sep.set_index(["ticker","date"]).index
        logging.info("Filtered SEP ≤ %s → %d rows", opts.max_date, len(sep_idx))

    # 4) Build feature blocks (each returns a DataFrame indexed on ['ticker','date'])
    logging.info("Building liquidity features: %s", opts.liquidity_windows)
    liq = build_liquidity(sep, windows=opts.liquidity_windows).sort_index()

    logging.info("Building momentum features: %s", opts.momentum_windows)
    mom = build_momentum(sep, windows=opts.momentum_windows).sort_index()

    logging.info("Building volatility features: %s", opts.volatility_windows)
    vol = build_volatility(sep, windows=opts.volatility_windows).sort_index()

    logging.info("Building technical indicators")
    tech = build_technicals(sep).sort_index()

    logging.info("Building seasonality features")
    seas = build_seasonality(sep).sort_index()

    logging.info("Building value features")
    val = build_value(metrics).sort_index()

    logging.info("Building quality features")
    qual = build_quality(fundamentals).sort_index()

    # 5) Assemble via LEFT-JOIN onto sep_idx
    logging.info("Assembling feature matrix (left-join onto SEP index)")
    features = pd.DataFrame(index=sep_idx)
    for name, block in [
        ("liquidity", liq),
        ("momentum",  mom),
        ("volatility",vol),
        ("technicals",tech),
        ("seasonality",seas),
        ("value",     val),
        ("quality",   qual),
    ]:
        features = features.join(block, how="left")
        logging.info("Joined %-12s → %d columns", name, features.shape[1])

    # 6) Forward & backward fill per ticker to eliminate edge NaNs
    logging.info("Forward/back-filling missing features per ticker")
    features = (
        features
        .reset_index()
        .sort_values(["ticker","date"])
    )
    # groupby should be fast on multi-index; we reindex by ticker, date
    features.set_index(["ticker","date"], inplace=True)
    features = features.groupby(level=0).ffill().groupby(level=0).bfill()
    features.reset_index(inplace=True)

    # 7) Write to Parquet
    logging.info("Writing final features to %s", opts.out)
    features.to_parquet(opts.out, index=False)
    logging.info("Done! Final matrix: %d rows × %d columns",
                 len(features), features.shape[1] - 2)


if __name__ == "__main__":
    main()
