#!/usr/bin/env python3
#===============================
# compute_features_per_ticker.py
#===============================
"""
Per-ticker feature generation to ensure 100% coverage:

  1) Loads SEP, METRICS, and SF1 FUNDAMENTALS.
  2) For each ticker, builds:
     • liquidity   (dvol windows)
     • momentum    (return windows)
     • volatility  (rolling‐std windows)
     • technicals  (MA, MACD, RSI, Bollinger)
     • seasonality (dow, tom, etc.)
     • value       (dividend yield + PE/PB placeholders)
     • quality     (roe, roa, debt/equity, accruals)
  3) Joins them onto the ticker’s own date index (guaranteed 6 882 rows).
  4) Forward/back-fills edges so only the placeholder ratios remain NA,
     then calls infer_objects() to ensure correct dtypes.
  5) Stitches all tickers back together.
"""
import argparse
import logging
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from feature_engineering.factors.liquidity   import build as build_liquidity
from feature_engineering.factors.momentum    import build as build_momentum
from feature_engineering.factors.volatility  import build as build_volatility
from feature_engineering.factors.technicals  import build as build_technicals
from feature_engineering.factors.seasonality import build as build_seasonality
from feature_engineering.factors.value       import build as build_value
from feature_engineering.factors.quality     import build as build_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def parse_windows(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def load_table(path: str) -> pd.DataFrame:
    """Load CSV/Parquet, lowercase cols, coerce date/calendardate→date."""
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
        raise ValueError(f"No date column in {path}")
    return df

def compute_ticker_features(
    ticker: str,
    sep_grp: pd.DataFrame,
    metrics: pd.DataFrame,
    fundamentals: pd.DataFrame,
    liq_windows, mom_windows, vol_windows
) -> pd.DataFrame:
    """
    Build all feature blocks for a single ticker, join to its own date index,
    forward/backfill any edge NaNs, then infer_objects() for proper dtypes.
    """
    sep_grp = sep_grp.sort_values("date")
    dates = sep_grp["date"]

    # build all blocks
    liq  = build_liquidity(sep_grp, windows=liq_windows)
    mom  = build_momentum (sep_grp, windows=mom_windows)
    vol  = build_volatility(sep_grp, windows=vol_windows)
    tech = build_technicals(sep_grp)
    seas = build_seasonality (sep_grp)

    met_tkr  = metrics   [metrics   .ticker == ticker]
    val      = build_value(met_tkr)
    fund_tkr = fundamentals[fundamentals.ticker == ticker]
    qual     = build_quality(fund_tkr)

    # drop ticker level so index becomes dates
    def drop_ticker_level(df_block):
        idx = df_block.index.get_level_values("date")
        df_block.index = idx
        return df_block

    for blk in (liq, mom, vol, tech, seas, val, qual):
        blk = drop_ticker_level(blk)

    # join onto date index
    df = pd.DataFrame(index=dates)
    for blk in (liq, mom, vol, tech, seas, val, qual):
        df = df.join(blk, how="left", sort=False)

    # forward/back-fill & infer_objects to silence downcast warning
    df = df.infer_objects(copy=False)
    # df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.ffill().bfill()
    # df.infer_objects(copy=False)

    # reattach ticker & date
    df = df.reset_index().rename(columns={"index": "date"})
    df.insert(0, "ticker", ticker)
    return df

def main():
    p = argparse.ArgumentParser(
        description="Per-ticker feature generation for 100% coverage"
    )
    p.add_argument("--sep",          required=True, help="Filtered SEP Parquet")
    p.add_argument("--metrics",      required=True, help="SHARADAR_METRICS CSV/Parquet")
    p.add_argument("--fundamentals", required=True, help="SHARADAR_SF1 CSV/Parquet")
    p.add_argument("--out", default="features_per_ticker.parquet",
                   help="Output Parquet")
    p.add_argument("--liquidity-windows",  type=parse_windows, default="21,63")
    p.add_argument("--momentum-windows",   type=parse_windows, default="21,63,126,252")
    p.add_argument("--volatility-windows", type=parse_windows, default="20,50,100")
    opts = p.parse_args()

    # load inputs
    sep = pd.read_parquet(opts.sep).infer_objects(copy=False)
    metrics = load_table(opts.metrics)
    fundamentals = load_table(opts.fundamentals)
    logging.info(
        "Loaded SEP (%d rows), METRICS (%d), FUNDAMENTALS (%d)",
        len(sep), len(metrics), len(fundamentals)
    )

    # process each ticker
    parts = []
    for i, (ticker, grp) in enumerate(sep.groupby("ticker"), start=1):
        parts.append(
            compute_ticker_features(
                ticker, grp, metrics, fundamentals,
                opts.liquidity_windows,
                opts.momentum_windows,
                opts.volatility_windows
            )
        )
        if i % 100 == 0:
            logging.info("Processed %d tickers…", i)

    # concatenate & write
    features = pd.concat(parts, ignore_index=True)
    logging.info(
        "Final feature matrix: %d rows × %d cols",
        len(features), features.shape[1]
    )
    features.to_parquet(opts.out, index=False)
    logging.info("Done → %s", opts.out)

if __name__ == "__main__":
    main()
