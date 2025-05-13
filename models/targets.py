#!/usr/bin/env python3
import pandas as pd
from typing import Optional, Sequence

def make_targets(
    df: pd.DataFrame,
    horizons: Optional[Sequence[int]] = None,
    price_col: str = "closeadj"
) -> pd.DataFrame:
    """
    Given a DataFrame with columns ['ticker','date', price_col] or indexed by ('ticker','date'),
    compute forward returns and direction flags at exactly h trading days ahead.

    Returns a DataFrame indexed by ['ticker','date'] with:
      - return_{h}d = (price_{t+h} / price_t) - 1
      - dir_{h}d    = 1 if return_{h}d > 0 else 0
    """
    df2 = df.copy()

    # 1) If it's MultiIndexed by ticker+date, pull them back as columns
    if {"ticker", "date"}.issubset(df2.index.names):
        df2 = df2.reset_index()

    # 2) Default to a 5-day horizon if none provided
    if horizons is None:
        horizons = [5]

    # 3) Normalize date and sort
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 4) Since the dataset has full trading-day coverage for each ticker,
    #    a plain shift(-h) yields the price h rows ahead.
    for h in horizons:
        ret_col = f"return_{h}d"
        dir_col = f"dir_{h}d"

        df2[ret_col] = (
            df2.groupby("ticker")[price_col]
               .shift(-h)
               .div(df2[price_col])
               .sub(1)
        )
        df2[dir_col] = (df2[ret_col] > 0).astype(int)

    # # 5) Drop any rows where *all* requested returns are NaN
    # ret_cols = [f"return_{h}d" for h in horizons]
    # df2 = df2.dropna(subset=ret_cols, how="all")

    # 6) Re-index by ticker+date for easy lookup
    return df2.set_index(["ticker", "date"], drop=False)
