# models/targets.py
#!/usr/bin/env python3
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import Optional, Sequence

def make_targets(
    df: pd.DataFrame,
    horizons: Optional[Sequence[int]] = None,
    price_col: str = "closeadj"
) -> pd.DataFrame:
    """
    Given a DataFrame with ['ticker','date', price_col] (or indexed by those),
    compute forward returns and direction flags at exactly h business days ahead.

    Returns a DataFrame indexed by ['ticker','date'] with:
      - return_{h}d = (price_{t + BDay(h)} / price_t) - 1
      - dir_{h}d    = 1 if return_{h}d > 0 else 0
    """
    # 1) Normalize input into columns
    df2 = df.copy()
    if {"ticker", "date"}.issubset(df2.index.names):
        df2 = df2.reset_index()

    # 2) Parse & sort by date
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 3) Default to 5 business days if no horizons supplied
    if horizons is None:
        horizons = [5]

    # 4) For each horizon, build out the shifted dates and merge prices
    out = df2[["ticker", "date", price_col]].copy()
    for h in horizons:
        date_h = f"date_{h}d"
        price_h = f"price_{h}d"
        ret_h   = f"return_{h}d"
        dir_h   = f"dir_{h}d"

        # compute the exact future business‐day
        out[date_h] = out["date"] + BDay(h)

        # prepare a lookup table of actual close prices at those future dates
        tmp = df2[["ticker", "date", price_col]].rename(
            columns={"date": date_h, price_col: price_h}
        )

        # left‐merge so any missing future dates become NaN
        out = out.merge(tmp, on=["ticker", date_h], how="left")

        # compute return & direction
        out[ret_h] = out[price_h] / out[price_col] - 1
        out[dir_h] = (out[ret_h] > 0).astype(int)

    # 5) re‐index for easy lookup
    return out.set_index(["ticker", "date"], drop=False)
