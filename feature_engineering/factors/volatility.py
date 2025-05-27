from typing import List
import pandas as pd

def build(sep: pd.DataFrame, windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
    """
    Compute rolling volatility factors:
      vol_{w}d = rolling standard deviation of trailing closeadj returns over w days.

    Args:
      sep: DataFrame with ['ticker','date','closeadj'], sorted by ticker+date.
      windows: list of lookback windows in trading days.

    Returns:
      DataFrame indexed by ['ticker','date'] with vol_{w}d columns.
    """
    # ensure correct sort
    sep = sep.sort_values(['ticker','date'])
    # pct change returns
    rets = sep.groupby("ticker")["closeadj"].pct_change()
    out = []
    for w in windows:
        name = f"vol_{w}d"
        # group returns by ticker, compute rolling std, then align back
        series = (
            rets
            .groupby(sep["ticker"])
            .rolling(window=w)
            .std()
            .reset_index(level=0, drop=True)
            .rename(name)
        )
        out.append(series)
    # concatenate and set proper MultiIndex
    df = pd.concat(out, axis=1)
    df.index = sep.set_index(['ticker','date']).index
    return df
