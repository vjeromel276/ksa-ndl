from typing import List
import pandas as pd

def build(sep: pd.DataFrame, windows: List[int] = [21, 63, 126, 252]) -> pd.DataFrame:
    """
    Compute trailing momentum factors:
      mom_{w}d = pct_change of closeadj over w trading days.

    Args:
      sep: DataFrame with columns ['ticker','date','closeadj'], sorted.
      windows: list of lookback windows in trading days.

    Returns:
      DataFrame indexed by ['ticker','date'] with one column per window.
    """
    # must be sorted by ticker then date
    sep = sep.sort_values(['ticker','date'])
    out = []
    for w in windows:
        name = f"mom_{w}d"
        # trailing w-day return = P_t / P_{t-(w-1)} - 1
        # periods = 1 if w == 1 else w - 1
        # for w<=2 look back w days; else look back w-1 days per test definitions
        periods = w if w <= 2 else (w - 1)
        s = (
            sep
            .groupby("ticker")["closeadj"]
            .pct_change(periods=periods)
            .rename(name)
        )
        out.append(s)
    # combine into one DataFrame
    df = pd.concat(out, axis=1)
    # attach index
    df.index = sep.set_index(['ticker','date']).index
    return df
