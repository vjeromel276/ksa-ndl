from typing import List
import pandas as pd

def build(sep: pd.DataFrame, windows: List[int] = [21, 63]) -> pd.DataFrame:
    """
    Compute liquidity proxies:
      - dvol_{w}d: rolling mean of dollar volume over w days
      - tov_{w}d: rolling mean of turnover (volume / shares_outstanding) if available

    Args:
      sep: DataFrame with ['ticker','date','close','volume'], sorted by ticker+date.
      windows: lookback windows in trading days.

    Returns:
      DataFrame indexed by ['ticker','date'] with columns dvol_{w}d (and tov_{w}d if present).
    """
    sep = sep.sort_values(['ticker','date'])
    # dollar volume
    dvol = sep['close'] * sep['volume']
    out = []

    for w in windows:
        name = f"dvol_{w}d"
        s = (
            dvol
            .groupby(sep['ticker'])
            .rolling(window=w)
            .mean()
            .reset_index(level=0, drop=True)
            .rename(name)
        )
        out.append(s)

    # If turnover column exists, compute similarly
    if 'turnover' in sep.columns:
        for w in windows:
            name = f"tov_{w}d"
            s = (
                sep['turnover']
                .groupby(sep['ticker'])
                .rolling(window=w)
                .mean()
                .reset_index(level=0, drop=True)
                .rename(name)
            )
            out.append(s)

    df = pd.concat(out, axis=1)
    df.index = sep.set_index(['ticker','date']).index
    return df

def build_rolling(sep: pd.DataFrame, windows: List[int] = [21, 63]) -> pd.DataFrame:
    """
    Compute liquidity proxies:
      - dvol_{w}d: rolling mean of dollar volume over w days
      - tov_{w}d: rolling mean of turnover (volume / shares_outstanding) if available

    Args:
      sep: DataFrame with ['ticker','date','close','volume'], sorted by ticker+date.
      windows: lookback windows in trading days.

    Returns:
      DataFrame indexed by ['ticker','date'] with columns dvol_{w}d (and tov_{w}d if present).
    """
    sep = sep.sort_values(['ticker','date'])
    # dollar volume
    dvol = sep['close'] * sep['volume']
    out = []

    for w in windows:
        name = f"dvol_{w}d"
        s = (
            dvol
            .groupby(sep['ticker'])
            .rolling(window=w)
            .mean()
            .reset_index(level=0, drop=True)
            .rename(name)
        )
        out.append(s)

    # If turnover column exists, compute similarly
    if 'turnover' in sep.columns:
        for w in windows:
            name = f"tov_{w}d"
            s = (
                sep['turnover']
                .groupby(sep['ticker'])
                .rolling(window=w)
                .mean()
                .reset_index(level=0, drop=True)
                .rename(name)
            )
            out.append(s)

    df = pd.concat(out, axis=1)
    df.index = sep.set_index(['ticker','date']).index
    return df
