# models/cherry_picker.py
# Utility for selecting tickers with sufficient trading‐history for a given prediction horizon.

import pandas as pd
from typing import List

# — map from prediction horizon to minimum history in trading days
_HORIZON_DAYS = {
    "1d":  52,
    "5d":  52 * 5,
    "10d": 52 * 10,
    "30d": 52 * 30,
}

def get_valid_tickers_for_horizon(
    universe_csv: str,
    horizon: str
) -> List[str]:
    """
    Returns the subset of tickers from `universe_csv` that have at least
    the required number of trading days for the given `horizon`.
    
    Args:
      universe_csv: Path to your ticker_universe_clean_<date>.csv
      horizon: One of "1d", "5d", "10d", or "30d"

    Returns:
      A list of tickers meeting the minimum‐history requirement.
    """
    if horizon not in _HORIZON_DAYS:
        raise ValueError(
            f"horizon must be one of {_HORIZON_DAYS.keys()}, got {horizon!r}"
        )
    min_days = _HORIZON_DAYS[horizon]

    # 1) read in whatever CSV you point at
    uni = pd.read_csv(universe_csv, dtype=str)

    # 2) normalize all column‐names to snake_case lowercase
    norm = {
        col: col.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
        for col in uni.columns
    }
    uni.rename(columns=norm, inplace=True)

    # 3) locate your ticker column
    if "ticker" not in uni.columns:
        raise ValueError(
            f"Could not find a 'ticker' column in {universe_csv!r}; "
            f"found columns: {list(uni.columns)}"
        )

    # 4) locate your “days of history” column
    if   "have_days"    in uni.columns:
        days_col = "have_days"
    elif "trading_days" in uni.columns:
        days_col = "trading_days"
    else:
        raise ValueError(
            f"Could not find a trading‐day count column in {universe_csv!r}; "
            f"expected 'have_days' or 'trading_days', got {list(uni.columns)}"
        )

    # 5) filter and return
    #    cast to numeric in case it was read as string
    uni[days_col] = pd.to_numeric(uni[days_col], errors="coerce")
    good = uni.loc[uni[days_col] >= min_days, "ticker"]
    return good.dropna().unique().tolist()
