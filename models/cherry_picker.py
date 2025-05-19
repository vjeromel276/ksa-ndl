# models/cherry_picker.py
# Utility for selecting tickers with sufficient trading-history for a given prediction horizon.

import pandas as pd
from typing import List

# —— Hard-coded mapping from prediction horizon to minimum history in trading days
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
    uni = pd.read_csv(
        universe_csv,
        parse_dates=["listed","delisted","win_start","win_end"]
    )
    valid = uni.loc[uni["have_days"] >= min_days, "ticker"].unique().tolist()
    return valid
