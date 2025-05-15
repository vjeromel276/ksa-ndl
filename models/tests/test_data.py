# models/tests/test_data.py
import pandas as pd
import numpy as np
import pytest

from models.data import load_features, load_targets

def make_sep():
    # generate 12 consecutive business days starting 2025-01-01
    dates  = pd.bdate_range("2025-01-01", periods=12)
    # strictly increasing “prices”
    prices = list(range(100, 100 + len(dates)))
    # dummy OHLC + volume
    return pd.DataFrame({
        "ticker": ["X"] * len(dates),
        "date":   dates,
        "open":   prices,
        "high":   [p + 1 for p in prices],
        "low":    [p - 1 for p in prices],
        "close":  prices,
        "volume": np.arange(len(dates))
    })

def test_load_features_and_targets_alignment():
    sep = make_sep()
    X   = load_features(sep)
    y   = load_targets(sep)

    # both must be MultiIndexed by ticker,date
    assert X.index.names == ["ticker","date"]
    assert y.index.names == ["ticker","date"]

    # must include some seasonality + technical columns
    assert "dow_0" in X.columns
    assert "boll_mid" in X.columns

    # they should align exactly (same rows in same order)
    assert X.shape[0] == y.shape[0]
    assert X.index.equals(y.index)

    # spot-check the 5-day return:
    # 2025-01-01 + 5 BDays → 2025-01-08, price = 105 → (105/100) − 1 = 0.05
    ret5 = y.loc[("X", pd.Timestamp("2025-01-01")), "return_5d"]
    assert pytest.approx(ret5, rel=1e-6) == (105/100 - 1)
    # direction should be positive
    assert y.loc[("X", pd.Timestamp("2025-01-01")), "dir_5d"] == 1
