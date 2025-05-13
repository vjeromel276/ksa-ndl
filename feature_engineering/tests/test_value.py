import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.value import build

def make_metrics():
    data = {
        "ticker": ["A","A","B","B"],
        "date": pd.to_datetime(["2025-01-01","2025-01-02"]*2),
        "pe": [10, 12, 8, np.nan],
        "pb": [1.5, 1.6, 0.8, 0.9],
        "ps": [2.0, 2.1, 1.0, 1.1],
        "ev_ebitda": [10, 12, 8, 9],
        "div_yield": [0.02, 0.021, 0.015, 0.016]
    }
    return pd.DataFrame(data)

def test_value_build():
    metrics = make_metrics()
    df = build(metrics)
    # shape: 4 rows × 5 columns
    assert df.shape == (4,5)
    # check a value
    assert df.loc[("A", pd.Timestamp("2025-01-02")), "pe"] == 12
    # preserve NaNs
    assert np.isnan(df.loc[("B", pd.Timestamp("2025-01-02")), "pe"])
