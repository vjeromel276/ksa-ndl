import pandas as pd
import numpy as np
import pytest

from models.targets import make_targets

def make_dummy():
    # 1 ticker, 12 trading days so 5-day shifts stay in bounds
    dates  = pd.bdate_range("2025-01-01", periods=12)
    prices = list(range(100, 100 + len(dates)))
    return pd.DataFrame({
        "ticker": ["X"] * len(dates),
        "date":   dates,
        "closeadj": prices
    }).set_index(["ticker", "date"])

def test_return_and_direction_default_5d():
    df  = make_dummy()
    tgt = make_targets(df)

    # On 2025-01-01: shift(-5) → 2025-01-08 has price=105 → (105/100)-1 == 0.05
    v = tgt.loc[("X", pd.Timestamp("2025-01-01")), "return_5d"]
    assert pytest.approx(v, rel=1e-6) == (105/100 - 1)
    assert tgt.loc[("X", pd.Timestamp("2025-01-01")), "dir_5d"] == 1

    # On 2025-01-02: shift(-5) → 2025-01-09 has price=106 → (106/101)-1
    r2 = tgt.loc[("X", pd.Timestamp("2025-01-02")), "return_5d"]
    assert pytest.approx(r2, rel=1e-6) == (106/101 - 1)

    # Beyond the last 5 rows → NaN & dir=0
    missing = pd.Timestamp("2025-01-10")  # first date where shift(-5) falls off end
    assert np.isnan(tgt.loc[("X", missing), "return_5d"])
    assert tgt.loc[("X", missing), "dir_5d"] == 0

def test_custom_horizon_and_price_col():
    df = make_dummy().rename(columns={"closeadj": "px"})
    tgt = make_targets(df, horizons=[3], price_col="px")

    # On 2025-01-01: shift(-3) → 2025-01-06 has px=105 → (105/100)-1
    val = tgt.loc[("X", pd.Timestamp("2025-01-01")), "return_3d"]
    assert pytest.approx(val, rel=1e-6) == (105/100 - 1)
    assert tgt.loc[("X", pd.Timestamp("2025-01-01")), "dir_3d"] == 1
