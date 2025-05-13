# models/tests/test_targets.py
import pandas as pd
import numpy as np
import pytest

from models.targets import make_targets

def make_dummy():
    # generate 12 consecutive business‐days starting 2025-01-01
    dates  = pd.bdate_range("2025-01-01", periods=12)
    # assign a strictly increasing integer price series
    prices = list(range(100, 100 + len(dates)))
    return (
        pd.DataFrame({
            "ticker": ["X"] * len(dates),
            "date": dates,
            "closeadj": prices
        })
        .set_index(["ticker", "date"])
    )

def test_return_and_direction_default_5d():
    df  = make_dummy()
    tgt = make_targets(df)

    # On 2025-01-01: +5 BDays → 2025-01-08, price=105 → (105/100)-1 = 0.05
    ret = tgt.loc[("X", pd.Timestamp("2025-01-01")), "return_5d"]
    assert pytest.approx(ret, rel=1e-6) == (105/100 - 1)
    assert tgt.loc[("X", pd.Timestamp("2025-01-01")), "dir_5d"] == 1

    # On 2025-01-02: +5 BDays → 2025-01-09, price=106 → (106/101)-1
    r2 = tgt.loc[("X", pd.Timestamp("2025-01-02")), "return_5d"]
    assert pytest.approx(r2, rel=1e-6) == (106/101 - 1)

def test_custom_horizon_and_price_col():
    df = make_dummy().rename(columns={"closeadj": "px"})
    tgt = make_targets(df, horizons=[3], price_col="px")

    # On 2025-01-01: +3 BDays → 2025-01-06, px=103 → (103/100)-1 = 0.03
    val = tgt.loc[("X", pd.Timestamp("2025-01-01")), "return_3d"]
    assert pytest.approx(val, rel=1e-6) == (103/100 - 1)
    assert tgt.loc[("X", pd.Timestamp("2025-01-01")), "dir_3d"] == 1
