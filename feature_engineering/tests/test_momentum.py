import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.momentum import build

def make_sep():
    # Two tickers, 6 days of closeadj
    data = {
        "ticker": ["A"]*6 + ["B"]*6,
        "date": pd.to_datetime(
            ["2025-01-01","2025-01-02","2025-01-03",
             "2025-01-06","2025-01-07","2025-01-08"] * 2
        ),
        "closeadj": [100,102,104,106,108,110, 50,49,48,47,46,45]
    }
    return pd.DataFrame(data)

def test_momentum_single_window():
    sep = make_sep()
    # compute only 2-day momentum
    df = build(sep, windows=[2])

    # For ticker A: (104/100 -1)=0.04 on 2025-01-03; (106/102 -1)=0.0392...
    a = df.xs("A", level="ticker")["mom_2d"]
    # First two entries should be NaN
    assert np.isnan(a.iloc[0])
    assert np.isnan(a.iloc[1])
    # Third entry: (104/100)-1 = 0.04
    assert pytest.approx(a.iloc[2], rel=1e-6) == 0.04
    # Fourth: (106/102)-1 ≈ 0.0392157
    assert pytest.approx(a.iloc[3], rel=1e-6) == (106/102 - 1)

    # For ticker B: downward momentum
    b = df.xs("B", level="ticker")["mom_2d"]
    # Third entry: (48/50)-1 = -0.04
    assert pytest.approx(b.iloc[2], rel=1e-6) == -0.04

def test_momentum_multiple_windows():
    sep = make_sep()
    df = build(sep, windows=[1,3])
    # Should have two columns
    assert set(df.columns) == {"mom_1d", "mom_3d"}
    # Check a known value: A on 2025-01-06, 3-day mom: (106/102)-1
    val = df.loc[("A", pd.Timestamp("2025-01-06")), "mom_3d"]
    assert pytest.approx(val, rel=1e-6) == (106/102 - 1)
