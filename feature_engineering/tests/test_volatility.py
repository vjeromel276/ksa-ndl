import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.volatitlity import build

def make_sep():
    # Single ticker with 6 days of closeadj: deterministic increments
    data = {
        "ticker": ["A"] * 6,
        "date": pd.to_datetime([
            "2025-01-01","2025-01-02","2025-01-03",
            "2025-01-06","2025-01-07","2025-01-08"
        ]),
        # Quadratic or linear prices → returns vary predictably
        "closeadj": [100, 110, 121, 133.1, 146.41, 161.051]
    }
    return pd.DataFrame(data)

def test_volatility_single_window():
    sep = make_sep()
    # 2-day vol: std dev of last 2 returns
    df = build(sep, windows=[2])
    v = df.xs("A", level="ticker")["vol_2d"]

    # First 1 (w-1) entries should be NaN
    assert np.isnan(v.iloc[0])
    # At index 2 (third day), two returns: (110/100-1)=0.1, (121/110-1)=0.1 → std=0
    assert pytest.approx(v.iloc[2], abs=1e-12) == 0.0
    # At index 3: returns [0.1,0.1,0.1] rolling window=2 picks last two [0.1,0.1]
    assert pytest.approx(v.iloc[3], abs=1e-12) == 0.0

def test_volatility_multiple_windows():
    sep = make_sep()
    df = build(sep, windows=[1,3])
    # Should produce two columns
    assert set(df.columns) == {"vol_1d", "vol_3d"}

    # vol_1d is always NaN (std of single return)
    v1 = df.xs("A", level="ticker")["vol_1d"]
    assert v1.isna().all()

    # vol_3d at index 3: std of returns over 3 days
    rets = sep["closeadj"].pct_change()
    expected = np.std(rets.iloc[1:4], ddof=0)  # pandas uses ddof=0
    v3 = df.xs("A", level="ticker")["vol_3d"].iloc[3]
    assert pytest.approx(v3, rel=1e-6) == expected
