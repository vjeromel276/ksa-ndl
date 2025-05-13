import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.liquidity import build

def make_sep():
    # Two tickers, 5 days of data
    data = {
        "ticker": ["A"]*5 + ["B"]*5,
        "date": pd.to_datetime([
            "2025-01-01","2025-01-02","2025-01-03",
            "2025-01-06","2025-01-07"] * 2),
        "close":   [10,12,11,13,12] * 2,
        "volume":  [100,150,200,250,300] * 2
    }
    return pd.DataFrame(data)

def test_dvol_only():
    sep = make_sep()
    df = build(sep, windows=[2])
    a = df.xs("A", level="ticker")["dvol_2d"]
    # First entry NaN
    assert np.isnan(a.iloc[0])
    # Second: mean of [(10*100),(12*150)] = mean([1000,1800]) = 1400
    assert pytest.approx(a.iloc[1], rel=1e-6) == (1000+1800)/2

def test_both_windows_and_turnover():
    sep = make_sep().copy()
    # add a turnover column (volume/shares), use simple 10k shares
    sep['turnover'] = sep['volume'] / 10_000
    df = build(sep, windows=[3])
    a = df.xs("A", level="ticker")
    # dvol_3d at index 2 (third day): mean of first 3 dollar-vols
    dvs = np.array([10*100,12*150,11*200])
    assert pytest.approx(a.loc[pd.Timestamp("2025-01-03"), "dvol_3d"], rel=1e-6) == dvs.mean()
    # tov_3d: mean of turnover [100/10k,150/10k,200/10k] = [0.01,0.015,0.02]
    tovs = np.array([0.01,0.015,0.02])
    assert pytest.approx(a.loc[pd.Timestamp("2025-01-03"), "tov_3d"], rel=1e-6) == tovs.mean()
