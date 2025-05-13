import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.technicals import build

def make_sep():
    # One ticker, 30 days of synthetic close prices
    dates = pd.bdate_range("2025-01-01", periods=30)
    prices = np.linspace(100, 130, 30)  # linear uptrend
    return pd.DataFrame({
        "ticker": ["A"]*30,
        "date": dates,
        "close": prices
    })

def test_ma_and_cross():
    sep = make_sep()
    df = build(sep)
    a = df.xs("A", level="ticker")

    # Before day 20, ma20 is the exact np.nan singleton
    assert a['ma20'].iloc[18] is np.nan

    # On day 20, ma20 = mean of the first 20 close prices
    expected_ma20 = sep['close'].iloc[:20].mean()
    assert pytest.approx(a['ma20'].iloc[19], rel=1e-6) == expected_ma20

    # ma50 always NaN (only 30 days), so ma_cross should be 0
    assert a['ma50'].isna().all()
    assert (a['ma_cross'] == 0).all()

def test_rsi():
    sep = make_sep()
    df = build(sep)
    a = df.xs("A", level="ticker")

    # RSI14 on first 14 days should be the exact np.nan singleton
    assert a['rsi14'].iloc[13] is np.nan

    # RSI14 on day 15: since all returns positive, RSI should be very high
    assert a['rsi14'].iloc[14] > 90

def test_macd_and_signal():
    sep = make_sep()
    df = build(sep)
    a = df.xs("A", level="ticker")

    # MACD line should be defined by day 26 (max of 12 and 26-day EMAs)
    assert not np.isnan(a['macd'].iloc[25])

    # Signal line is the 9-day EMA of MACD
    macd_series = a['macd']
    signal_expected = macd_series.ewm(span=9, adjust=False).mean().iloc[25]
    assert pytest.approx(a['macd_signal'].iloc[25], rel=1e-6) == signal_expected

def test_bollinger_bands():
    sep = make_sep()
    df = build(sep)
    a = df.xs("A", level="ticker")

    # On day 20, boll_mid equals ma20
    assert pytest.approx(a['boll_mid'].iloc[19], rel=1e-6) == a['ma20'].iloc[19]

    # boll_upper - boll_mid == 2 * std20, where std20 is computed on sep['close'][:20]
    std20 = sep['close'].iloc[:20].std(ddof=0)
    diff = a['boll_upper'].iloc[19] - a['boll_mid'].iloc[19]
    assert pytest.approx(diff, rel=1e-6) == 2 * std20
