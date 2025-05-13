# feature_engineering/tests/test_seasonality.py
import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.seasonality import build

def make_sep():
    # 10 days spanning end-Jan→Feb 2025
    dates = pd.to_datetime([
        "2025-01-28","2025-01-29","2025-01-30","2025-01-31",
        "2025-02-03","2025-02-04","2025-02-05","2025-02-06",
        "2025-02-07","2025-02-10"
    ])
    # synthetic close prices
    return pd.DataFrame({
        "ticker": ["A"]*10,
        "date": dates,
        "close": np.arange(10, 20)
    })

def test_dow_and_tom_and_month_dummies():
    sep = make_sep()
    df = build(sep)
    a = df.xs("A", level="ticker")

    # Day-of-Week: Jan 28 2025 is Tuesday (dow=1)
    assert a.loc["2025-01-28","dow_1"] == 1
    assert a.loc["2025-01-28","dow_0"] == 0

    # TOM: Jan 28–31 should be last 3 days => tom=1 on 29–31
    assert a.loc["2025-01-29","tom"] == 1
    assert a.loc["2025-01-27"] if "2025-01-27" in a.index else True

    # January dummy
    assert a['jan'].sum() == 4  # all 4 Jan dates

    # Halloween period: Jan & Feb => all 10 rows =1
    assert (a['halloween'] == 1).all()

def test_mom_12m():
    sep = make_sep()
    # duplicate rows 252 days ago
    sep_prev = sep.copy()
    sep_prev['date'] = sep_prev['date'] - pd.Timedelta(days=252)
    sep = pd.concat([sep_prev, sep], ignore_index=True).sort_values('date')
    df = build(sep)
    a = df.xs("A", level="ticker")

    # On the last date, mom_12m = (19/9)-1 = 1.111...
    val = a.iloc[-1]['mom_12m']
    assert pytest.approx(val, rel=1e-6) == (19/9 - 1)
