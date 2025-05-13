import pandas as pd
import numpy as np
import pytest

from feature_engineering.factors.quality import build

def make_metrics():
    data = {
        'ticker': ['A','A','B','B'],
        'date': pd.to_datetime(['2025-01-01','2025-01-02']*2),
        'roe': [0.10,0.12,0.08,0.09],
        'roa': [0.05,0.06,0.04,0.045],
        'debt_to_equity': [1.5,1.4,0.8,0.85],
        'current_ratio': [2.0,1.9,1.5,1.6],
        'grossmargin': [0.40,0.42,0.35,0.36],
        'netinc': [10,11,8,9],
        'cffo': [8,9,7,7.5],
        'assets': [100,110,80,90]
    }
    return pd.DataFrame(data)

def test_quality_build():
    metrics = make_metrics()
    df = build(metrics)
    # shape: 4 rows × 6 columns
    assert df.shape == (4,6)
    # check accruals for A@2025-01-01: (10-8)/100 = 0.02
    val = df.loc[('A', pd.Timestamp('2025-01-01')), 'accruals']
    assert pytest.approx(val, rel=1e-6) == 0.02
    # preserve values for other fields
    assert df.loc[('B', pd.Timestamp('2025-01-02')), 'roa'] == 0.045
