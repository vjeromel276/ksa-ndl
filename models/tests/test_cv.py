import pandas as pd
import numpy as np
import pytest

from models.cv import TimeSeriesSplitter

def make_dummy():
    # 10 business days from 2020-01-01
    dates = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame({
        "date":  dates,
        "value": np.arange(10)
    })

def test_basic_splits():
    df = make_dummy()
    # 5-day train, 2-day test, step=1 → expect 4 folds:
    #   train 0–4 → test 5–6
    #   train 1–5 → test 6–7
    #   train 2–6 → test 7–8
    #   train 3–7 → test 8–9
    splitter = TimeSeriesSplitter(train_window=5, test_window=2, step=1)
    splits = list(splitter.split(df, date_col="date"))
    assert len(splits) == 4

    # first fold
    train_idx, test_idx = splits[0]
    assert np.array_equal(train_idx, np.arange(0, 5))
    assert np.array_equal(test_idx,  np.arange(5, 7))

    # last fold
    train_idx, test_idx = splits[-1]
    assert np.array_equal(train_idx, np.arange(3, 8))
    assert np.array_equal(test_idx,  np.arange(8, 10))

def test_no_splits_if_insufficient_data():
    df = make_dummy().iloc[:6]  # only 6 rows
    splitter = TimeSeriesSplitter(train_window=5, test_window=2, step=1)
    splits = list(splitter.split(df, date_col="date"))
    # cannot fit a 2-day test after 5-day train in only 6 rows
    assert splits == []

def test_step_greater_than_one():
    df = make_dummy()
    # step=2: start at 0, then 2, then 4...
    splitter = TimeSeriesSplitter(train_window=4, test_window=2, step=2)
    splits = list(splitter.split(df, date_col="date"))
    # For 10 days, with 4+2 spans and step=2, we get 3 folds:
    #   train 0–3 → test 4–5
    #   train 2–5 → test 6–7
    #   train 4–7 → test 8–9
    assert len(splits) == 3

    t0, e0 = splits[0]
    assert np.array_equal(t0, np.arange(0, 4))
    assert np.array_equal(e0, np.arange(4, 6))

    t1, e1 = splits[1]
    assert np.array_equal(t1, np.arange(2, 6))
    assert np.array_equal(e1, np.arange(6, 8))

    t2, e2 = splits[2]
    assert np.array_equal(t2, np.arange(4, 8))
    assert np.array_equal(e2, np.arange(8, 10))
