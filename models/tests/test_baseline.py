# models/tests/test_baseline.py
import pandas as pd
import numpy as np
import pytest

from models.baseline import train_and_evaluate
from models.cv import TimeSeriesSplitter
from models.metrics import return_accuracy, regression_mae

def make_dummy_features():
    # 6 consecutive business‐days for a single ticker, two simple features
    dates = pd.bdate_range("2025-01-01", periods=6)
    df = pd.DataFrame({
        "ticker": ["X"]*6,
        "date": dates,
        "f1": np.arange(6),      # [0,1,2,3,4,5]
        "f2": np.arange(10,16),  # [10,11,12,13,14,15]
    })
    return df.set_index(["ticker","date"])

def make_classification_target(X):
    # pretend “dir=1” if f1 ≥ 3 else 0
    return (X["f1"] >= 3).astype(int)

def make_regression_target(X):
    # a toy “return” equal to f1 * 0.1
    return X["f1"] * 0.1

def test_train_and_evaluate_classification():
    X = make_dummy_features()
    y = make_classification_target(X)

    # one fold: train 4 days → test 2 days
    splitter = TimeSeriesSplitter(train_window=4, test_window=2, step=2)
    scores = train_and_evaluate(X, y, splitter, mode="classify")

    # should produce exactly one fold
    assert isinstance(scores, list)
    assert len(scores) == 1

    # compute expected “most frequent” baseline accuracy by hand
    train_y = y.iloc[:4]
    test_y  = y.iloc[4:]
    majority = train_y.mode().iat[0]
    expected_score = return_accuracy(test_y, [majority]*len(test_y))
    assert pytest.approx(scores[0], rel=1e-8) == expected_score

def test_train_and_evaluate_regression():
    X = make_dummy_features()
    y = make_regression_target(X)

    splitter = TimeSeriesSplitter(train_window=4, test_window=2, step=2)
    scores = train_and_evaluate(X, y, splitter, mode="regress")

    assert isinstance(scores, list)
    assert len(scores) == 1

    # expected DummyRegressor always predicts the TRAIN mean
    train_y = y.iloc[:4]
    test_y  = y.iloc[4:]
    pred_val = train_y.mean()
    expected_score = regression_mae(test_y, [pred_val]*len(test_y))
    assert pytest.approx(scores[0], rel=1e-8) == expected_score
