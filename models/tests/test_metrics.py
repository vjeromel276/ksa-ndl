# models/tests/test_metrics.py

import numpy as np
import pytest

from models.metrics import return_accuracy, regression_mse, regression_mae

def test_return_accuracy():
    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]
    # 3 out of 4 correct
    assert pytest.approx(return_accuracy(y_true,y_pred)) == 0.75

def test_regression_errors():
    y_true = [1., 2., 3.]
    y_pred = [1.5, 1.5, 2.5]
    # MSE = ((0.5^2 + 0.5^2 + 0.5^2)/3) = 0.25/3 * 3 = 0.25
    assert pytest.approx(regression_mse(y_true,y_pred), rel=1e-6) == 0.25
    # MAE = (0.5 + 0.5 + 0.5)/3 = 0.5
    assert pytest.approx(regression_mae(y_true,y_pred), rel=1e-6) == 0.5
