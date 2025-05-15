# models/metrics.py

#!/usr/bin/env python3
"""
METRICS MODULE CONTRACT
-----------------------
All metric functions take:
  • y_true: array‐like of true values
  • y_pred: array‐like of predicted values

And return a single float.

Functions:
  • return_accuracy(y_true, y_pred) → float
  • regression_mse(y_true, y_pred)   → float
  • regression_mae(y_true, y_pred)   → float
"""

import numpy as np

def return_accuracy(y_true, y_pred) -> float:
    """
    Classification accuracy: fraction of exact matches.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p))

def regression_mse(y_true, y_pred) -> float:
    """
    Mean squared error.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_p - y_t) ** 2))

def regression_mae(y_true, y_pred) -> float:
    """
    Mean absolute error.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_p - y_t)))
