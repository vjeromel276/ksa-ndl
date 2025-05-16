# models/baseline.py
#!/usr/bin/env python3
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

from models.metrics import return_accuracy, regression_mae

def train_baseline_classification(X_train, y_train):
    """
    A trivial “most-frequent” classifier.
    """
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model

def train_baseline_regression(X_train, y_train):
    """
    A trivial “mean” regressor.
    """
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    return model

def train_and_evaluate(X, y, splitter, mode="classify"):
    """
    Run rolling time-series CV over (X, y), return one score per fold.

    Parameters
    ----------
    X : DataFrame
      Features, indexed by ['ticker','date'].
    y : Series
      Target (dir_5d or return_5d), indexed by ['ticker','date'].
    splitter : TimeSeriesSplitter
    mode : {"classify","regress"}
      Which baseline+metric to use.

    Returns
    -------
    List[float]
      One accuracy (if classify) or MAE (if regress) per fold.
    """
    # we need a flat DataFrame to split on the 'date' column
    X_df = X.reset_index()
    scores = []

    for train_idx, test_idx in splitter.split(X_df, date_col="date"):
        # re-index back to (ticker,date)
        X_tr = X_df.iloc[train_idx].set_index(["ticker","date"])
        X_te = X_df.iloc[test_idx].set_index(["ticker","date"])
        y_tr = y.loc[X_tr.index]
        y_te = y.loc[X_te.index]

        if mode == "classify":
            clf = train_baseline_classification(X_tr, y_tr)
            preds = clf.predict(X_te)
            score = return_accuracy(y_te, preds)
        else:
            reg = train_baseline_regression(X_tr, y_tr)
            preds = reg.predict(X_te)
            score = regression_mae(y_te, preds)

        scores.append(score)

    return scores
