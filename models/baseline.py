#!/usr/bin/env python3
"""
models/baseline.py

Baseline training routines for classification and regression with dummy,
XGBoost, and PyTorch backends — now with GPU‐optimized tree building
and GPU predictor support.
"""
import logging
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    # Torch only needed for the torch backend
    pass

logger = logging.getLogger(__name__)


def train_baseline_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    backend: str = "dummy",
    device: str = "cpu",
    num_classes: int = None,
):
    """
    Train a baseline classifier.
    backends:
      • dummy — sklearn DummyClassifier(strategy="most_frequent")
      • xgb   — XGBoost XGBClassifier(tree_method, predictor)
      • torch — PyTorch MLP classifier
    """
    if backend == "dummy":
        model = DummyClassifier(strategy="most_frequent")

    elif backend == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Please install xgboost to use the xgb backend")

        # EASY MISTAKE 👉 forget to switch predictor to GPU
        params = {
            "tree_method": "gpu_hist" if device == "gpu" else "hist",
            "eval_metric": "logloss"
        }
        if device == "gpu":
            # MUST match device string exactly
            params["predictor"] = "gpu_predictor"

        model = XGBClassifier(**params)

    elif backend == "torch":
        # (unchanged)
        X_tensor = torch.from_numpy(X_train.values).float()
        y_tensor = torch.from_numpy(y_train.values).long()
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        input_dim = X_train.shape[1]
        output_dim = num_classes or int(y_train.max()) + 1
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
        torch_device = torch.device("cuda" if device == "gpu" else "cpu")
        model = model.to(torch_device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(5):
            total_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(torch_device), yb.to(torch_device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            logger.debug("Epoch %d loss: %.4f", epoch, avg_loss)
        return model

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # fit on CPU or GPU as configured
    model.fit(X_train, y_train)
    return model


def train_baseline_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    backend: str = "dummy",
    device: str = "cpu"
):
    """
    Train a baseline regressor.
    backends:
      • dummy — sklearn DummyRegressor(strategy="mean")
      • xgb   — XGBoost XGBRegressor(tree_method, predictor)
      • torch — not implemented
    """
    if backend == "dummy":
        model = DummyRegressor(strategy="mean")

    elif backend == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Please install xgboost to use the xgb backend")

        # EASY MISTAKE 👉 using cpu_hist by accident
        params = {
            "tree_method": "gpu_hist" if device == "gpu" else "hist",
        }
        if device == "gpu":
            params["predictor"] = "gpu_predictor"

        model = XGBRegressor(**params)

    elif backend == "torch":
        raise NotImplementedError("PyTorch backend not implemented yet for regression")

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    model.fit(X_train, y_train)
    return model


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    mode: str = "classify",
    backend: str = "dummy",
    device: str = "cpu"
):
    """
    Run rolling time-series CV over (X, y), return one score per fold.

    mode: "classify" → accuracy; "regress" → MAE
    """
    # 0) Ensure feature columns are numeric
    try:
        X = X.astype(np.float32)
    except Exception as e:
        logger.error("Feature dtype conversion error: %s", e)
        raise

    # 1) Encode classification labels once
    num_classes = None
    if mode == "classify":
        le = LabelEncoder()
        y_array = le.fit_transform(y.values)
        logger.debug("LabelEncoder classes: %s", le.classes_)
        logger.debug("Encoded y sample (first 20): %s", y_array[:20])
        y = pd.Series(y_array, index=y.index)
        num_classes = len(le.classes_)

    X_df = X.reset_index()
    scores = []

    for train_idx, test_idx in splitter.split(X_df, date_col="date"):
        X_tr = X_df.iloc[train_idx].set_index(["ticker","date"])
        X_te = X_df.iloc[test_idx].set_index(["ticker","date"])
        y_tr = y.loc[X_tr.index]
        y_te = y.loc[X_te.index]

        if mode == "classify":
            model = train_baseline_classification(
                X_tr, y_tr,
                backend=backend,
                device=device,
                num_classes=num_classes
            )
            preds = model.predict(X_te)
            score = return_accuracy(y_te, preds)
        else:
            model = train_baseline_regression(
                X_tr, y_tr,
                backend=backend,
                device=device
            )
            preds = model.predict(X_te)
            score = regression_mae(y_te, preds)

        scores.append(score)

    return scores
