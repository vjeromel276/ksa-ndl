#!/usr/bin/env python3
import joblib, pandas as pd, numpy as np, cupy as cp
from xgboost import DMatrix
from models.data import load_features, _coerce_sep_dtypes
from core.schema import validate_full_sep

# 1) Load & validate SEP (up to your as-of-date)
sep = pd.read_parquet("sep_dataset/SHARADAR_SEP_common.parquet")
sep = _coerce_sep_dtypes(sep)
validate_full_sep(sep)

# 2) Build features, cast to float32
X = load_features(sep).astype(np.float32)

# 3) Filter to your as-of-date (e.g. the most recent date in SEP)
as_of = pd.to_datetime("2025-05-09")
X_date = X.xs(as_of, level="date")  # drops the date index, leaves index=ticker

# 4) Move to GPU and build DMatrix
feature_names = X_date.columns.tolist()
gpu_array = cp.asarray(X_date.values)
dmat = DMatrix(gpu_array, feature_names=feature_names)

# 5) Load your classifier and predict probabilities
clf = joblib.load("models/dir_5d_clf.joblib")
probs = clf.get_booster().predict(dmat)  # array of P(up)

# 6) Load your regressor and predict returns
reg = joblib.load("models/return_5d_reg.joblib")
rets = reg.get_booster().predict(dmat)    # array of 5d_return

# 7) Assemble results
results = pd.DataFrame({
    "ticker": X_date.index,
    "P_up": probs,
    "pred_return": rets
})
# Apply threshold
th = 0.95
results["signal"] = np.where(results["P_up"] >= th, "up",
                      np.where(1 - results["P_up"] >= th, "down", "no_signal"))

# 8) Inspect top signals
top_up     = results[results.signal=="up"].sort_values("P_up", ascending=False)
top_down   = results[results.signal=="down"].sort_values("P_up")
print("High-confidence UP signals:\n", top_up.head(10))
print("\nHigh-confidence DOWN signals:\n", top_down.head(10))
