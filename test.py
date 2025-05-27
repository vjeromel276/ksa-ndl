import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1) Load your features Parquet
#    (adjust path as needed)
df = pd.read_parquet("sep_dataset/features_2025-05-23.parquet")
# df = pd.read_parquet("sep_dataset/SHARADAR_SEP_filtered_2025-05-23.parquet")
# df = pd.read_parquet("sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet")

# 2) Data-quality checks
print("=== Missing values per column ===")
print(df.isna().sum().sort_values(ascending=False).head(20))
print()

print("=== Basic summary statistics ===")
print(df.describe().T)

# 3) Correlation with a simple binary target
#    Here: next-day return > 0
df = df.sort_values(["ticker", "date"])
df["next_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["target"] = (df["next_return"] > 0).astype(int)

corr = df.corr()["target"].abs().sort_values(ascending=False)
print()
print("=== Top 10 features by correlation with target ===")
print(corr.head(11))  # includes 'target' itself at top

# 4) Quick time-series cross-validation
#    Drop non-feature cols
X = df.drop(columns=["ticker","date","next_return","target"])
y = df["target"].fillna(0).astype(int)

tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

scores = cross_val_score(model, X.fillna(0), y, cv=tscv, scoring="accuracy")
print()
print("=== TimeSeries CV Accuracy Scores ===")
print(scores)
print("Mean accuracy:", scores.mean())
