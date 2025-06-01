import pandas as pd

# 1) Load the “full” SEP
df_full = pd.read_parquet("sep_dataset/SHARADAR_SEP_2.parquet")

# 2) Suppose “my_large_caps” is a precomputed list of tickers for your training universe:
large_caps = pd.read_csv("ticker_history_completeness.csv")["ticker"].tolist()

# 3) Select only rows AND exactly the columns your schema needs:
required_cols = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "closeadj", "closeunadj","lastupdated"   # ← make sure this is not dropped
    # …plus any other columns that validate_full_sep expects (e.g. lastupdated)…
]

df_tv = (
    df_full
    .loc[ df_full["ticker"].isin(large_caps), required_cols ]
    .copy()
)

# 4) Write it back out:
df_tv.to_parquet("sep_dataset/sep_training_universe.parquet")
