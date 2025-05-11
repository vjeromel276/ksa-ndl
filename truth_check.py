import pandas as pd

# 1) Load your final clean universe
clean = pd.read_csv("ticker_universe_clean.csv")
print(f"{len(clean)} tickers in clean universe\n")

# 2) Check exchange breakdown
meta = pd.read_parquet("sep_dataset/SHARADAR_TICKERS_2.parquet",
                       columns=["ticker","exchange","category"])
clean_meta = meta[meta["ticker"].isin(clean["ticker"])]
print("Tickers by exchange:")
print(clean_meta.groupby("exchange")["ticker"].nunique())

# 3) Check category consistency
print("\nTickers by category substring:")
cats = clean_meta["category"].value_counts()
print(cats[cats.index.str.contains("Common", case=False)])

# 4) Sample spot-checks
sample = clean["ticker"].sample(10, random_state=42).tolist()
print("\nSample tickers for manual review:", sample)

# 5) Verify expected_days vs. actual days distribution
cov = pd.read_csv("ticker_coverage_with_volume.csv")
check = cov[cov["ticker"].isin(clean["ticker"])]
print("\nCoverage stats for clean universe:")
print(check["coverage"].describe())

# 6) Ensure listed spans back before 2010 for legacy names
meta_dates = pd.read_parquet("sep_dataset/SHARADAR_TICKERS_2.parquet",
                             columns=["ticker","firstpricedate"])
meta_dates["firstpricedate"] = pd.to_datetime(meta_dates["firstpricedate"], errors="coerce").dt.date
legacy = meta_dates[meta_dates["ticker"].isin(clean["ticker"]) & 
                    (meta_dates["firstpricedate"] < pd.to_datetime("2010-01-01").date())]
print(f"\nTickers listed before 2010: {legacy['ticker'].nunique()} / {len(clean)}")
