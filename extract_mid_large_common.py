#!/usr/bin/env python3
import pandas as pd

# ———————————————
# CONFIGURATION
# ———————————————
TICKERS_PARQ = "sep_dataset/SHARADAR_TICKERS_2.parquet"
OUTPUT_CSV   = "common_mid_large_caps.csv"

# We want any category that contains "Common Stock"
CATEGORY_SUBSTR = "Common Stock"

# ScaleMarketCap levels for Mid Cap or larger
TARGET_CAPS = {"4 - Mid", "5 - Large", "6 - Mega"}

def main():
    # 1) Load the full ticker metadata
    df = pd.read_parquet(TICKERS_PARQ)

    # 2) Convert listing/delisting to datetime.date
    df["firstpricedate"] = pd.to_datetime(df["firstpricedate"], errors="coerce").dt.date
    df["lastpricedate"]  = pd.to_datetime(df["lastpricedate"],  errors="coerce").dt.date

    # 3) Filter for any category containing "Common Stock"
    mask_category = df["category"].str.contains(CATEGORY_SUBSTR, na=False)

    # 4) Filter for scale market cap in {"4 - Mid", "5 - Large", "6 - Mega"}
    mask_cap = df["scalemarketcap"].isin(TARGET_CAPS)

    # 5) Apply both filters
    subset = df.loc[mask_category & mask_cap, ["ticker", "firstpricedate", "lastpricedate"]]

    # 6) Sort by ticker (optional)
    subset = subset.sort_values("ticker").reset_index(drop=True)

    # 7) Write out to CSV (dates in YYYY-MM-DD format)
    subset.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")

    print(f"Wrote {len(subset):,} tickers to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
