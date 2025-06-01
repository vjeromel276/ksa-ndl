#!/usr/bin/env python3
import pandas as pd

# -------------------------------
# CONFIGURATION
# -------------------------------
TICKERS_PARQ = "sep_dataset/SHARADAR_TICKERS_2.parquet"
OUTPUT_CSV   = "common_mid_large_primary_caps.csv"

CATEGORY_SUBSTR = "Common Stock"          # any category containing this
TARGET_CAPS     = {"4 - Mid", "5 - Large", "6 - Mega"}  
TICKER_REGEX    = r"^[A-Z]+$"              # only A–Z, no digits or punctuation

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

    # 5) Build ticker mask:
    #    a) Letters only (A–Z)
    mask_letters = df["ticker"].str.match(TICKER_REGEX)

    #    b) Identify delisted flags (True/Y/1 → delisted)
    isdel_s = df["isdelisted"].astype(str).str.upper()
    mask_delisted = isdel_s.isin({"TRUE", "Y", "1"})

    #    c) Only drop “Q” if delisted; otherwise keep
    #       - drop_if_q: ticker endswith 'Q' AND is marked delisted
    drop_if_q = df["ticker"].str.endswith("Q") & mask_delisted
    mask_not_badQ = ~drop_if_q

    #    d) Combined ticker mask
    mask_ticker = mask_letters & mask_not_badQ

    # 6) Apply all filters
    subset = df.loc[
        mask_category & mask_cap & mask_ticker,
        ["ticker", "firstpricedate", "lastpricedate"]
    ]

    # 7) Drop duplicates so each ticker appears only once
    subset = subset.drop_duplicates(subset=["ticker"])

    # 8) Sort by ticker
    subset = subset.sort_values("ticker").reset_index(drop=True)

    # 9) Write out to CSV
    subset.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(subset):,} unique tickers to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
