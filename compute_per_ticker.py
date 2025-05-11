#!/usr/bin/env python3
import pandas as pd
import pandas_market_calendars as mcal
import sys

# ── PARAMETERS ────────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD = 0.99     # require ≥99% trading‑day coverage
VOLUME_THRESHOLD   = 100_000  # require ≥100k avg daily volume
SEP_MASTER         = "sep_dataset/SHARADAR_SEP.parquet"
TICKERS_META       = "sep_dataset/SHARADAR_TICKERS_2.parquet"

# 1) Load master SEP (prices) and parse dates
df = pd.read_parquet(SEP_MASTER, columns=["ticker","date","volume"])
print(f"[INFO] Loaded {SEP_MASTER} ({df.shape[0]} rows)")
df["date"] = pd.to_datetime(df["date"]).dt.date
print(f"[INFO] Parsed date column ({df['date'].min()} to {df['date'].max()})")

# 1b) Load ticker metadata for category & exchange
meta = pd.read_parquet(
    TICKERS_META,
    columns=["ticker","exchange","category"]
)
print(f"[INFO] Unique exchanges: {meta['exchange'].unique()}")
print(f"[INFO] Unique categories: {meta['category'].unique()[:10]}")

# 2) Filter to Common Stock tickers (case‑insensitive substring)
is_common = meta["category"].str.contains("Common Stock", case=False, na=False)
valid = meta[is_common]["ticker"].unique()
print(f"[INFO] Common Stock tickers: {len(valid)} of {meta.shape[0]}")
if len(valid) == 0:
    sys.exit("[ERROR] No Common Stock tickers found. Check 'category' values.")

# Restrict SEP to the clean list
before = df.shape[0]
df = df[df.ticker.isin(valid)]
after = df.shape[0]
print(f"[INFO] Restricted SEP from {before}→{after} rows for common stocks")

# 3) Load ticker metadata for price dates
meta_dates = pd.read_parquet(
    TICKERS_META,
    columns=["ticker","firstpricedate","lastpricedate"]
)
meta_dates = meta_dates.rename(columns={
    "firstpricedate": "listed",
    "lastpricedate":  "delisted"
})
meta_dates["listed"]   = pd.to_datetime(meta_dates["listed"], errors="coerce").dt.date
meta_dates["delisted"] = pd.to_datetime(meta_dates["delisted"], errors="coerce").dt.date

# 4) Build master NYSE calendar
nyse          = mcal.get_calendar("NYSE")
global_start  = df["date"].min()
global_end    = df["date"].max()
sched         = nyse.schedule(start_date=global_start, end_date=global_end)
all_days      = sorted(sched.index.date)

# 5) Compute coverage per ticker
cov = []
for ticker, sub in df.groupby("ticker"):
    row = meta_dates[meta_dates["ticker"] == ticker]
    listed   = row["listed"].iat[0]   if not row.empty and pd.notna(row["listed"].iat[0]) else global_start
    delisted = row["delisted"].iat[0] if not row.empty and pd.notna(row["delisted"].iat[0]) else global_end

    win_start = max(listed, global_start)
    win_end   = min(delisted, global_end)
    expected_days = [d for d in all_days if win_start <= d <= win_end]
    have_days     = [d for d in sub["date"] if win_start <= d <= win_end]
    coverage      = len(have_days) / len(expected_days) if expected_days else 0.0

    cov.append({
        "ticker":        ticker,
        "listed":        listed,
        "delisted":      delisted,
        "win_start":     win_start,
        "win_end":       win_end,
        "have_days":     len(have_days),
        "expected_days": len(expected_days),
        "coverage":      coverage
    })

cov_df = pd.DataFrame(cov).sort_values("coverage", ascending=False)

# 6) Export coverage metrics
cov_df.to_csv("ticker_coverage.csv", index=False)
print(f"Wrote ticker_coverage.csv ({len(cov_df)} rows)")

# 7) Compute and merge average volume
avg_vol = df.groupby("ticker")["volume"].mean().reset_index(name="avg_volume")
cov_vol = cov_df.merge(avg_vol, on="ticker", how="left")
cov_vol.to_csv("ticker_coverage_with_volume.csv", index=False)
print(f"Wrote ticker_coverage_with_volume.csv ({len(cov_vol)} rows)")

# 8) Filter by coverage + liquidity
clean = cov_vol[(cov_vol.coverage >= COVERAGE_THRESHOLD) & (cov_vol.avg_volume >= VOLUME_THRESHOLD)]
clean.to_csv("ticker_universe_clean.csv", index=False)
print(f"Wrote ticker_universe_clean.csv ({len(clean)} tickers)")

# 9) Summary stats
print("\nFilter summary:")
print(f"  ≥{int(COVERAGE_THRESHOLD*100)}% coverage: {cov_vol[cov_vol.coverage>=COVERAGE_THRESHOLD].shape[0]}")
print(f"  ≥{VOLUME_THRESHOLD:,} avg volume:       {cov_vol[cov_vol.avg_volume>=VOLUME_THRESHOLD].shape[0]}")
print(f"  Both criteria:       {len(clean)} tickers")
print(f"  Filtered out:        {len(cov_vol)-len(clean)} tickers")
print(f"  Total tickers:       {len(cov_vol)}")
print(f"  Clean tickers:       {len(clean)}")
print(f"  Clean coverage:      {len(clean) / len(cov_vol):.2%}")
print(f"  Clean avg volume:    {clean['avg_volume'].mean():,.0f}")