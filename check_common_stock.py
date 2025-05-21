#!/usr/bin/env python3
import pandas as pd
import logging
import code

def configure_logging():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = configure_logging()

COMMON_FILE  = "sep_dataset/SHARADAR_SEP_common_2025-05-20.parquet"
TICKERS_FILE = "sep_dataset/SHARADAR_TICKERS_2.parquet"

# 1) Load your “common” universe
logger.debug(f"Loading common stock universe from {COMMON_FILE!r}")
common_df = pd.read_parquet(COMMON_FILE)
logger.debug(f" → common_df: {common_df.shape[0]} rows, {common_df.shape[1]} cols")

# 2) Load your master tickers file
logger.debug(f"Loading tickers metadata from {TICKERS_FILE!r}")
tickers_df = pd.read_parquet(TICKERS_FILE)
logger.debug(f" → tickers_df before dedupe: {tickers_df.shape[0]} rows, {tickers_df.shape[1]} cols")

# 3) Drop duplicate ticker rows (keep the first category seen)
before = len(tickers_df)
tickers_df = tickers_df.drop_duplicates(subset="ticker", keep="first")
after = len(tickers_df)
logger.debug(f"Dropped {before - after} duplicate ticker entries → {after} unique tickers")

# 4) Extract unique tickers from your common universe
unique_tickers = pd.Series(common_df["ticker"].unique(), name="ticker")
logger.debug(f"Found {len(unique_tickers)} unique tickers in common universe")

# 5) Build a mapping from ticker → category (now safe: unique index)
ticker_to_cat = tickers_df.set_index("ticker")["category"]

# 6) Map & detect anything that isn’t Common Stock
mapped = unique_tickers.to_frame().assign(
    category=lambda df: df["ticker"].map(ticker_to_cat)
)
logger.debug(f"After mapping, {mapped['category'].isna().sum()} tickers had no match")
non_common = mapped[mapped["category"] != "Common Stock"]
logger.debug(f"Tickers with category ≠ 'Common Stock': {len(non_common)}")

print("\n>>> REPL VARIABLES READY:")
print("    common_df     — full common universe DataFrame")
print("    tickers_df    — deduped ticker metadata")
print("    unique_tickers — Series of tickers from common_df")
print("    mapped        — DataFrame of ticker + category")
print("    non_common    — subset where category ≠ 'Common Stock'\n")

# 7) Interactive shell
code.interact(local=globals())
