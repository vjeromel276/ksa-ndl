# tests/test_compute_per_ticker.py
import pandas as pd
import pytest
from compute_per_ticker import main as compute_main

def make_sep(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA","AAA","BBB","BBB"],
        "date": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-01","2025-01-02"]),
        "volume": [200_000, 200_000, 50_000, 50_000]
    })
    path = tmp_path/"sep.parquet"
    df.to_parquet(path, index=False)
    return path

def make_meta(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA","BBB"],
        "exchange": ["XNYS","XNAS"],
        "category": ["Domestic Common Stock","Domestic Common Stock"],
        "firstpricedate": ["2024-12-31","2024-12-31"],
        "lastpricedate": ["2025-12-31","2025-12-31"]
    })
    path = tmp_path/"tickers.parquet"
    df.to_parquet(path, index=False)
    return path

def test_universe_filters(tmp_path, monkeypatch):
    sep_path = make_sep(tmp_path)
    meta_path = make_meta(tmp_path)

    # cd into tmp_path so outputs land here
    monkeypatch.chdir(tmp_path)

    # Run with thresholds: AAA has vol 200k ok, BBB has 50k -> should drop BBB
    compute_main([
        "--sep-master", str(sep_path),
        "--meta-table", str(meta_path),
        "--cov-thresh", "1.0",
        "--vol-thresh", "100000",
    ])

    cov = pd.read_csv(tmp_path/"ticker_coverage.csv")
    univ = pd.read_csv(tmp_path/"ticker_universe_clean.csv")

    assert set(cov.ticker) == {"AAA","BBB"}
    assert set(univ.ticker) == {"AAA"}
