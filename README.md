# ksa-ndl

Quant pipeline for 5-day return forecasting using Sharadar EOD data.

## Repo Structure

- **sep_dataset/**  
  Raw & master Parquet data for  
  - SEP (prices)  
  - ACTIONS (corporate events)  
  - TICKERS (metadata)  
  - METRICS (fundamentals)  

- **ingest_sharadar_day.py**  
  Bootstrap + daily ingest script  

- **completeness_check.py**, **export_missing.py**, **filter_missing_common.py**, **make_complete.py**  
  Gap-filling / backfill tools  

- **compute_per_ticker.py**  
  Per-ticker coverage & volume metrics → clean common-stock universe  

- **feature_engineering/** *(future)*  
  Factor calculators (momentum, volatility, liquidity, etc.)  

- **models/** *(future)*  
  Training, backtesting, and forecasting code  

## Development Workflow

Each of the six pipeline stages lives on its own feature branch.  
After you finish and test a stage, open a PR, merge to `main`, then delete the branch.

1. **Bootstrap ingest**  
   ```bash
   git checkout -b feature/bootstrap-full-history
   # implement ingest_sharadar_day.py + tests
   pytest
   git checkout main
   git merge --no-ff feature/bootstrap-full-history
   git branch -d feature/bootstrap-full-history
