# ksa-ndl
# ksa-ndl

## Development Workflow

1. **Bootstrap ingest**  
   ```bash
   git checkout -b feature/bootstrap-full-history
   # ... implement ingest_sharadar_day.py tests ...
   pytest
   git merge main


Quant pipeline for 5-day return forecasting using Sharadar EOD data.

## Repo Structure

- `sep_dataset/` — raw and master Parquet data for SEP, ACTIONS, TICKERS, METRICS  
- `ingest_sharadar_day.py` — bootstrap + daily ingest script  
- `compute_per_ticker.py` — builds and filters clean common-stock universe  
- `completeness_check.py` / `export_missing.py` / `make_complete.py` — gap-filling tools  
- `compute_per_ticker.py` — coverage & liquidity metrics  
- `feature_engineering/` — (future) your factor calculators  
- `models/` — (future) training and backtest code  

## Branching & Commit Guidelines

- **Branch per feature**:  
  - `feature/<short-descriptor>`  
  - `fix/<short-descriptor>`  
- **Commit convention**: 

- Merge back to `main` via PR once each feature is done and tested.

## Getting Started

1. `git clone … && cd ksa-ndl`  
2. `pip install -r requirements.txt`  
3. Set your API key:  
 ```bash
 export NASDAQ_API_KEY=your_key_here

## 
## Getting Started

1. `git clone … && cd ksa-ndl`  
2. `pip install -r requirements.txt`  
3. Set your API key:  
 ```bash
 export NASDAQ_API_KEY=your_key_here

## 