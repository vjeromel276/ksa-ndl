# ksa-ndl

Quant pipeline for forecasting multi-day Sharadar EOD returns.

---

## Repo Structure

### Data ingestion & maintenance  
- **sep_dataset/**  
  - `SHARADAR_SEP.parquet` — historical end-of-day price data  
  - `SHARADAR_ACTIONS.parquet` — corporate actions (splits, dividends)  
  - `SHARADAR_TICKERS.parquet` — ticker metadata  
  - `SHARADAR_METRICS.parquet` — fundamental metrics  
- **ingest_sharadar_day.py**  
  - Bootstrap full history & daily updates via NASDAQ API  

### Data quality & backfill  
- **completeness_check.py** — check for missing ticker×date combinations  
- **export_missing.py** — dump missing pairs to JSON  
- **filter_missing_common.py** — filter for common-stock tickers  
- **make_complete.py** — backfill missing dates by fetching data  

### Universe construction  
- **compute_per_ticker.py**  
  - Calculates per-ticker trading-day coverage & average volume  
  - Filters to a clean common-stock universe  

### Feature engineering  
- **feature_engineering/**  
  - `factors/` — individual factor modules (momentum, volatility, liquidity, value, quality, seasonality, technicals, …)  
  - `build.py` (or `__init__.py`) — orchestrator that maps raw SEP → feature matrix  

### Modeling & backtesting  
- **models/**  
  - `targets.py` — constructs forward-return & direction labels  
  - `cv.py` — time-series cross-validation utilities  
  - `train.py` — baseline training script (e.g. logistic/RF)  
  - `backtest.py` — rolling backtest harness  

### Notebooks & docs  
- **notebooks/** — exploratory analyses, model diagnostics  
- **docs/** — design docs, data dictionaries, API spec  

---

## Getting Started

1. **Clone & install**  
   ```bash
   git clone <repo_url> ksa-ndl
   cd ksa-ndl
   pip install -r requirements.txt

   ###
   added backtesting 
