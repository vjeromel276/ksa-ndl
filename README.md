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
python daily_download.py --date 2025-06-09 
python merge_daily_download.py 2025-06-09 --update-gold        
python data_analysis.py --date 2025-06-09 
python compute_per_ticker.py  --common-sep sep_dataset/SHARADAR_SEP_fully_filtered_2025-06-09.parquet
python compute_features.py
  --sep               sep_dataset/SHARADAR_SEP_fully_filtered_2025-06-09.parquet \
  --metrics           original_files/SHARADAR_METRICS_2.csv \
  --fundamentals      original_files/SHARADAR_SF1.csv \
  --liquidity-windows 21,63 \
  --momentum-windows  21,63,126,252 \
  --volatility-windows 20,50,100 \
  --out               sep_dataset/features_sep_training_universe.parquet
  
 python train_and_save_models.py \
  --sep-master sep_dataset/SHARADAR_SEP_filtered_2025-06-09.parquet \
  --features    sep_dataset/features_sep_training_universe.parquet \
  --horizon     5d \
  --backend     xgb \
  --device      gpu  --date 2025-06-09

❯ python predict_universe.py \
  --sep-master       sep_dataset/SHARADAR_SEP_fully_filtered_2025-06-09.parquet \
  --ticker-universe  ticker_universe_clean.csv \
  --features         sep_dataset/features_sep_training_universe.parquet \
  --date             2025-06-09 \
  --horizon          5d \
  --threshold        0.6

❯ python score_universe_predictions.py \
  --sep sep_dataset/SHARADAR_SEP_fully_filtered_2025-06-09.parquet  \
  --predictions predictions_universe_2025-06-09.parquet   --horizon 5d \
  --date 2025-06-09 \
  --output scored_2025-06-09_5d.csv

2025-06-09 19:27:34 INFO: Loading SEP from sep_dataset/SHARADAR_SEP_fully_filtered_2025-06-09.parquet
2025-06-09 19:27:39 INFO: Loading predictions from predictions_universe_2025-06-09.parquet
2025-06-09 19:27:39 INFO: Writing 1564 rows to scored_2025-06-09_5d.csv
2025-06-09 19:27:39 INFO: Also wrote Parquet → scored_2025-06-09_5d.parquet
2025-06-09 19:27:39 INFO: Scoring complete.
                              