# 1) grab & merge the raw SEP snapshot for a given date
python daily_download.py --date 2025-05-23 --merge

# 2) restrict to common stocks, price/liquidity floors & ≥252 days history
python filter_common_with_history.py --date 2025-05-23

# 3) drop tickers with odd suffixes, enforce ≥5 full 252-day windows,
#    emit both a coverage CSV and your “fully filtered” SEP parquet
python data_analysis.py --date 2025-05-23

# 4) train your direction-classifier & return-regressor on that fully filtered SEP
python train_and_save_models.py \
  --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
  --horizon 5d

# 5) roll through history (or just do one-date) to produce predictions
python predict_history.py \
  --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
  --horizon 5 \
  --clf-model models/dir_5d_clf_2025-05-23.joblib \
  --reg-model models/return_5d_reg_2025-05-23.joblib \
  --threshold 0.95 \
  --output predictions/all_preds_2025-05-23.csv

# 6) backtest those signals (and even sweep multiple thresholds)
python backtest.py \
  --predictions predictions/all_preds_2025-05-28.csv \
  --actuals sep_dataset/SHARADAR_SEP_clean_universe_2025-05-23.parquet \
  --horizon 5 \
  --output backtest/full_backtest_2025-05-23.csv