
#                       EXECUTION CHECKLIST v0.1                          

# ── DATA PREP & FILTERING ────────────

1) Download & merge the raw SEP snapshot for your target date:
   python daily_download.py --date 2025-05-23 --merge

2) Restrict to common stocks, enforce price/liquidity floors & ≥252 days history:
   python filter_common_with_history.py --date 2025-05-23

3) Drop odd‐suffix tickers, ensure ≥5 full 252-day windows,
   emit both the coverage CSV and your “fully filtered” SEP parquet:
   python data_analysis.py --date 2025-05-23

# ── MODEL TRAINING & ONE-OFF PREDICTIONS ─────────────

4) Train your direction-classifier & return-regressor on that parquet:
   python train_and_save_models.py \
     --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
     --horizon 5d

5) Roll through history (or just do one date) to produce per-stock predictions:
   python predict_history.py \
     --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
     --horizon 5 \
     --clf-model models/dir_5d_clf_2025-05-23.joblib \
     --reg-model models/return_5d_reg_2025-05-23.joblib \
     --threshold 0.95 \
     --output predictions/all_preds_2025-05-23.csv

# ── SINGLE-PERIOD BACKTEST ────────────────

6) Backtest those signals (optionally sweeping thresholds):
   python backtest.py \
     --predictions predictions/all_preds_2025-05-23.csv \
     --actuals sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
     --horizon 5 \
     --output backtest/full_backtest_2025-05-23.csv

# ── WALK-FORWARD BACKTEST & VERSIONING ─────────

7) Run your new GPU-accelerated walk-forward backtest over all folds:
   python walk_forward_backtest.py \
     --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_2025-05-23.parquet \
     --universe-csv ticker_coverage_summary_filtered_2025-05-23.csv \
     --horizon 5d \
     --train-window 252 \
     --test-window 21 \
     --step-size 21 \
     --purge-days 4 \
     --embargo-days 2 \
     --backend xgb \
     --device gpu \
     --threshold 0.60 \
     --out-detail backtest/wf_folds_5d_purged_6t.csv \
     --out-summary backtest/wf_summary_5d_purged_6t.csv

8) Tag this milestone in git so you have a stable v0.1 reference:
   git tag v0.1  
   git push origin v0.1

# ── BRANCHING FOR HYPERPARAMETER TUNING ─────────

9) Create a new feature branch for your HPO experiments (keep main/develop clean):
   git checkout -b feature/hpo-xgb-2025-05

10) In that branch, wire up your tuning harness (Optuna, Ray Tune, etc.), 
    iterate freely, and commit only the minimal “skeleton” back to develop 
    once it’s stable.

# ─────────────────────────────────────────────────
