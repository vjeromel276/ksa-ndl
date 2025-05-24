#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 YYYY-MM-DD [HORIZON] [THRESHOLD]"
  exit 1
fi

DATE=$1
HORIZON=${2:-5d}                # default 5d
NUM_HORIZON=${HORIZON%d}        # strip trailing “d” for scripts that want int
THRESH=${3:-0.95}

echo "▶️  Running full pipeline for ${DATE}, horizon=${HORIZON}, threshold=${THRESH}"

echo "1) Download SEP…"
python daily_download.py --date $DATE --merge

echo "2) Filter common stock + history…"
python filter_common_with_history.py --date $DATE

echo "3) Analyze & fully-filter (≥5 windows)…"
python data_analysis.py --date $DATE

echo "4) Train models…"
python train_and_save_models.py \
  --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_${DATE}.parquet \
  --universe-csv ticker_universe_clean_${DATE}.csv \
  --horizon $HORIZON

echo "5) Predict full history…"
python predict_history.py \
  --sep-master sep_dataset/SHARADAR_SEP_fully_filtered_${DATE}.parquet \
  --
