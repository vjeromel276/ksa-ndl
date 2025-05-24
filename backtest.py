#!/usr/bin/env python3
# backtest.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_predictions(pred_path):
    return pd.read_csv(pred_path, parse_dates=['date'])

def load_actuals(actual_path):
    return pd.read_parquet(actual_path, columns=['ticker', 'date', 'close'])

def run_backtest(predictions, actuals, threshold=0.75):
    logging.info("Starting backtest...")
    merged = pd.merge(predictions, actuals, on=['ticker', 'date'], how='inner')
    # Simple entry logic example:
    merged['signal'] = (merged['prediction_prob'] > threshold).astype(int)
    merged['return'] = merged['signal'] * merged['next_period_return']
    return merged

def calculate_metrics(df):
    accuracy = (df['signal'] == df['actual_direction']).mean()
    cumulative_return = df['return'].sum()
    logging.info(f"Accuracy: {accuracy:.2%}, Cumulative Return: {cumulative_return:.2%}")
    return {"accuracy": accuracy, "cumulative_return": cumulative_return}

if __name__ == "__main__":
    preds = load_predictions('predictions.csv')
    actuals = load_actuals('actuals.parquet')
    results = run_backtest(preds, actuals)
    metrics = calculate_metrics(results)
