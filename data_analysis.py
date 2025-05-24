#!/usr/bin/env python3
# data_analysis.py
# Console script to analyze and finalize SEP dataset filtering based on date argument.

import argparse
import pandas as pd
import pandas_market_calendars as mcal
from rich.console import Console
from rich.table import Table
import re
import os

console = Console()

def display_df(df: pd.DataFrame, title: str = None):
    if title:
        console.print(f"\n[bold underline]{title}[/]\n")
    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*[str(x) for x in row])
    console.print(table)

def is_valid_ticker(ticker: str) -> bool:
    return not bool(re.search(r'[Q\d\.]$', ticker))

def parse_args():
    parser = argparse.ArgumentParser(description="Finalize SEP filtering & perform analysis.")
    parser.add_argument(
        "--date", required=True,
        help="Snapshot date (YYYY-MM-DD) for analysis and final filtering."
    )
    parser.add_argument(
        "--min-windows", type=int, default=5,
        help="Minimum full 252-day windows required per ticker."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    date = args.date
    MIN_WINDOWS = args.min_windows

    # Define dynamic paths
    sep_input_path = f'sep_dataset/SHARADAR_SEP_filtered_{date}.parquet'
    sep_output_path = f'sep_dataset/SHARADAR_SEP_fully_filtered_{date}.parquet'
    summary_csv_path = f'ticker_coverage_summary_filtered_{date}.csv'

    # Check if input file exists
    if not os.path.exists(sep_input_path):
        console.print(f"[red bold]Input file missing:[/] {sep_input_path}")
        exit(1)

    # Load dataset
    df = pd.read_parquet(sep_input_path)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Filter special tickers
    original_count = df['ticker'].nunique()
    df = df[df['ticker'].apply(is_valid_ticker)]
    filtered_count = df['ticker'].nunique()
    console.print(f"[green]Filtered special tickers:[/] {original_count} → {filtered_count}")

    # NYSE calendar setup
    nyse = mcal.get_calendar('NYSE')
    sched = nyse.schedule(start_date='2000-01-01', end_date=date)
    all_trading_days = sched.index.date

    # Trading days per ticker
    ticker_dates = df.groupby('ticker')['date'].apply(set)

    # Coverage analysis
    records = []
    for ticker, dates in ticker_dates.items():
        coverage = len(dates)
        num_windows = coverage // 252
        if num_windows >= MIN_WINDOWS:
            records.append({
                'Ticker': ticker,
                'Trading Days': coverage,
                'Full 252-Day Windows': num_windows
            })

    # Create summary DataFrame and save
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(summary_csv_path, index=False)
    display_df(summary_df, f"Ticker Coverage (Filtered ≥ {MIN_WINDOWS} windows)")

    # Additional statistics summary
    stats = {
        'Total Tickers After Filter': len(summary_df),
        'Average Trading Days': round(summary_df['Trading Days'].mean(), 2),
        'Average Full 252-Day Windows': round(summary_df['Full 252-Day Windows'].mean(), 2),
        'Max Trading Days': summary_df['Trading Days'].max(),
        'Min Trading Days': summary_df['Trading Days'].min()
    }
    stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
    display_df(stats_df, "Summary Statistics")

    # Finalize SEP filtering based on final summary
    valid_tickers = set(summary_df['Ticker'])
    final_filtered_df = df[df['ticker'].isin(valid_tickers)]

    # Save final fully-filtered SEP
    final_filtered_df.to_parquet(sep_output_path, index=False)
    console.print(f"[bold green]🎯 Fully-filtered SEP saved:[/] {sep_output_path} "
                  f"({len(final_filtered_df):,} rows; {final_filtered_df['ticker'].nunique():,} tickers)")

if __name__ == "__main__":
    main()
