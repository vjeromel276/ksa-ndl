#!/usr/bin/env python3
# fetch_data.py
# Utility to download recent historical OHLCV data for tickers using yfinance

import argparse
import pandas as pd
import yfinance as yf


def fetch_price_data(tickers, start_date, end_date, interval="1d"):  # interval can be 1d, 1h, etc.
    """
    Fetch OHLCV data from Yahoo Finance for given tickers and date range.

    Returns a DataFrame with columns:
      ['ticker','date','open','high','low','close','adj_close','volume']
    """
    # Download using yfinance
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by='ticker',
        auto_adjust=False,
        threads=True,
    )

    # If multiple tickers, data is a dict-like multi-indexed DataFrame
    if len(tickers) > 1:
        # Stack tickers
        frames = []
        for ticker in tickers:
            df = data[ticker].copy()
            df['ticker'] = ticker
            frames.append(df)
        data = pd.concat(frames)
    else:
        data = data.copy()
        data['ticker'] = tickers[0]

    # Reset index to have date column
    data.reset_index(inplace=True)
    data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }, inplace=True)

    # Reorder columns
    cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    return data[cols]


def main():
    parser = argparse.ArgumentParser(description="Download historical price data for tickers.")
    parser.add_argument(
        '--tickers', nargs='+', required=True,
        help='List of ticker symbols to fetch'
    )
    parser.add_argument(
        '--start', required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', required=True,
        help='End date (YYYY-MM-DD), exclusive'
    )
    parser.add_argument(
        '--interval', default='1d',
        help="Data interval (e.g., '1d', '1h')"
    )
    parser.add_argument(
        '--out', required=True,
        help='Output CSV file path'
    )
    args = parser.parse_args()

    df = fetch_price_data(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval
    )
    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved fetched data to {args.out}")


if __name__ == '__main__':
    main()
