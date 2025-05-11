#!/usr/bin/env python3
import os
import nasdaqdatalink
import pandas as pd
import pandas_market_calendars as mcal


#import tickers.parquet
tickers = pd.read_parquet('tickers.parquet')
# Set the environment variable for the API key
api_key=os.environ['NASDAQ_API_KEY']
# 1) Authentication
nasdaqdatalink.ApiConfig.api_key = api_key
# 2) Build the NYSE trading calendar
nyse = mcal.get_calendar('NYSE')
sched = nyse.schedule(start_date='1998-01-01', end_date='2025-05-10')
trading_days = sched.index  # DatetimeIndex of all open sessions

def main():

    for ticker in tickers['ticker']:
        # 3) Fetch SEP data in bulk (qopts.export for full CSV if needed)
       
        # Fetching data for a specific stock symbol
        data = nasdaqdatalink.get_table('SHARADAR/INDICATORS',
                                            start_date='1998-01-01',
                                            end_date='2025-05-10',
                                            qopts={'export': 'true'},
                                            paginate=True)


if __name__ == "__main__":
    main()
