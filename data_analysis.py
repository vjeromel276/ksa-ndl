import pandas as pd
import pandas_market_calendars as mcal
from rich.console import Console
from rich.table import Table

# Initialize console for rich output
console = Console()

def display_df(df: pd.DataFrame, title: str = None):
    """
    Display a pandas DataFrame in the console using Rich tables.
    """
    if title:
        console.print(f"\n[bold underline]{title}[/]\n")
    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*[str(x) for x in row])
    console.print(table)

# Load your dataset
sep_path = 'sep_dataset/SHARADAR_SEP_filtered_2025-05-23.parquet'
df = pd.read_parquet(sep_path, columns=['ticker', 'date'])

# Convert date column to datetime.date
df['date'] = pd.to_datetime(df['date']).dt.date

# Build NYSE trading calendar
nyse = mcal.get_calendar('NYSE')
sched = nyse.schedule(start_date='2000-01-01', end_date='2025-05-23')
all_trading_days = sched.index.date

# Determine trading days per ticker
ticker_dates = df.groupby('ticker')['date'].apply(set)

# Assess coverage and generate window counts
records = []
for ticker, dates in ticker_dates.items():
    available_days = sorted(dates)
    coverage = len(available_days)
    # Determine the number of non-overlapping 252-day windows
    num_windows = coverage // 252
    records.append({
        'Ticker': ticker,
        'Trading Days': coverage,
        'Full 252-Day Windows': num_windows
    })

# Create summary DataFrame
summary_df = pd.DataFrame(records)
summary_df.to_csv('ticker_coverage_summary.csv', index=False)
display_df(summary_df, "Ticker Coverage and 252-Day Window Count")
