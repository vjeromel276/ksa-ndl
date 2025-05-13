# feature_engineering/factors/seasonality.py
import pandas as pd
import numpy as np

def build(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Compute seasonal dummies and calendar returns:
      - dow_{0..4}: Day‐of‐Week one‐hot (Mon=0…Fri=4)
      - tom: Turn‐of‐Month (first 3 & last 3 trading days of each month)
      - jan: January dummy
      - halloween: 1 if month in [Nov–Apr]
      - mom_12m: same‐calendar 1‐year return (exactly 252 days prior)

    Args:
      sep: DataFrame with ['ticker','date','close'] (or 'closeadj']),
           sorted by ticker+date, covering only trading days.

    Returns:
      DataFrame indexed by ['ticker','date'] with seasonality columns.
    """
    # 1) Copy & sort
    df = sep.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker','date']).reset_index(drop=True)

    # 2) Build MultiIndex output
    idx = pd.MultiIndex.from_frame(df[['ticker','date']])
    out = pd.DataFrame(index=idx)

    # 3) Calendar arrays
    dow   = df['date'].dt.weekday.to_numpy()  # 0=Mon…6=Sun (we use 0–4)
    month = df['date'].dt.month.to_numpy()

    # Day‐of‐Week one‐hot
    for d in range(5):
        out[f'dow_{d}'] = (dow == d).astype(int)

    # 4) Turn‐of‐Month
    ym = df['date'].dt.to_period('M').astype(str)
    df['rank_fwd'] = df.groupby(['ticker', ym]).cumcount().to_numpy() + 1
    rev = df.iloc[::-1].reset_index(drop=True)
    rev['rank_rev'] = rev.groupby(
        ['ticker', rev['date'].dt.to_period('M').astype(str)]
    ).cumcount().to_numpy() + 1
    rank_rev = rev['rank_rev'].iloc[::-1].to_numpy()
    out['tom'] = ((df['rank_fwd'].to_numpy() <= 3) | (rank_rev <= 3)).astype(int)

    # 5) Month dummies
    out['jan']       = (month == 1).astype(int)
    out['halloween'] = np.isin(month, [11,12,1,2,3,4]).astype(int)

    # 6) Same‐calendar 1‐year return via explicit date_prev merge
    price_col = 'close' if 'close' in df.columns else 'closeadj'
    df['date_prev'] = df['date'] - pd.Timedelta(days=252)

    # Build lookup table on computed date_prev
    prev = df[['ticker','date_prev', price_col]].rename(
        columns={'date_prev': 'date', price_col: 'close_prev'}
    )

    # Merge back on actual 'date'
    merged = df.merge(prev, on=['ticker','date'], how='left')
    mom12 = merged[price_col] / merged['close_prev'] - 1

    # Fill NaNs (where no exact 252-day back match) with zero
    out['mom_12m'] = mom12.fillna(0).to_numpy()

    return out
