import pandas as pd

def build(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Extract core quality ratios from SHARADAR/METRICS:
      - roe (return on equity)
      - roa (return on assets)
      - debt_to_equity
      - current_ratio
      - gross_margin
      - accruals = (net_income - cash_from_ops) / total_assets

    Args:
      metrics: DataFrame with columns
        ['ticker','date',
         'roe','roa','debt_to_equity','current_ratio','grossmargin',
         'netinc','cffo','assets']

    Returns:
      DataFrame indexed by ['ticker','date'] with those quality columns.
    """
    df = metrics.copy()
    # compute accruals if possible
    if {'netinc','cffo','assets'}.issubset(df.columns):
        df['accruals'] = (df['netinc'] - df['cffo']) / df['assets']
    # select & rename
    cols = {
        'roe':            'roe',
        'roa':            'roa',
        'debt_to_equity': 'debt_to_equity',
        'current_ratio':  'current_ratio',
        'grossmargin':    'gross_margin',
        'accruals':       'accruals'
    }
    out = (
        df
        .rename(columns=cols)
        .set_index(['ticker','date'])
        [list(cols.values())]
    )
    return out
