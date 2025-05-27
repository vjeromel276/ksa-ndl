# Full quality.py implementation with SF1 support

import pandas as pd
import warnings

def build(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Extract core quality ratios from SHARADAR SF1 fundamentals:
      - roe (return on equity)
      - roa (return on assets)
      - debt_to_equity (ratio)
      - current_ratio
      - gross_margin
      - accruals = (netinc - ncfo) / assets

    Args:
      metrics: DataFrame with at least:
        ['ticker', 'date', 'roe','roa','de','currentratio','grossmargin',
         'netinc','ncfo','assets']

    Returns:
      DataFrame indexed by ['ticker','date'] with columns:
      ['roe','roa','debt_to_equity','current_ratio','gross_margin','accruals'].
    """
    # 1) Copy & normalize column names
    df = metrics.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) Compute accruals if all pieces exist, else NA
    required = ['netinc', 'ncfo', 'assets']
    missing_req = [c for c in required if c not in df.columns]
    if not missing_req:
        df['accruals'] = (df['netinc'] - df['ncfo']) / df['assets']
    else:
        df['accruals'] = pd.NA
        warnings.warn(f"[QUALITY] missing {missing_req} for accruals; filling with NA")

    # 3) Rename SF1 column keys to canonical quality names
    rename_map = {
        'roe':          'roe',
        'roa':          'roa',
        'de':           'debt_to_equity',
        'currentratio': 'current_ratio',
        'grossmargin':  'gross_margin'
    }
    df = df.rename(columns=rename_map)

    # 4) Ensure all target columns exist, filling with NA if missing
    target_cols = ['roe', 'roa', 'debt_to_equity', 'current_ratio', 'gross_margin']
    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA
            warnings.warn(f"[QUALITY] missing '{col}'; filling with NA")

    # 5) Select and index the final columns
    out = df.set_index(['ticker', 'date'])[
        target_cols + ['accruals']
    ]
    return out

# Preview usage (REPL):
if __name__ == "__main__":
    import pandas as pd
    # Small slice of SF1 for testing
    sf1_sample = pd.read_csv(
        'original_files/SHARADAR_SF1.csv',
        usecols=['ticker','date','roe','roa','de','currentratio','grossmargin','netinc','ncfo','assets'],
        nrows=5,
        parse_dates=['date']
    )
    print(build(sf1_sample))
