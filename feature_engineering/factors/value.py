import pandas as pd

def build(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Pull core value ratios from SHARADAR/METRICS:
      - pe, pb, ps, ev_ebitda, dividend_yield

    Args:
      metrics: DataFrame with columns 
        ['ticker','date','pe','pb','ps','ev_ebitda','div_yield']

    Returns:
      DataFrame indexed by ['ticker','date'] with those columns.
    """
    cols = {
        "pe":           "pe",
        "pb":           "pb",
        "ps":           "ps",
        "ev_ebitda":    "ev_ebitda",
        "dividend_yield":"div_yield"
    }
    df = (
        metrics
        .rename(columns=cols)
        .set_index(["ticker","date"])
        [list(cols.values())]
    )
    return df
