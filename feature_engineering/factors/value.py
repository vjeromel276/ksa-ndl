import pandas as pd

def build(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Pull core value ratios from SHARADAR/METRICS, with robust handling:
      - pe, pb, ps, ev_ebitda (filled with NA)
      - div_yield (from trailing dividend yield)

    Args:
      metrics: DataFrame with at least 'ticker', 'date', and one of
               'dividendyieldtrailing' or 'dividendyieldforward'.

    Returns:
      DataFrame indexed by ['ticker','date'] with exactly these columns:
      ['pe','pb','ps','ev_ebitda','div_yield'].
    """
    # 1) Normalize incoming column names
    df = metrics.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) Map the dividend yield (prefer trailing)
    if "dividendyieldtrailing" in df.columns:
        df["div_yield"] = df["dividendyieldtrailing"]
    elif "dividendyieldforward" in df.columns:
        df["div_yield"] = df["dividendyieldforward"]
    else:
        df["div_yield"] = pd.NA

    # 3) Fill missing ratios with NA
    for col in ("pe", "pb", "ps", "ev_ebitda"):
        df[col] = pd.NA

    # 4) Select & index
    out = (
        df
        .set_index(["ticker", "date"])
        [["pe", "pb", "ps", "ev_ebitda", "div_yield"]]
    )
    return out

