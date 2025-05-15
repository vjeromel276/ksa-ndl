# models/data.py
#!/usr/bin/env python3
import pandas as pd
from feature_engineering.factors.technicals import build as _build_technicals
from feature_engineering.factors.seasonality import build as _build_seasonality
from models.targets import make_targets

def load_features(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw SEP DataFrame with columns ['ticker','date',...,'close' or 'closeadj'],
    compute and return a DataFrame of technical + seasonality features,
    indexed by ['ticker','date'].
    """
    df = sep.copy()
    # ensure we have a datetime date column
    df['date'] = pd.to_datetime(df['date'])
    # make sure there's a 'close' column for our factor builders
    if 'close' not in df.columns:
        if 'closeadj' in df.columns:
            df['close'] = df['closeadj']
        else:
            raise KeyError("load_features: SEP must have 'close' or 'closeadj' column")
    # build each factor set (they both return a MultiIndex dataframe)
    tech = _build_technicals(df)
    seas = _build_seasonality(df)
    # concatenate side-by-side
    X = pd.concat([tech, seas], axis=1)
    return X

def load_targets(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw SEP DataFrame with columns ['ticker','date',...,'close' or 'closeadj'],
    compute and return forward-return & direction targets at exactly 5 BDays ahead
    (by default), indexed by ['ticker','date'].
    """
    df = sep.copy()
    df['date'] = pd.to_datetime(df['date'])
    # make sure make_targets sees a 'closeadj'
    if 'closeadj' not in df.columns:
        if 'close' in df.columns:
            df['closeadj'] = df['close']
        else:
            raise KeyError("load_targets: SEP must have 'closeadj' or 'close' column")
    y = make_targets(df)
    return y
