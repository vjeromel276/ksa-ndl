#!/usr/bin/env python3
import pandas as pd
from feature_engineering.factors.technicals import build as _build_technicals
from feature_engineering.factors.seasonality import build as _build_seasonality
from models.targets import make_targets
from core.schema import validate_min_sep

def _coerce_sep_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast ticker, date, and all numeric SEP columns to the types
    required by our schema: ticker→category, date→datetime64,
    prices & volume→float64.
    """
    # ticker → categorical
    df['ticker'] = df['ticker'].astype('category')
    # date → datetime
    df['date']   = pd.to_datetime(df['date'])
    # numeric cols → float64
    for col in ['open', 'high', 'low', 'close', 'volume',
                'closeadj', 'closeunadj']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    # lastupdated if present → datetime
    if 'lastupdated' in df.columns:
        df['lastupdated'] = pd.to_datetime(df['lastupdated'], errors='coerce')
    return df

def load_features(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw SEP DataFrame, cast to correct dtypes, validate schema,
    then compute technical + seasonality features indexed by ['ticker','date'].
    """
    df = sep.copy()
    # cast to our core types + validate *minimal* SEP contract
    df = _coerce_sep_dtypes(df)
    validate_min_sep(df)

    # make sure there's a 'close' column for our factor builders
    if 'close' not in df.columns:
        if 'closeadj' in df.columns:
            df['close'] = df['closeadj']
        else:
            raise KeyError("load_features: SEP must have 'close' or 'closeadj'")

    tech = _build_technicals(df)
    seas = _build_seasonality(df)
    X = pd.concat([tech, seas], axis=1)
    return X

def load_targets(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw SEP DataFrame, cast to correct dtypes, validate schema,
    then compute forward-return & direction targets at exactly 5 BDays ahead.
    """
    df = sep.copy()
    # cast to our core types + validate *minimal* SEP contract
    df = _coerce_sep_dtypes(df)
    validate_min_sep(df)

    # ensure make_targets sees a 'closeadj'
    if 'closeadj' not in df.columns:
        if 'close' in df.columns:
            df['closeadj'] = df['close']
        else:
            raise KeyError("load_targets: SEP must have 'closeadj' or 'close'")

    y = make_targets(df)
    return y
