# core/schema.py

import pandas as pd

# 1) Define your exact “contract” for the SEP DataFrame
REQUIRED_COLUMNS = {
    "ticker",    # string or categorical
    "date",      # datetime
    "open",      # float
    "high",      # float
    "low",       # float
    "close",     # float
    "volume",    # int or float
}

ALLOWED_DTYPE = {
    "ticker":       (str, "category"),
    "date":         "datetime64[ns]",
    "open":         "float64",
    "high":         "float64",
    "low":          "float64",
    "close":        "float64",
    "volume":       ("int64","float64")
}

def validate_sep_df(df: pd.DataFrame) -> None:
    """
    Raises ValueError if `df` is not exactly in the shape
    our pipeline expects.
    """
    cols = set(df.columns)
    missing = REQUIRED_COLUMNS - cols
    extra   = cols - REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"[INPUT SCHEMA] missing required columns: {missing}")
    if extra:
        # you can choose to error or just warn
        import warnings
        warnings.warn(f"[INPUT SCHEMA] dropping unexpected columns: {extra}", UserWarning)
        df.drop(columns=list(extra), inplace=True)

    # 2) Type‐check each column
    for col, allowed in ALLOWED_DTYPE.items():
        actual = df.dtypes[col]
        if isinstance(allowed, tuple):
            if str(actual) not in allowed:
                raise ValueError(f"[INPUT SCHEMA] column {col!r}: expected one of {allowed}, got {actual}")
        else:
            if str(actual) != allowed:
                raise ValueError(f"[INPUT SCHEMA] column {col!r}: expected {allowed}, got {actual}")

    # 3) Index check: require ticker+date unique
    if not {"ticker","date"}.issubset(df.index.names):
        # if they haven’t yet indexed on ticker+date, that’s fine—
        # we’ll make sure it's unique once we do:
        idx = df.set_index(["ticker","date"], verify_integrity=True)
