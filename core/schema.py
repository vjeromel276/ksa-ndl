# core/schema.py

import pandas as pd
import warnings

# ── FULL SEP SCHEMA (all 10 Sharadar columns) ─────────────────────────────────

REQUIRED_FULL_COLUMNS = {
    "ticker",    # string or categorical
    "date",      # datetime
    "open",      # float64
    "high",      # float64
    "low",       # float64
    "close",     # float64
    "volume",    # int64 or float64
    "closeadj",  # float64
    "closeunadj",# float64
    "lastupdated"# datetime64
}

ALLOWED_FULL_DTYPE = {
    "ticker":       (str, "category"),
    "date":         "datetime64[ns]",
    "open":         "float64",
    "high":         "float64",
    "low":          "float64",
    "close":        "float64",
    "volume":       ("int64", "float64"),
    "closeadj":     "float64",
    "closeunadj":   "float64",
    "lastupdated":  "datetime64[ns]",
}

def validate_full_sep(df: pd.DataFrame) -> None:
    """
    Validate presence & dtypes of the *full* Sharadar SEP schema (10 cols).
    Drops any extra columns, warns about them.
    """
    cols = set(df.columns)
    missing = REQUIRED_FULL_COLUMNS - cols
    extra   = cols - REQUIRED_FULL_COLUMNS
    if missing:
        raise ValueError(f"[FULL SEP SCHEMA] missing required columns: {missing}")
    if extra:
        warnings.warn(f"[FULL SEP SCHEMA] dropping unexpected columns: {extra}", UserWarning)
        df.drop(columns=list(extra), inplace=True)

    for col, allowed in ALLOWED_FULL_DTYPE.items():
        actual = df.dtypes[col]
        if isinstance(allowed, tuple):
            if str(actual) not in allowed:
                raise ValueError(f"[FULL SEP SCHEMA] column {col!r}: expected one of {allowed}, got {actual}")
        else:
            if str(actual) != allowed:
                raise ValueError(f"[FULL SEP SCHEMA] column {col!r}: expected {allowed}, got {actual}")


# ── MINIMAL SEP SCHEMA (seven core cols for feature/target loaders) ─────────────

REQUIRED_MIN_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

ALLOWED_MIN_DTYPE = {
    "ticker": (str, "category"),
    "date":   "datetime64[ns]",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": ("int64", "float64"),
}

def validate_min_sep(df: pd.DataFrame) -> None:
    """
    Validate presence & dtypes of the *minimal* SEP schema (7 cols).
    Drops any extra columns, warns about them.
    """
    cols = set(df.columns)
    missing = REQUIRED_MIN_COLUMNS - cols
    extra   = cols - REQUIRED_MIN_COLUMNS
    if missing:
        raise ValueError(f"[MIN SEP SCHEMA] missing required columns: {missing}")
    if extra:
        warnings.warn(f"[MIN SEP SCHEMA] dropping unexpected columns: {extra}", UserWarning)
        df.drop(columns=list(extra), inplace=True)

    for col, allowed in ALLOWED_MIN_DTYPE.items():
        actual = df.dtypes[col]
        if isinstance(allowed, tuple):
            if str(actual) not in allowed:
                raise ValueError(f"[MIN SEP SCHEMA] column {col!r}: expected one of {allowed}, got {actual}")
        else:
            if str(actual) != allowed:
                raise ValueError(f"[MIN SEP SCHEMA] column {col!r}: expected {allowed}, got {actual}")


# ── VOLUME‐ONLY SCHEMA (for compute_per_ticker) ───────────────────────────────────

VOLUME_REQUIRED_COLUMNS = {"ticker", "date", "volume"}
VOLUME_ALLOWED_DTYPE    = {
    "ticker": (str, "category"),
    "date":   "datetime64[ns]",
    "volume": ("int64", "float64"),
}

def validate_volume_df(df: pd.DataFrame) -> None:
    """
    Validate that df has exactly (ticker, date, volume) with correct dtypes.
    Drops any extra columns, warns about them.
    """
    cols = set(df.columns)
    missing = VOLUME_REQUIRED_COLUMNS - cols
    extra   = cols - VOLUME_REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"[VOLUME SCHEMA] missing required columns: {missing}")
    if extra:
        warnings.warn(f"[VOLUME SCHEMA] dropping unexpected columns: {extra}", UserWarning)
        df.drop(columns=list(extra), inplace=True)

    for col, allowed in VOLUME_ALLOWED_DTYPE.items():
        actual = df.dtypes[col]
        if isinstance(allowed, tuple):
            if str(actual) not in allowed:
                raise ValueError(f"[VOLUME SCHEMA] column {col!r}: expected one of {allowed}, got {actual}")
        else:
            if str(actual) != allowed:
                raise ValueError(f"[VOLUME SCHEMA] column {col!r}: expected {allowed}, got {actual}")
