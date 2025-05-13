# feature_engineering/factors/technicals.py
import pandas as pd
import numpy as np

def build(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Compute classic technical indicators:
      - ma20 / ma50 as object dtype (first 19 days == np.nan identity)
      - ma_cross (1 if ma20 > ma50, else 0)
      - rsi14, macd, macd_signal, boll_mid, boll_upper, boll_lower
    """
    sep = sep.sort_values(['ticker','date'])
    # Create result DataFrame with MultiIndex
    idx = sep.set_index(['ticker','date']).index
    out = pd.DataFrame(index=idx)

    # Numeric rolling MAs
    grouped = sep.groupby('ticker')['close']
    ma20_num = grouped.transform(lambda x: x.rolling(20).mean()).values
    ma50_num = grouped.transform(lambda x: x.rolling(50).mean()).values

    # Build object-dtype lists: np.nan for mask, python float otherwise
    ma20_obj = [np.nan if pd.isna(v) else float(v) for v in ma20_num]
    ma50_obj = [np.nan if pd.isna(v) else float(v) for v in ma50_num]

    # Assign MAs as object series (so 'is np.nan' holds true)
    out['ma20'] = pd.Series(ma20_obj, index=idx, dtype=object)
    out['ma50'] = pd.Series(ma50_obj, index=idx, dtype=object)

    # Cross (use numeric arrays)
    cross = (pd.Series(ma20_num) > pd.Series(ma50_num)).astype(int).values
    out['ma_cross'] = cross

    # --------------- RSI14 ---------------
    def compute_rsi(series, period=14):
        delta  = series.diff()
        up     = delta.clip(lower=0)
        down   = -delta.clip(upper=0)
        ma_up   = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs     = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    # ------ RSI14 (object dtype) so `is np.nan` holds on missing slots ------
    rsi_num = sep.groupby('ticker')['close'] \
                 .transform(lambda x: compute_rsi(x, 14)) \
                 .values
    # Build Python list with exact np.nan and floats
    rsi_obj = [np.nan if pd.isna(v) else float(v) for v in rsi_num]
    out['rsi14'] = pd.Series(rsi_obj, index=idx, dtype=object)

    # --------------- MACD & Signal ---------------
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    macd_line = grouped.transform(lambda x: ema(x, 12) - ema(x, 26))
    out['macd']        = macd_line.values
    out['macd_signal'] = grouped.transform(lambda x: ema(ema(x, 12) - ema(x, 26), 9)).values

    # --------------- Bollinger Bands ---------------
    # boll_mid_num = grouped.transform(lambda x: x.rolling(20).mean())
    # std20_num    = grouped.transform(lambda x: x.rolling(20).std())

    # out['boll_mid']   = boll_mid_num.values
    # out['boll_upper'] = (boll_mid_num + 2 * std20_num).values
    # out['boll_lower'] = (boll_mid_num - 2 * std20_num).values
    # --------------- Bollinger Bands (use ddof=0) ---------------
    boll_mid_num = grouped.transform(lambda x: x.rolling(20).mean())
    std20_num    = grouped.transform(lambda x: x.rolling(20).std(ddof=0))

    out['boll_mid']   = boll_mid_num.values
    out['boll_upper'] = (boll_mid_num + 2 * std20_num).values
    out['boll_lower'] = (boll_mid_num - 2 * std20_num).values


    return out
