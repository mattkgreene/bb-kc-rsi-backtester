import pandas as pd
import numpy as np

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14, smoothing_type: str = "ema") -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    if smoothing_type == "sma":
        roll_up = up.rolling(n).mean()
        roll_down = down.rolling(n).mean()
    elif smoothing_type == "rma":
        roll_up = rma(up, n)
        roll_down = rma(down, n)
    else:  # default ema
        roll_up = up.ewm(alpha=1/n, adjust=False).mean()
        roll_down = down.ewm(alpha=1/n, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bbands(close: pd.Series, n: int = 20, k: float = 2.0, basis_type: str = "sma"):
    if basis_type == "ema":
        mid = ema(close, n)
    else:
        mid = sma(close, n)
    std = close.rolling(n).std()
    up = mid + k * std
    low = mid - k * std
    return mid, up, low

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def keltner(high: pd.Series, low: pd.Series, close: pd.Series,
            ema_len: int = 20, atr_len: int = 14, mult: float = 2.0,
            mid_type: str = "ema"):

    mid = ema(close, ema_len) if mid_type == "ema" else sma(close, ema_len)
    a = atr(high, low, close, atr_len)
    up = mid + mult * a
    lowb = mid - mult * a

    return mid, up, lowb, a

def add_bb_kc_rsi(
    df: pd.DataFrame,
    bb_len: int = 20, bb_std: float = 2.0, bb_basis_type: str = "sma",
    kc_ema_len: int = 20, kc_atr_len: int = 14, kc_mult: float = 2.0, kc_mid_type: str = "ema",
    rsi_len_30m: int = 14, rsi_ma_len: int = 10,
    rsi_smoothing_type: str = "ema", rsi_ma_type: str = "sma"
) -> pd.DataFrame:
  
    bb_mid, bb_up, bb_low = bbands(df['Close'], n=bb_len, k=bb_std, basis_type=bb_basis_type)
    df['bb_mid'], df['bb_up'], df['bb_low'] = bb_mid, bb_up, bb_low

    kc_mid, kc_up, kc_low, kc_atr = keltner(
        df['High'], df['Low'], df['Close'],
        ema_len=kc_ema_len, atr_len=kc_atr_len, mult=kc_mult, mid_type=kc_mid_type
    )
    df['kc_mid'], df['kc_up'], df['kc_low'], df['kc_atr'] = kc_mid, kc_up, kc_low, kc_atr

    df30 = df[['Close']].resample('30T').last().dropna()
    df30['rsi'] = rsi(df30['Close'], n=rsi_len_30m, smoothing_type=rsi_smoothing_type)
    df30['rsi_ma'] = ema(df30['rsi'], rsi_ma_len) if rsi_ma_type == "ema" else sma(df30['rsi'], rsi_ma_len)

    df[['rsi30', 'rsi30_ma']] = df30[['rsi', 'rsi_ma']].reindex(df.index, method='ffill')
    return df
