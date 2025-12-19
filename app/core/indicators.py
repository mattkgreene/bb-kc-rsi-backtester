"""
Technical indicators module for the BB+KC+RSI strategy.

This module provides implementations of common technical indicators used
in the backtesting strategy:
- Moving Averages: SMA, EMA, RMA (Wilder's smoothing)
- Oscillators: RSI with configurable smoothing
- Volatility Bands: Bollinger Bands, Keltner Channels
- Volatility: Average True Range (ATR)

All functions are designed to work with pandas Series/DataFrames and
return results aligned with the input index.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def sma(s: pd.Series, n: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        s: Input price series.
        n: Lookback period (number of bars).
    
    Returns:
        Series containing SMA values. First n-1 values will be NaN.
    
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14])
        >>> sma(prices, 3)
        0     NaN
        1     NaN
        2    11.0
        3    12.0
        4    13.0
    """
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Uses span-based EMA calculation where alpha = 2 / (n + 1).
    
    Args:
        s: Input price series.
        n: Span period for EMA calculation.
    
    Returns:
        Series containing EMA values. Starts calculating from first value.
    
    Note:
        EMA gives more weight to recent prices compared to SMA.
        The decay factor alpha = 2 / (span + 1).
    
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14])
        >>> ema(prices, 3)  # Returns EMA with span=3
    """
    return s.ewm(span=n, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Wilder's Smoothing (RMA - Running Moving Average).
    
    This is the smoothing method originally used by J. Welles Wilder
    for indicators like RSI and ATR. It uses alpha = 1/period instead
    of the standard EMA's alpha = 2/(period+1).
    
    Args:
        series: Input price series.
        period: Smoothing period.
    
    Returns:
        Series containing RMA values.
    
    Note:
        RMA is more reactive than standard EMA for the same period.
        It's commonly used in RSI calculations on platforms like TradingView.
    """
    return series.ewm(alpha=1/period, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14, smoothing_type: str = "ema") -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. Values range from 0 to 100.
    
    Args:
        close: Closing price series.
        n: Lookback period for RSI calculation. Default 14.
        smoothing_type: Method for smoothing gains/losses.
            - 'ema': Exponential moving average (default)
            - 'sma': Simple moving average
            - 'rma': Wilder's smoothing (TradingView style)
    
    Returns:
        Series containing RSI values (0-100 scale).
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
    
    Interpretation:
        - RSI > 70: Typically considered overbought
        - RSI < 30: Typically considered oversold
        - RSI = 50: Neutral momentum
    
    Example:
        >>> rsi_values = rsi(df['Close'], n=14, smoothing_type='ema')
        >>> overbought = rsi_values > 70
    """
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


def bbands(close: pd.Series, n: int = 20, k: float = 2.0, basis_type: str = "sma") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (moving average) with upper
    and lower bands at k standard deviations above and below.
    
    Args:
        close: Closing price series.
        n: Period for moving average and standard deviation. Default 20.
        k: Number of standard deviations for band width. Default 2.0.
        basis_type: Type of moving average for middle band.
            - 'sma': Simple Moving Average (default, traditional)
            - 'ema': Exponential Moving Average
    
    Returns:
        Tuple of (mid, upper, lower) Series:
            - mid: Middle band (moving average)
            - upper: Upper band (mid + k * std)
            - lower: Lower band (mid - k * std)
    
    Interpretation:
        - Price touching upper band: Potentially overbought
        - Price touching lower band: Potentially oversold
        - Band width: Indicates volatility (narrow = low vol, wide = high vol)
    
    Example:
        >>> mid, upper, lower = bbands(df['Close'], n=20, k=2.0)
        >>> df['bb_mid'], df['bb_up'], df['bb_low'] = mid, upper, lower
    """
    if basis_type == "ema":
        mid = ema(close, n)
    else:
        mid = sma(close, n)
    std = close.rolling(n).std()
    up = mid + k * std
    low = mid - k * std
    return mid, up, low


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    ATR measures market volatility by decomposing the entire range of
    an asset price for a period. True Range accounts for gaps.
    
    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        n: Smoothing period for ATR. Default 14.
    
    Returns:
        Series containing ATR values.
    
    Formula:
        True Range = max(
            High - Low,
            abs(High - Previous Close),
            abs(Low - Previous Close)
        )
        ATR = SMA(True Range, n)
    
    Note:
        ATR is commonly used for:
        - Position sizing (volatility-based)
        - Stop loss placement (e.g., 2 * ATR from entry)
        - Identifying low/high volatility regimes
    
    Example:
        >>> atr_values = atr(df['High'], df['Low'], df['Close'], n=14)
        >>> stop_distance = 2 * atr_values.iloc[-1]
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_len: int = 20,
    atr_len: int = 14,
    mult: float = 2.0,
    mid_type: str = "ema"
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channel.
    
    Keltner Channels are volatility-based bands set above and below
    a moving average, using ATR for band width instead of standard deviation.
    
    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        ema_len: Period for middle line moving average. Default 20.
        atr_len: Period for ATR calculation. Default 14.
        mult: ATR multiplier for band width. Default 2.0.
        mid_type: Type of moving average for middle line.
            - 'ema': Exponential Moving Average (default)
            - 'sma': Simple Moving Average
    
    Returns:
        Tuple of (mid, upper, lower, atr_values):
            - mid: Middle line (moving average)
            - upper: Upper channel (mid + mult * ATR)
            - lower: Lower channel (mid - mult * ATR)
            - atr_values: ATR series used for channel width
    
    Comparison to Bollinger Bands:
        - KC uses ATR (accounts for gaps), BB uses standard deviation
        - KC tends to be smoother and less reactive to sudden moves
        - When BB is outside KC: "squeeze" condition (low volatility)
    
    Example:
        >>> mid, upper, lower, atr_vals = keltner(
        ...     df['High'], df['Low'], df['Close'],
        ...     ema_len=20, atr_len=14, mult=2.0
        ... )
    """
    mid = ema(close, ema_len) if mid_type == "ema" else sma(close, ema_len)
    a = atr(high, low, close, atr_len)
    up = mid + mult * a
    lowb = mid - mult * a

    return mid, up, lowb, a


def add_bb_kc_rsi(
    df: pd.DataFrame,
    bb_len: int = 20,
    bb_std: float = 2.0,
    bb_basis_type: str = "sma",
    kc_ema_len: int = 20,
    kc_atr_len: int = 14,
    kc_mult: float = 2.0,
    kc_mid_type: str = "ema",
    rsi_len_30m: int = 14,
    rsi_ma_len: int = 10,
    rsi_smoothing_type: str = "ema",
    rsi_ma_type: str = "sma"
) -> pd.DataFrame:
    """
    Add all strategy indicators to a DataFrame.
    
    This function calculates and adds Bollinger Bands, Keltner Channels,
    and RSI (with 30-minute resampling) to the input DataFrame.
    
    Args:
        df: OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            and a DatetimeIndex.
        bb_len: Bollinger Bands period. Default 20.
        bb_std: Bollinger Bands standard deviation multiplier. Default 2.0.
        bb_basis_type: BB middle line type ('sma' or 'ema'). Default 'sma'.
        kc_ema_len: Keltner Channel middle line period. Default 20.
        kc_atr_len: Keltner Channel ATR period. Default 14.
        kc_mult: Keltner Channel ATR multiplier. Default 2.0.
        kc_mid_type: KC middle line type ('ema' or 'sma'). Default 'ema'.
        rsi_len_30m: RSI period (on 30m resampled data). Default 14.
        rsi_ma_len: RSI moving average period. Default 10.
        rsi_smoothing_type: RSI smoothing ('ema', 'sma', 'rma'). Default 'ema'.
        rsi_ma_type: RSI MA type ('ema' or 'sma'). Default 'sma'.
    
    Returns:
        DataFrame with added columns:
            - bb_mid, bb_up, bb_low: Bollinger Band values
            - kc_mid, kc_up, kc_low, kc_atr: Keltner Channel values
            - rsi30, rsi30_ma: RSI values (forward-filled from 30m resample)
    
    Note:
        RSI is calculated on 30-minute resampled data regardless of the
        input timeframe, then forward-filled back to the original index.
        This provides a smoother RSI signal on shorter timeframes.
    
    Example:
        >>> df = fetch_ohlcv('bitstamp', 'BTC/USD', '30m')
        >>> df = add_bb_kc_rsi(df, bb_len=20, kc_mult=2.0, rsi_len_30m=14)
        >>> entry_signal = (df['Close'] >= df['kc_up']) & (df['rsi30'] >= 70)
    """
    # Bollinger Bands
    bb_mid, bb_up, bb_low = bbands(df['Close'], n=bb_len, k=bb_std, basis_type=bb_basis_type)
    df['bb_mid'], df['bb_up'], df['bb_low'] = bb_mid, bb_up, bb_low

    # Keltner Channel
    kc_mid, kc_up, kc_low, kc_atr = keltner(
        df['High'], df['Low'], df['Close'],
        ema_len=kc_ema_len, atr_len=kc_atr_len, mult=kc_mult, mid_type=kc_mid_type
    )
    df['kc_mid'], df['kc_up'], df['kc_low'], df['kc_atr'] = kc_mid, kc_up, kc_low, kc_atr

    # RSI on 30-minute resampled data
    # This provides a smoother signal when using shorter timeframes
    df30 = df[['Close']].resample('30T').last().dropna()
    df30['rsi'] = rsi(df30['Close'], n=rsi_len_30m, smoothing_type=rsi_smoothing_type)
    df30['rsi_ma'] = ema(df30['rsi'], rsi_ma_len) if rsi_ma_type == "ema" else sma(df30['rsi'], rsi_ma_len)

    # Forward-fill RSI values back to original timeframe
    df[['rsi30', 'rsi30_ma']] = df30[['rsi', 'rsi_ma']].reindex(df.index, method='ffill')
    return df
