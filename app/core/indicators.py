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


# =============================================================================
# Volume Indicators
# =============================================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a cumulative indicator that adds volume on up days
    and subtracts on down days. Divergences between OBV and price
    can signal potential reversals.
    
    Args:
        close: Closing price series.
        volume: Volume series.
    
    Returns:
        Series containing OBV values.
    
    Interpretation:
        - Rising OBV with rising price: Confirms uptrend
        - Rising OBV with falling price: Potential bullish reversal
        - Falling OBV with rising price: Potential bearish reversal
        - Falling OBV with falling price: Confirms downtrend
    
    For Mean Reversion:
        Look for OBV divergence from price as confirmation signal.
    """
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def obv_divergence(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 14
) -> pd.Series:
    """
    Detect OBV divergence from price.
    
    Returns a score indicating divergence:
        Positive: Bullish divergence (price down, OBV up) - good for shorting exit
        Negative: Bearish divergence (price up, OBV down) - good for shorting entry
        Near 0: No significant divergence
    
    Args:
        close: Closing price series
        volume: Volume series
        lookback: Lookback period for comparison
    
    Returns:
        Series with divergence scores (-100 to 100)
    """
    obv_val = obv(close, volume)
    
    # Calculate rate of change for both
    price_roc = (close - close.shift(lookback)) / close.shift(lookback) * 100
    obv_roc = (obv_val - obv_val.shift(lookback)) / obv_val.shift(lookback).abs().replace(0, 1) * 100
    
    # Divergence: difference in direction/magnitude
    divergence = obv_roc - price_roc
    
    # Normalize to -100 to 100 range
    div_std = divergence.rolling(lookback * 2).std()
    normalized = (divergence / div_std.replace(0, np.nan)).clip(-3, 3) * 33.33
    
    return normalized


def volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate volume relative to its moving average.
    
    Ratio > 1.5: High volume (significant move, potential exhaustion)
    Ratio < 0.5: Low volume (weak move, may not sustain)
    Ratio ~1.0: Normal volume
    
    Returns:
        Series of volume/SMA ratios
    """
    vol_sma = volume.rolling(period).mean()
    return volume / vol_sma.replace(0, np.nan)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Money Flow Index (MFI) - Volume-weighted RSI.
    
    MFI incorporates volume into overbought/oversold analysis,
    providing confirmation for RSI signals.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Lookback period (default 14)
    
    Returns:
        Series containing MFI values (0-100 scale)
    
    Interpretation:
        - MFI > 80: Overbought (like RSI > 70)
        - MFI < 20: Oversold (like RSI < 30)
    
    For Short Strategy:
        MFI > 80 combined with RSI > 70 = stronger signal
    """
    # Typical price
    tp = (high + low + close) / 3
    
    # Raw money flow
    raw_mf = tp * volume
    
    # Positive and negative money flow
    pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
    neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
    
    pos_mf = pd.Series(pos_mf, index=close.index)
    neg_mf = pd.Series(neg_mf, index=close.index)
    
    # Sum over period
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    
    # Money flow ratio and index
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi_val = 100 - (100 / (1 + mfr))
    
    return mfi_val


# =============================================================================
# Momentum Confirmation Indicators
# =============================================================================

def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Measures where the close is relative to the high-low range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Lookback for %K
        d_period: Smoothing for %D
    
    Returns:
        Tuple of (%K, %D) Series
    
    Interpretation:
        - %K > 80: Overbought
        - %K < 20: Oversold
        - %K crossing %D: Momentum shift
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    
    return k, d


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.
    
    Similar to Stochastic but inverted. Measures overbought/oversold.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period
    
    Returns:
        Series with Williams %R values (-100 to 0)
    
    Interpretation:
        - %R > -20: Overbought (good for short entry)
        - %R < -80: Oversold (good for short exit)
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    
    return wr


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI measures deviation from statistical mean, useful for
    identifying cyclical trends and extremes.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period
    
    Returns:
        Series with CCI values (unbounded, typically -200 to +200)
    
    Interpretation:
        - CCI > 100: Overbought
        - CCI < -100: Oversold
        - Extreme CCI (>200 or <-200): Strong overbought/oversold
    """
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def momentum_composite(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    rsi_val: pd.Series,
    rsi_weight: float = 0.4,
    stoch_weight: float = 0.3,
    cci_weight: float = 0.3,
    stoch_period: int = 14,
    cci_period: int = 20
) -> pd.Series:
    """
    Create composite momentum score from multiple indicators.
    
    Combines RSI, Stochastic, and CCI into a single score.
    Useful for confirming overbought/oversold conditions.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        rsi_val: Pre-calculated RSI values
        rsi_weight: Weight for RSI (default 0.4)
        stoch_weight: Weight for Stochastic (default 0.3)
        cci_weight: Weight for CCI (default 0.3)
    
    Returns:
        Series with composite score (0-100 scale)
        Higher values = more overbought
    """
    # Get individual indicators
    stoch_k, _ = stochastic(high, low, close, stoch_period)
    cci_val = cci(high, low, close, cci_period)
    
    # Normalize CCI to 0-100 scale
    cci_normalized = ((cci_val + 200) / 400 * 100).clip(0, 100)
    
    # Calculate weighted composite
    composite = (
        rsi_weight * rsi_val +
        stoch_weight * stoch_k +
        cci_weight * cci_normalized
    )
    
    return composite


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


def add_confirmation_indicators(
    df: pd.DataFrame,
    include_volume: bool = True,
    include_momentum: bool = True,
    include_divergence: bool = True,
    obv_div_lookback: int = 14,
    mfi_period: int = 14,
    stoch_period: int = 14,
    cci_period: int = 20,
    williams_period: int = 14
) -> pd.DataFrame:
    """
    Add confirmation indicators for more robust signal generation.
    
    These additional indicators help confirm overbought/oversold conditions
    and reduce false signals that hurt long-term performance.
    
    Args:
        df: OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        include_volume: Add volume-based indicators (OBV, MFI)
        include_momentum: Add momentum indicators (Stochastic, Williams %R, CCI)
        include_divergence: Add divergence detection
        Various period parameters for customization
    
    Returns:
        DataFrame with added indicator columns:
            - obv: On-Balance Volume
            - obv_divergence: OBV price divergence score
            - vol_ratio: Volume/SMA ratio
            - mfi: Money Flow Index
            - stoch_k, stoch_d: Stochastic oscillator
            - williams_r: Williams %R
            - cci: Commodity Channel Index
            - momentum_score: Composite momentum score
    """
    # Check for volume column
    has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    
    if include_volume and has_volume:
        # On-Balance Volume
        df['obv'] = obv(df['Close'], df['Volume'])
        
        # Volume ratio
        df['vol_ratio'] = volume_sma_ratio(df['Volume'])
        
        # Money Flow Index
        df['mfi'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'], mfi_period)
        
        if include_divergence:
            # OBV Divergence
            df['obv_divergence'] = obv_divergence(df['Close'], df['Volume'], obv_div_lookback)
    
    if include_momentum:
        # Stochastic
        df['stoch_k'], df['stoch_d'] = stochastic(
            df['High'], df['Low'], df['Close'], stoch_period
        )
        
        # Williams %R
        df['williams_r'] = williams_r(df['High'], df['Low'], df['Close'], williams_period)
        
        # CCI
        df['cci'] = cci(df['High'], df['Low'], df['Close'], cci_period)
        
        # Composite momentum score (if RSI available)
        if 'rsi30' in df.columns:
            df['momentum_score'] = momentum_composite(
                df['High'], df['Low'], df['Close'],
                df['rsi30'],
                stoch_period=stoch_period,
                cci_period=cci_period
            )
    
    return df


def add_all_indicators(
    df: pd.DataFrame,
    # BB/KC/RSI params
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
    rsi_ma_type: str = "sma",
    # Additional indicator params
    include_confirmation: bool = True,
    include_volume: bool = True,
    include_momentum: bool = True,
    include_regime: bool = True,
    adx_period: int = 14,
    vol_lookback: int = 100
) -> pd.DataFrame:
    """
    Add all indicators to a DataFrame in one call.
    
    This is a comprehensive function that adds:
    - Core BB/KC/RSI indicators
    - Volume confirmation indicators
    - Momentum confirmation indicators
    - Regime detection indicators
    
    Args:
        df: OHLCV DataFrame
        Various parameters for each indicator type
        include_confirmation: Add volume/momentum indicators
        include_regime: Add regime detection indicators
    
    Returns:
        DataFrame with all indicators added
    """
    # Add core indicators
    df = add_bb_kc_rsi(
        df,
        bb_len=bb_len, bb_std=bb_std, bb_basis_type=bb_basis_type,
        kc_ema_len=kc_ema_len, kc_atr_len=kc_atr_len, kc_mult=kc_mult, kc_mid_type=kc_mid_type,
        rsi_len_30m=rsi_len_30m, rsi_ma_len=rsi_ma_len,
        rsi_smoothing_type=rsi_smoothing_type, rsi_ma_type=rsi_ma_type
    )
    
    # Add confirmation indicators
    if include_confirmation:
        df = add_confirmation_indicators(
            df,
            include_volume=include_volume,
            include_momentum=include_momentum
        )
    
    # Add regime indicators
    if include_regime:
        try:
            from core.regime import add_regime_indicators
            df = add_regime_indicators(df, adx_period=adx_period, vol_lookback=vol_lookback)
        except ImportError:
            pass  # Regime module not available
    
    return df
