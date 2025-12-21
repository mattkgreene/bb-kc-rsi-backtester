"""
Market Regime Detection Module.

This module provides tools for identifying market regimes and conditions
that help determine when strategies are likely to succeed or fail.

Key Features:
- Trend strength detection (ADX-based)
- Volatility regime classification (low/normal/high)
- Market state identification (trending/ranging/volatile)
- Regime change detection for adaptive strategies

Why This Matters for Long-Term Profitability:
Mean reversion strategies fail in strong trends. By detecting regime,
we can avoid taking trades in unfavorable conditions.
"""

from __future__ import annotations

from typing import Tuple, Dict, Literal, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from core.indicators import ema, sma, atr


# =============================================================================
# Trend Strength Indicators
# =============================================================================

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) for trend strength.
    
    ADX measures the strength of a trend regardless of direction.
    Values above 25 indicate a strong trend (bad for mean reversion).
    Values below 20 indicate a ranging market (good for mean reversion).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period (default 14)
    
    Returns:
        Tuple of (adx, plus_di, minus_di):
            - adx: Average Directional Index (0-100 scale)
            - plus_di: Plus Directional Indicator
            - minus_di: Minus Directional Indicator
    
    Interpretation:
        - ADX < 20: Weak/No trend (GOOD for mean reversion)
        - ADX 20-25: Emerging trend (CAUTION)
        - ADX 25-50: Strong trend (AVOID mean reversion)
        - ADX > 50: Very strong trend (DEFINITELY avoid)
    """
    # Calculate directional movement
    up_move = high.diff()
    down_move = -low.diff()
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # Smoothed values (Wilder's smoothing)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_di_smooth / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_di_smooth / atr_val.replace(0, np.nan)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx_val, plus_di, minus_di


def trend_direction(close: pd.Series, fast_period: int = 10, slow_period: int = 30) -> pd.Series:
    """
    Determine trend direction using EMA crossover.
    
    Returns:
        Series with values:
            1: Uptrend (fast EMA > slow EMA)
            -1: Downtrend (fast EMA < slow EMA)
            0: Neutral (EMAs close together)
    """
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)
    
    # Calculate distance as percentage
    distance_pct = (fast_ema - slow_ema) / slow_ema * 100
    
    # Threshold for "neutral" zone
    threshold = 0.1  # 0.1% difference
    
    direction = pd.Series(0, index=close.index)
    direction[distance_pct > threshold] = 1   # Uptrend
    direction[distance_pct < -threshold] = -1  # Downtrend
    
    return direction


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD helps identify trend direction and momentum changes.
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# =============================================================================
# Volatility Regime Detection
# =============================================================================

def volatility_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    lookback: int = 100
) -> pd.Series:
    """
    Classify volatility regime based on ATR percentile.
    
    Compares current ATR to historical ATR to determine if we're in
    a low, normal, or high volatility environment.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        atr_period: Period for ATR calculation
        lookback: Lookback for percentile calculation
    
    Returns:
        Series with values:
            0: Low volatility (ATR < 25th percentile)
            1: Normal volatility (25th-75th percentile)
            2: High volatility (> 75th percentile)
    
    Interpretation for Mean Reversion:
        - Low vol: Signals may be weak, tight stops needed
        - Normal vol: Ideal conditions
        - High vol: Wider stops needed, more false signals
    """
    atr_val = atr(high, low, close, atr_period)
    
    # Calculate rolling percentile rank
    def percentile_rank(x):
        if len(x) < lookback:
            return 50  # Default to middle
        return (x.iloc[-1] >= x).sum() / len(x) * 100
    
    pct_rank = atr_val.rolling(lookback).apply(percentile_rank, raw=False)
    
    # Classify regime
    regime = pd.Series(1, index=close.index)  # Default normal
    regime[pct_rank <= 25] = 0  # Low volatility
    regime[pct_rank >= 75] = 2  # High volatility
    
    return regime


def volatility_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int = 5,
    long_period: int = 20
) -> pd.Series:
    """
    Calculate volatility ratio (short-term vs long-term ATR).
    
    Ratio > 1.5: Volatility expansion (potential breakout)
    Ratio < 0.7: Volatility contraction (potential squeeze)
    Ratio ~1.0: Normal volatility
    
    Returns:
        Series of volatility ratios
    """
    short_atr = atr(high, low, close, short_period)
    long_atr = atr(high, low, close, long_period)
    
    return short_atr / long_atr.replace(0, np.nan)


def bollinger_bandwidth(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.Series:
    """
    Calculate Bollinger Bandwidth (measure of volatility).
    
    Bandwidth = (Upper Band - Lower Band) / Middle Band
    
    Low bandwidth indicates low volatility and potential squeeze.
    High bandwidth indicates high volatility.
    """
    mid = sma(close, period)
    std = close.rolling(period).std()
    
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    
    return bandwidth


def bandwidth_percentile(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 126  # ~6 months of trading days
) -> pd.Series:
    """
    Calculate Bollinger Bandwidth percentile over lookback period.
    
    Returns a value 0-100 indicating where current bandwidth ranks
    in the historical distribution.
    
    Low percentile (<20): Squeeze - potential breakout incoming
    High percentile (>80): Extended - may revert to mean
    """
    bw = bollinger_bandwidth(close, period, std_dev)
    
    def pct_rank(x):
        if pd.isna(x.iloc[-1]) or len(x.dropna()) < 2:
            return np.nan
        return (x.iloc[-1] >= x.dropna()).sum() / len(x.dropna()) * 100
    
    return bw.rolling(lookback, min_periods=20).apply(pct_rank, raw=False)


# =============================================================================
# Combined Regime Classification
# =============================================================================

MarketRegime = Literal["trending_up", "trending_down", "ranging", "volatile", "squeeze"]


@dataclass
class RegimeState:
    """Current market regime state with all indicators."""
    regime: MarketRegime
    adx_value: float
    trend_direction: int  # 1=up, -1=down, 0=neutral
    volatility_state: int  # 0=low, 1=normal, 2=high
    bandwidth_percentile: float
    mean_reversion_score: float  # 0-100, higher = better for MR
    
    def is_good_for_mean_reversion(self) -> bool:
        """Returns True if conditions favor mean reversion strategy."""
        return self.mean_reversion_score >= 50


def classify_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    adx_period: int = 14,
    vol_lookback: int = 100
) -> pd.DataFrame:
    """
    Classify market regime at each bar.
    
    Combines multiple indicators to determine the overall market state
    and whether mean reversion strategies are appropriate.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        adx_period: Period for ADX calculation
        vol_lookback: Lookback for volatility percentile
    
    Returns:
        DataFrame with columns:
            - adx: ADX value
            - trend_dir: Trend direction (1/-1/0)
            - vol_regime: Volatility regime (0/1/2)
            - bw_pct: Bandwidth percentile
            - regime: Market regime string
            - mr_score: Mean reversion suitability score (0-100)
    """
    # Calculate indicators
    adx_val, plus_di, minus_di = adx(high, low, close, adx_period)
    trend_dir = trend_direction(close)
    vol_reg = volatility_regime(high, low, close, atr_period=adx_period, lookback=vol_lookback)
    bw_pct = bandwidth_percentile(close)
    
    # Build result DataFrame
    result = pd.DataFrame(index=close.index)
    result['adx'] = adx_val
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    result['trend_dir'] = trend_dir
    result['vol_regime'] = vol_reg
    result['bw_pct'] = bw_pct
    
    # Classify regime
    def classify_row(row):
        adx = row['adx']
        vol = row['vol_regime']
        bw = row['bw_pct']
        trend = row['trend_dir']
        
        # Handle NaN
        if pd.isna(adx):
            return 'ranging', 50
        
        # Check for squeeze (low bandwidth)
        if not pd.isna(bw) and bw < 10:
            return 'squeeze', 30  # Caution during squeeze
        
        # Check for high volatility
        if vol == 2 or (not pd.isna(bw) and bw > 90):
            return 'volatile', 40  # Risky for MR
        
        # Check for strong trend
        if adx > 30:
            if trend > 0:
                return 'trending_up', 20  # Bad for shorting
            elif trend < 0:
                return 'trending_down', 70  # Good for shorting
            else:
                return 'ranging', 50
        
        # Moderate trend
        if adx > 20:
            if trend > 0:
                return 'trending_up', 40
            elif trend < 0:
                return 'trending_down', 60
            else:
                return 'ranging', 60
        
        # No trend - ideal for mean reversion
        return 'ranging', 80
    
    regimes = result.apply(classify_row, axis=1)
    result['regime'] = regimes.apply(lambda x: x[0])
    result['mr_score'] = regimes.apply(lambda x: x[1])
    
    return result


def add_regime_indicators(
    df: pd.DataFrame,
    adx_period: int = 14,
    vol_lookback: int = 100,
    prefix: str = ""
) -> pd.DataFrame:
    """
    Add regime detection indicators to an OHLCV DataFrame.
    
    This is the main function to call before backtesting to add
    regime filtering capabilities.
    
    Args:
        df: OHLCV DataFrame with High, Low, Close columns
        adx_period: Period for ADX
        vol_lookback: Lookback for volatility percentile
        prefix: Prefix for new column names
    
    Returns:
        DataFrame with added columns:
            - {prefix}adx: ADX value
            - {prefix}trend_dir: Trend direction
            - {prefix}vol_regime: Volatility regime
            - {prefix}regime: Market regime classification
            - {prefix}mr_score: Mean reversion suitability (0-100)
    """
    regime_df = classify_regime(
        df['High'], df['Low'], df['Close'],
        adx_period=adx_period,
        vol_lookback=vol_lookback
    )
    
    # Add columns with prefix
    for col in regime_df.columns:
        df[f"{prefix}{col}"] = regime_df[col]
    
    return df


# =============================================================================
# Regime Filter Functions for Strategy
# =============================================================================

def create_regime_filter(
    regime_df: pd.DataFrame,
    min_mr_score: float = 50,
    allowed_regimes: Optional[list] = None,
    max_adx: Optional[float] = None,
    block_strong_uptrend: bool = True
) -> pd.Series:
    """
    Create a boolean filter for trade entry based on regime.
    
    Args:
        regime_df: DataFrame with regime indicators (from classify_regime)
        min_mr_score: Minimum mean reversion score (0-100)
        allowed_regimes: List of allowed regime strings (None = all)
        max_adx: Maximum ADX value (None = no limit)
        block_strong_uptrend: Block trades when ADX high and trending up
    
    Returns:
        Boolean Series - True where trades are allowed
    """
    # Start with all True
    filter_mask = pd.Series(True, index=regime_df.index)
    
    # Apply mean reversion score filter
    if 'mr_score' in regime_df.columns:
        filter_mask &= regime_df['mr_score'] >= min_mr_score
    
    # Apply regime filter
    if allowed_regimes is not None and 'regime' in regime_df.columns:
        filter_mask &= regime_df['regime'].isin(allowed_regimes)
    
    # Apply ADX filter
    if max_adx is not None and 'adx' in regime_df.columns:
        filter_mask &= (regime_df['adx'] <= max_adx) | regime_df['adx'].isna()
    
    # Block strong uptrends (bad for short-only strategy)
    if block_strong_uptrend and 'adx' in regime_df.columns and 'trend_dir' in regime_df.columns:
        strong_uptrend = (regime_df['adx'] > 25) & (regime_df['trend_dir'] > 0)
        filter_mask &= ~strong_uptrend
    
    return filter_mask


# =============================================================================
# Helper for Regime Analysis
# =============================================================================

def analyze_regime_distribution(regime_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of regimes in the data.
    
    Useful for understanding what market conditions dominate
    and whether there's enough variety for robust testing.
    
    Returns:
        Dictionary with regime statistics
    """
    if 'regime' not in regime_df.columns:
        return {"error": "No regime column found"}
    
    total = len(regime_df.dropna(subset=['regime']))
    
    regime_counts = regime_df['regime'].value_counts()
    regime_pcts = regime_df['regime'].value_counts(normalize=True) * 100
    
    # Average MR score by regime
    mr_by_regime = regime_df.groupby('regime')['mr_score'].mean() if 'mr_score' in regime_df.columns else {}
    
    # Time spent in each regime
    results = {
        "total_bars": total,
        "regime_distribution": regime_pcts.to_dict(),
        "regime_counts": regime_counts.to_dict(),
    }
    
    if isinstance(mr_by_regime, pd.Series):
        results["avg_mr_score_by_regime"] = mr_by_regime.to_dict()
    
    # ADX statistics
    if 'adx' in regime_df.columns:
        results["adx_stats"] = {
            "mean": float(regime_df['adx'].mean()),
            "median": float(regime_df['adx'].median()),
            "pct_above_25": float((regime_df['adx'] > 25).mean() * 100),
            "pct_below_20": float((regime_df['adx'] < 20).mean() * 100),
        }
    
    return results
