"""
Multi-Timeframe (MTF) Analysis Module.

This module provides tools for confirming signals across multiple timeframes,
which is essential for creating robust strategies that work over time.

Why Multi-Timeframe Analysis Matters:
- Reduces false signals by requiring alignment across timeframes
- Filters out noise from lower timeframes
- Confirms trend on higher timeframes before taking mean reversion trades
- Improves win rate and reduces overfitting

Key Concepts:
- Higher Timeframe (HTF): Used for trend/regime confirmation
- Lower Timeframe (LTF): Used for entry timing
- Signal alignment: Trade only when signals agree across timeframes
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np

from core.indicators import rsi, ema, sma


# =============================================================================
# Timeframe Conversion
# =============================================================================

# Mapping of timeframe strings to pandas resample rules
TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
}

# Timeframe hierarchy (from lowest to highest)
TF_HIERARCHY = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]


def tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    tf = tf.lower()
    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    
    num = ""
    unit = ""
    for char in tf:
        if char.isdigit():
            num += char
        else:
            unit += char
    
    return int(num) * multipliers.get(unit, 1)


def get_higher_timeframes(base_tf: str, levels: int = 2) -> List[str]:
    """
    Get higher timeframes for multi-timeframe analysis.
    
    Args:
        base_tf: Base timeframe (e.g., "30m")
        levels: Number of higher levels to include
    
    Returns:
        List of higher timeframes (e.g., ["1h", "4h"] for base "30m")
    """
    if base_tf.lower() not in TF_HIERARCHY:
        return []
    
    idx = TF_HIERARCHY.index(base_tf.lower())
    higher = TF_HIERARCHY[idx + 1:idx + 1 + levels]
    
    return higher


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        target_tf: Target timeframe string
    
    Returns:
        Resampled DataFrame
    """
    rule = TF_MAP.get(target_tf.lower(), target_tf)
    
    resampled = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled


# =============================================================================
# MTF Indicators
# =============================================================================

def mtf_rsi(
    df: pd.DataFrame,
    timeframes: List[str],
    rsi_period: int = 14,
    smoothing_type: str = "ema"
) -> Dict[str, pd.Series]:
    """
    Calculate RSI across multiple timeframes.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        timeframes: List of timeframes to calculate
        rsi_period: RSI period
        smoothing_type: RSI smoothing method
    
    Returns:
        Dictionary mapping timeframe -> RSI series (forward-filled to base TF)
    """
    results = {}
    
    for tf in timeframes:
        # Resample to target timeframe
        resampled = resample_ohlcv(df, tf)
        
        if len(resampled) < rsi_period:
            continue
        
        # Calculate RSI
        rsi_vals = rsi(resampled['Close'], n=rsi_period, smoothing_type=smoothing_type)
        
        # Forward-fill back to original index
        results[tf] = rsi_vals.reindex(df.index, method='ffill')
    
    return results


def mtf_trend(
    df: pd.DataFrame,
    timeframes: List[str],
    fast_period: int = 10,
    slow_period: int = 30
) -> Dict[str, pd.Series]:
    """
    Calculate trend direction across multiple timeframes.
    
    Uses EMA crossover to determine trend:
        1: Uptrend (fast > slow)
        -1: Downtrend (fast < slow)
        0: Neutral
    
    Returns:
        Dictionary mapping timeframe -> trend direction series
    """
    results = {}
    
    for tf in timeframes:
        resampled = resample_ohlcv(df, tf)
        
        if len(resampled) < slow_period:
            continue
        
        fast_ema = ema(resampled['Close'], fast_period)
        slow_ema = ema(resampled['Close'], slow_period)
        
        # Calculate trend direction
        trend = pd.Series(0, index=resampled.index)
        trend[fast_ema > slow_ema] = 1
        trend[fast_ema < slow_ema] = -1
        
        results[tf] = trend.reindex(df.index, method='ffill')
    
    return results


def mtf_momentum_score(
    df: pd.DataFrame,
    timeframes: List[str],
    rsi_period: int = 14
) -> pd.Series:
    """
    Calculate a composite momentum score across timeframes.
    
    Higher scores indicate stronger overbought conditions across
    multiple timeframes (better for short entry).
    
    Returns:
        Series with composite score (0-100)
    """
    rsi_values = mtf_rsi(df, timeframes, rsi_period)
    
    if not rsi_values:
        return pd.Series(50, index=df.index)
    
    # Average RSI across timeframes
    rsi_df = pd.DataFrame(rsi_values)
    composite = rsi_df.mean(axis=1)
    
    return composite


# =============================================================================
# MTF Alignment Detection
# =============================================================================

@dataclass
class MTFAlignment:
    """Result of multi-timeframe alignment check."""
    is_aligned: bool
    alignment_score: float  # 0-100
    details: Dict[str, bool]


def check_rsi_alignment(
    df: pd.DataFrame,
    timeframes: List[str],
    overbought_threshold: float = 70,
    require_all: bool = False
) -> pd.DataFrame:
    """
    Check if RSI is overbought across multiple timeframes.
    
    Args:
        df: OHLCV DataFrame
        timeframes: Timeframes to check
        overbought_threshold: RSI level for overbought
        require_all: If True, all TFs must be overbought. If False, majority.
    
    Returns:
        DataFrame with columns:
            - is_aligned: Boolean, True if alignment conditions met
            - alignment_score: 0-100, percentage of TFs that are overbought
            - {tf}_rsi: RSI value for each timeframe
    """
    rsi_values = mtf_rsi(df, timeframes)
    
    result = pd.DataFrame(index=df.index)
    
    # Add RSI for each timeframe
    for tf, rsi_series in rsi_values.items():
        result[f'{tf}_rsi'] = rsi_series
    
    # Calculate alignment
    rsi_cols = [f'{tf}_rsi' for tf in rsi_values.keys()]
    overbought_df = result[rsi_cols] >= overbought_threshold
    
    result['alignment_score'] = overbought_df.sum(axis=1) / len(rsi_cols) * 100
    
    if require_all:
        result['is_aligned'] = overbought_df.all(axis=1)
    else:
        # Majority alignment (>50% of TFs overbought)
        result['is_aligned'] = result['alignment_score'] > 50
    
    return result


def check_trend_alignment(
    df: pd.DataFrame,
    timeframes: List[str],
    required_direction: int = -1  # -1 for downtrend (good for shorts)
) -> pd.DataFrame:
    """
    Check if trend direction is aligned across timeframes.
    
    For short-only mean reversion, we want higher timeframes to show
    either ranging or downtrend (not strong uptrend).
    
    Args:
        df: OHLCV DataFrame
        timeframes: Timeframes to check
        required_direction: -1 for downtrend, 1 for uptrend, 0 for any non-contrary
    
    Returns:
        DataFrame with alignment information
    """
    trend_values = mtf_trend(df, timeframes)
    
    result = pd.DataFrame(index=df.index)
    
    for tf, trend_series in trend_values.items():
        result[f'{tf}_trend'] = trend_series
    
    trend_cols = [f'{tf}_trend' for tf in trend_values.keys()]
    
    if required_direction == 0:
        # For mean reversion, we don't want strong opposite trends
        # Count how many are NOT in strong uptrend (for shorts)
        favorable = (result[trend_cols] <= 0).sum(axis=1)
    else:
        # Count how many match the required direction
        favorable = (result[trend_cols] == required_direction).sum(axis=1)
    
    result['alignment_score'] = favorable / len(trend_cols) * 100
    result['is_aligned'] = result['alignment_score'] >= 50
    
    return result


# =============================================================================
# Integrated MTF Filter
# =============================================================================

def create_mtf_filter(
    df: pd.DataFrame,
    base_tf: str = "30m",
    htf_levels: int = 2,
    rsi_overbought: float = 70,
    require_rsi_alignment: bool = True,
    block_uptrend: bool = True,
    min_alignment_score: float = 50
) -> pd.Series:
    """
    Create a comprehensive MTF filter for trade entry.
    
    Combines RSI alignment and trend alignment across multiple
    timeframes to filter out low-probability trades.
    
    Args:
        df: OHLCV DataFrame
        base_tf: Base timeframe of the data
        htf_levels: Number of higher timeframes to analyze
        rsi_overbought: RSI threshold for overbought
        require_rsi_alignment: Require RSI alignment for entry
        block_uptrend: Block entries when higher TF shows uptrend
        min_alignment_score: Minimum alignment score (0-100)
    
    Returns:
        Boolean Series - True where trades are allowed
    """
    # Get higher timeframes
    htfs = get_higher_timeframes(base_tf, htf_levels)
    
    if not htfs:
        # No higher timeframes available, allow all
        return pd.Series(True, index=df.index)
    
    # Start with all True
    filter_mask = pd.Series(True, index=df.index)
    
    # Check RSI alignment
    if require_rsi_alignment:
        rsi_align = check_rsi_alignment(df, htfs, rsi_overbought)
        filter_mask &= rsi_align['alignment_score'] >= min_alignment_score
    
    # Check trend alignment (block strong uptrends)
    if block_uptrend:
        trend_align = check_trend_alignment(df, htfs, required_direction=0)
        # We want at least some TFs to NOT be in uptrend
        filter_mask &= trend_align['alignment_score'] >= 50
    
    return filter_mask


def add_mtf_indicators(
    df: pd.DataFrame,
    base_tf: str = "30m",
    htf_levels: int = 2,
    rsi_period: int = 14,
    prefix: str = "mtf_"
) -> pd.DataFrame:
    """
    Add multi-timeframe indicators to a DataFrame.
    
    This is the main function to call to add MTF analysis capabilities
    to your dataset before backtesting.
    
    Args:
        df: OHLCV DataFrame
        base_tf: Base timeframe
        htf_levels: Number of higher TFs to analyze
        rsi_period: RSI period
        prefix: Prefix for new columns
    
    Returns:
        DataFrame with added MTF columns:
            - {prefix}rsi_{tf}: RSI for each higher TF
            - {prefix}trend_{tf}: Trend for each higher TF
            - {prefix}rsi_alignment: RSI alignment score
            - {prefix}trend_alignment: Trend alignment score
            - {prefix}filter: Combined MTF filter
    """
    htfs = get_higher_timeframes(base_tf, htf_levels)
    
    if not htfs:
        return df
    
    # Get RSI values
    rsi_values = mtf_rsi(df, htfs, rsi_period)
    for tf, rsi_series in rsi_values.items():
        df[f'{prefix}rsi_{tf}'] = rsi_series
    
    # Get trend values
    trend_values = mtf_trend(df, htfs)
    for tf, trend_series in trend_values.items():
        df[f'{prefix}trend_{tf}'] = trend_series
    
    # Calculate alignment scores
    rsi_align = check_rsi_alignment(df, htfs, overbought_threshold=70)
    df[f'{prefix}rsi_alignment'] = rsi_align['alignment_score']
    
    trend_align = check_trend_alignment(df, htfs, required_direction=0)
    df[f'{prefix}trend_alignment'] = trend_align['alignment_score']
    
    # Create combined filter
    df[f'{prefix}filter'] = create_mtf_filter(df, base_tf, htf_levels)
    
    return df


# =============================================================================
# MTF Analysis Summary
# =============================================================================

def analyze_mtf_conditions(
    df: pd.DataFrame,
    base_tf: str = "30m",
    htf_levels: int = 2
) -> Dict:
    """
    Analyze multi-timeframe conditions across the dataset.
    
    Useful for understanding how often favorable MTF conditions occur.
    
    Returns:
        Dictionary with MTF analysis statistics
    """
    htfs = get_higher_timeframes(base_tf, htf_levels)
    
    if not htfs:
        return {"error": "No higher timeframes available for analysis"}
    
    # Get alignment data
    rsi_align = check_rsi_alignment(df, htfs)
    trend_align = check_trend_alignment(df, htfs, required_direction=0)
    
    total_bars = len(df)
    
    results = {
        "base_timeframe": base_tf,
        "higher_timeframes": htfs,
        "total_bars": total_bars,
        "rsi_alignment": {
            "mean_score": float(rsi_align['alignment_score'].mean()),
            "pct_above_50": float((rsi_align['alignment_score'] >= 50).mean() * 100),
            "pct_above_75": float((rsi_align['alignment_score'] >= 75).mean() * 100),
            "pct_100_aligned": float((rsi_align['alignment_score'] == 100).mean() * 100),
        },
        "trend_alignment": {
            "mean_score": float(trend_align['alignment_score'].mean()),
            "pct_favorable": float((trend_align['alignment_score'] >= 50).mean() * 100),
        }
    }
    
    # RSI stats per timeframe
    rsi_values = mtf_rsi(df, htfs)
    for tf, rsi_series in rsi_values.items():
        results[f"rsi_{tf}"] = {
            "mean": float(rsi_series.mean()),
            "pct_overbought": float((rsi_series >= 70).mean() * 100),
            "pct_oversold": float((rsi_series <= 30).mean() * 100),
        }
    
    return results
