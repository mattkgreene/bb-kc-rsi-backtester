"""
Shared utilities for the BB+KC+RSI backtester.

This module contains common helper functions used across multiple
modules in the application to avoid code duplication.
"""

from typing import Union
import numpy as np
import pandas as pd


def cmp(a: float, b: float, op: str) -> bool:
    """
    Compare two values using a dynamic operator.
    
    This function allows runtime selection of comparison operators,
    useful for configurable trading rules.
    
    Args:
        a: First value (left operand).
        b: Second value (right operand).
        op: Comparison operator string.
            Supported: '<', '<=', '>', '>='
    
    Returns:
        Result of the comparison. Returns True if operator is invalid.
    
    Examples:
        >>> cmp(5, 3, '>')
        True
        >>> cmp(5, 5, '>=')
        True
        >>> cmp(3, 5, '<')
        True
    """
    if op == "<":  return a < b
    if op == "<=": return a <= b
    if op == ">":  return a > b
    if op == ">=": return a >= b
    return True


def vectorized_cmp(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """
    Vectorized comparison of two arrays using a dynamic operator.
    
    This is the vectorized version of cmp() for use with numpy arrays
    in performance-critical signal detection.
    
    Args:
        a: First array (left operand).
        b: Second array (right operand).
        op: Comparison operator string.
            Supported: '<', '<=', '>', '>='
    
    Returns:
        Boolean array with comparison results.
        Returns array of True if operator is invalid.
    
    Examples:
        >>> a = np.array([1, 5, 3])
        >>> b = np.array([2, 3, 3])
        >>> vectorized_cmp(a, b, '<')
        array([ True, False, False])
    """
    if op == "<":  return a < b
    if op == "<=": return a <= b
    if op == ">":  return a > b
    if op == ">=": return a >= b
    return np.ones(len(a), dtype=bool)


def format_duration(td: pd.Timedelta) -> str:
    """
    Format a Timedelta as a human-readable string.
    
    Args:
        td: pandas Timedelta object.
    
    Returns:
        Formatted string like '2d 5h 30m' or '45m 30s'.
    
    Examples:
        >>> format_duration(pd.Timedelta(hours=26, minutes=30))
        '1d 2h 30m'
    """
    if pd.isna(td):
        return "N/A"
    
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "N/A"
    
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and days == 0 and hours == 0:
        parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"


def format_pct(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%).
        decimals: Number of decimal places.
    
    Returns:
        Formatted string like '5.00%'.
    
    Examples:
        >>> format_pct(0.0523, 2)
        '5.23%'
    """
    if pd.isna(value) or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value.
        symbol: Currency symbol prefix.
        decimals: Number of decimal places.
    
    Returns:
        Formatted string like '$1,234.56'.
    
    Examples:
        >>> format_currency(1234.5678)
        '$1,234.57'
    """
    if pd.isna(value) or not isinstance(value, (int, float)):
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if division is not possible.
    
    Returns:
        Result of division or default value.
    
    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=float('inf'))
        inf
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator


def calculate_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Calculate drawdown series and maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values over time.
    
    Returns:
        Tuple of (drawdown_pct_array, max_drawdown_pct):
            - drawdown_pct_array: Drawdown at each point as percentage
            - max_drawdown_pct: Maximum drawdown as percentage
    
    Examples:
        >>> equity = np.array([100, 110, 105, 115, 100])
        >>> dd, max_dd = calculate_drawdown(equity)
        >>> max_dd  # (115 - 100) / 115 * 100
        13.04...
    """
    if equity_curve is None or len(equity_curve) == 0:
        return np.array([]), 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = (running_max - equity_curve) / running_max * 100.0
    max_drawdown_pct = float(np.nanmax(drawdown_pct))
    
    return drawdown_pct, max_drawdown_pct
