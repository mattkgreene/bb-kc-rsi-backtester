"""
Grid search optimization for the BB+KC+RSI backtester.

This module provides parameter optimization via grid search to find
configurations that maximize profit factor or other metrics.

Key features:
- Configurable parameter grids
- Multiple optimization targets (profit_factor, sharpe, win_rate, etc.)
- Progress tracking with callback support
- Result ranking and analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Any, Optional, Callable, Literal
import math
import pandas as pd
import numpy as np

from backtest.engine import run_backtest


# =============================================================================
# Type Definitions
# =============================================================================

OptimizationMetric = Literal[
    "profit_factor",
    "sharpe_ratio", 
    "sortino_ratio",
    "win_rate",
    "total_equity_return_pct",
    "calmar_ratio",
]


@dataclass
class OptimizationResult:
    """
    Result of a single backtest run during optimization.
    
    Attributes:
        params: Dictionary of parameters used
        profit_factor: Gross profit / gross loss ratio
        win_rate: Percentage of winning trades
        total_return: Total equity return percentage
        max_drawdown: Maximum drawdown percentage
        sharpe_ratio: Risk-adjusted return (annualized)
        sortino_ratio: Downside-adjusted return
        calmar_ratio: Return / max drawdown
        num_trades: Total number of trades
        avg_return: Average return per trade
    """
    params: Dict[str, Any]
    profit_factor: float
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    num_trades: int
    avg_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for DataFrame creation."""
        return {
            **self.params,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "num_trades": self.num_trades,
            "avg_return": self.avg_return,
        }


@dataclass
class GridSearchConfig:
    """
    Configuration for grid search optimization.
    
    Attributes:
        param_grid: Dictionary mapping parameter names to lists of values
        metric: Optimization target metric
        min_trades: Minimum trades required for valid result
        top_n: Number of top results to return
    """
    param_grid: Dict[str, List[Any]]
    metric: OptimizationMetric = "profit_factor"
    min_trades: int = 5
    top_n: int = 20


# =============================================================================
# Default Parameter Grids
# =============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    # Entry conditions - most impactful parameters
    "rsi_min": [65, 68, 70, 72, 75],
    "rsi_ma_min": [65, 68, 70, 72],
    "entry_band_mode": ["Either", "KC", "BB", "Both"],
    
    # Band parameters
    "bb_std": [1.8, 2.0, 2.2],
    "kc_mult": [1.8, 2.0, 2.2],
    
    # Risk parameters
    "stop_pct": [1.5, 2.0, 2.5, 3.0],
    "trail_pct": [1.0, 1.5, 2.0],
}
# Total combinations: 5 * 4 * 4 * 3 * 3 * 4 * 3 = 4,320


QUICK_PARAM_GRID: Dict[str, List[Any]] = {
    # Reduced grid for faster testing
    "rsi_min": [68, 70, 72],
    "rsi_ma_min": [68, 70],
    "entry_band_mode": ["Either", "Both"],
    "stop_pct": [1.5, 2.0, 2.5],
}
# Total combinations: 3 * 2 * 2 * 3 = 36


COMPREHENSIVE_PARAM_GRID: Dict[str, List[Any]] = {
    # Full parameter sweep - use with caution (many combinations)
    "rsi_min": [62, 65, 68, 70, 72, 75, 78],
    "rsi_ma_min": [62, 65, 68, 70, 72, 75],
    "entry_band_mode": ["Either", "KC", "BB", "Both"],
    "bb_std": [1.6, 1.8, 2.0, 2.2, 2.4],
    "kc_mult": [1.6, 1.8, 2.0, 2.2, 2.4],
    "stop_pct": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "trail_pct": [0.8, 1.0, 1.5, 2.0, 2.5],
    "exit_level": ["mid", "lower"],
}
# Total combinations: 7 * 6 * 4 * 5 * 5 * 6 * 5 * 2 = 252,000


# =============================================================================
# Grid Search Implementation
# =============================================================================

def _count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Calculate total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def _generate_param_combinations(
    param_grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from the grid.
    
    Args:
        param_grid: Dictionary mapping parameter names to value lists
    
    Returns:
        List of parameter dictionaries, one per combination
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def run_grid_search(
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    base_params: Dict[str, Any],
    metric: OptimizationMetric = "profit_factor",
    min_trades: int = 5,
    top_n: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Run grid search optimization over parameter combinations.
    
    Iterates through all combinations of parameters in param_grid,
    runs a backtest for each, and returns the top results ranked
    by the specified metric.
    
    Args:
        df: OHLCV DataFrame for backtesting
        param_grid: Dictionary mapping parameter names to lists of values
                   to test (e.g., {"rsi_min": [65, 70, 75]})
        base_params: Base parameter dictionary to merge with grid params.
                    Should include all required backtest parameters.
        metric: Optimization target. One of:
                - "profit_factor" (default, recommended)
                - "sharpe_ratio"
                - "sortino_ratio"
                - "win_rate"
                - "total_equity_return_pct"
                - "calmar_ratio"
        min_trades: Minimum number of trades required for a valid result.
                   Configurations with fewer trades are filtered out.
        top_n: Number of top results to return
        progress_callback: Optional callback(current, total) for progress updates
    
    Returns:
        DataFrame with top_n results, sorted by metric (descending).
        Columns include all grid parameters plus performance metrics.
    
    Example:
        >>> param_grid = {"rsi_min": [68, 70, 72], "stop_pct": [1.5, 2.0]}
        >>> results = run_grid_search(df, param_grid, base_params)
        >>> best_config = results.iloc[0].to_dict()
    """
    combinations = _generate_param_combinations(param_grid)
    total = len(combinations)
    results: List[OptimizationResult] = []
    
    for i, combo in enumerate(combinations):
        # Merge with base params
        params = base_params.copy()
        params.update(combo)
        
        # Handle stop_pct / stop_atr_mult based on stop_mode
        if "stop_pct" in combo and params.get("stop_mode") == "ATR":
            # If testing stop_pct but mode is ATR, skip or convert
            continue
        
        try:
            # Run backtest
            stats, ds, trades, equity_curve = run_backtest(
                df,
                timeframe=params.get("timeframe", "30m"),
                bb_len=params.get("bb_len", 20),
                bb_std=params.get("bb_std", 2.0),
                bb_basis_type=params.get("bb_basis_type", "sma"),
                kc_ema_len=params.get("kc_ema_len", 20),
                kc_atr_len=params.get("kc_atr_len", 14),
                kc_mult=params.get("kc_mult", 2.0),
                kc_mid_type=params.get("kc_mid_type", "ema"),
                rsi_len_30m=params.get("rsi_len_30m", 14),
                rsi_ma_len=params.get("rsi_ma_len", 10),
                rsi_smoothing_type=params.get("rsi_smoothing_type", "ema"),
                rsi_ma_type=params.get("rsi_ma_type", "sma"),
                rsi_min=params.get("rsi_min", 70),
                rsi_ma_min=params.get("rsi_ma_min", 70),
                use_rsi_relation=params.get("use_rsi_relation", True),
                rsi_relation=params.get("rsi_relation", ">="),
                entry_band_mode=params.get("entry_band_mode", "Either"),
                exit_channel=params.get("exit_channel", "BB"),
                exit_level=params.get("exit_level", "mid"),
                cash=params.get("cash", 10000),
                commission=params.get("commission", 0.001),
                trade_mode=params.get("trade_mode", "Simple (1x spot-style)"),
                use_stop=params.get("use_stop", True),
                stop_mode=params.get("stop_mode", "Fixed %"),
                stop_pct=params.get("stop_pct", 2.0),
                stop_atr_mult=params.get("stop_atr_mult", 2.0),
                use_trailing=params.get("use_trailing", False),
                trail_pct=params.get("trail_pct", 1.0),
                max_bars_in_trade=params.get("max_bars_in_trade", 100),
                daily_loss_limit=params.get("daily_loss_limit", 3.0),
                risk_per_trade_pct=params.get("risk_per_trade_pct", 1.0),
                max_leverage=params.get("max_leverage"),
                maintenance_margin_pct=params.get("maintenance_margin_pct"),
                max_margin_utilization=params.get("max_margin_utilization"),
            )
            
            num_trades = int(stats.get("trades", 0))
            
            # Skip if not enough trades
            if num_trades < min_trades:
                continue
            
            # Extract metrics
            profit_factor = float(stats.get("profit_factor", 0))
            if math.isinf(profit_factor):
                profit_factor = 999.0  # Cap infinity for sorting
            
            result = OptimizationResult(
                params=combo,
                profit_factor=profit_factor,
                win_rate=float(stats.get("win_rate", 0)),
                total_return=float(stats.get("total_equity_return_pct", 0)),
                max_drawdown=float(stats.get("max_drawdown_pct", 0)),
                sharpe_ratio=float(stats.get("sharpe_ratio", 0)),
                sortino_ratio=float(stats.get("sortino_ratio", 0)),
                calmar_ratio=float(stats.get("calmar_ratio", 0)),
                num_trades=num_trades,
                avg_return=float(stats.get("avg_return_pct", 0)),
            )
            results.append(result)
            
        except Exception as e:
            # Skip failed backtests
            continue
        
        # Progress callback
        if progress_callback:
            progress_callback(i + 1, total)
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame([r.to_dict() for r in results])
    
    # Sort by metric (descending)
    if metric in results_df.columns:
        results_df = results_df.sort_values(metric, ascending=False)
    
    # Return top N
    return results_df.head(top_n).reset_index(drop=True)


def create_custom_grid(
    rsi_range: tuple = (65, 75),
    stop_range: tuple = (1.5, 3.0),
    band_mult_range: tuple = (1.8, 2.2),
    steps: int = 3,
    include_entry_modes: bool = True,
    include_exit_levels: bool = False,
) -> Dict[str, List[Any]]:
    """
    Create a custom parameter grid based on ranges.
    
    Args:
        rsi_range: (min, max) for RSI minimum threshold
        stop_range: (min, max) for stop loss percentage
        band_mult_range: (min, max) for BB std and KC multiplier
        steps: Number of steps between min and max
        include_entry_modes: Include all entry band modes
        include_exit_levels: Include mid and lower exit levels
    
    Returns:
        Parameter grid dictionary
    """
    def linspace(start, end, n):
        return [round(start + i * (end - start) / (n - 1), 1) for i in range(n)]
    
    grid = {
        "rsi_min": [int(x) for x in linspace(rsi_range[0], rsi_range[1], steps)],
        "rsi_ma_min": [int(x) for x in linspace(rsi_range[0], rsi_range[1], steps)],
        "stop_pct": linspace(stop_range[0], stop_range[1], steps),
        "bb_std": linspace(band_mult_range[0], band_mult_range[1], steps),
        "kc_mult": linspace(band_mult_range[0], band_mult_range[1], steps),
    }
    
    if include_entry_modes:
        grid["entry_band_mode"] = ["Either", "KC", "BB", "Both"]
    
    if include_exit_levels:
        grid["exit_level"] = ["mid", "lower"]
    
    return grid


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze optimization results to find patterns.
    
    Args:
        results_df: DataFrame from run_grid_search
    
    Returns:
        Dictionary with analysis insights:
        - best_params: Best parameter combination
        - param_importance: Which parameters vary most in top results
        - recommended_ranges: Suggested parameter ranges
    """
    if results_df.empty:
        return {"error": "No results to analyze"}
    
    # Identify grid parameters (non-metric columns)
    metric_cols = {
        "profit_factor", "win_rate", "total_return", "max_drawdown",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "num_trades", "avg_return"
    }
    param_cols = [c for c in results_df.columns if c not in metric_cols]
    
    # Best parameters
    best_row = results_df.iloc[0]
    best_params = {col: best_row[col] for col in param_cols}
    
    # Parameter frequency in top 10
    top_10 = results_df.head(10)
    param_modes = {}
    for col in param_cols:
        if col in top_10.columns:
            mode_val = top_10[col].mode()
            if len(mode_val) > 0:
                param_modes[col] = mode_val.iloc[0]
    
    # Calculate parameter ranges in top results
    recommended_ranges = {}
    for col in param_cols:
        if col in top_10.columns:
            vals = top_10[col]
            if vals.dtype in [np.float64, np.int64, float, int]:
                recommended_ranges[col] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                }
    
    return {
        "best_params": best_params,
        "common_in_top_10": param_modes,
        "recommended_ranges": recommended_ranges,
        "top_profit_factor": float(results_df["profit_factor"].iloc[0]),
        "avg_profit_factor_top_10": float(top_10["profit_factor"].mean()),
    }


def export_results_to_preset(
    result_row: pd.Series,
    name: str,
    description: str,
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert an optimization result to a preset format.
    
    Args:
        result_row: Row from optimization results DataFrame
        name: Name for the new preset
        description: Description for the preset
        base_params: Base parameters to merge with
    
    Returns:
        Preset dictionary ready to add to STRATEGY_PRESETS
    """
    # Metric columns to exclude from params
    metric_cols = {
        "profit_factor", "win_rate", "total_return", "max_drawdown",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "num_trades", "avg_return"
    }
    
    # Extract optimized parameters
    optimized = {
        k: v for k, v in result_row.to_dict().items()
        if k not in metric_cols
    }
    
    # Merge with base
    preset = base_params.copy()
    preset.update(optimized)
    preset["name"] = name
    preset["description"] = description
    preset["category"] = "optimized"
    
    return preset
