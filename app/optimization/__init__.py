"""
Optimization module for the BB+KC+RSI backtester.

This module provides tools for parameter optimization including:
- Grid search over parameter combinations
- Result analysis and ranking
- Configuration export

Usage:
    from optimization.grid_search import run_grid_search, DEFAULT_PARAM_GRID
"""

from optimization.grid_search import (
    run_grid_search,
    OptimizationResult,
    DEFAULT_PARAM_GRID,
    QUICK_PARAM_GRID,
)

__all__ = [
    "run_grid_search",
    "OptimizationResult",
    "DEFAULT_PARAM_GRID",
    "QUICK_PARAM_GRID",
]
