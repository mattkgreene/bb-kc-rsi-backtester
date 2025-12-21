"""
Optimization module for the BB+KC+RSI backtester.

This package keeps imports lazy so unit test discovery doesn't require the full
scientific stack (pandas/numpy) unless optimization is actually used.
"""

__all__ = ["run_grid_search", "OptimizationResult", "DEFAULT_PARAM_GRID", "QUICK_PARAM_GRID"]


def __getattr__(name: str):
    if name in __all__:
        from . import grid_search

        return getattr(grid_search, name)
    raise AttributeError(name)
