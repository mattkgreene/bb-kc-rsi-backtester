"""
Strategy Discovery System.

This package provides automated strategy discovery, testing, and learning
capabilities for the BB+KC+RSI backtester.

Modules:
- database: SQLite persistence layer for backtest results
- engine: Discovery runner for systematic parameter search
- rules: Pattern discovery and rule extraction
- leaderboard: Winning strategy management and ranking
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "DiscoveryDatabase",
    "run_discovery",
    "DiscoveryConfig",
    "find_winning_patterns",
    "DiscoveredRule",
    "Leaderboard",
    "WinningStrategy",
    "WinCriteria",
    "BacktestRun",
]
