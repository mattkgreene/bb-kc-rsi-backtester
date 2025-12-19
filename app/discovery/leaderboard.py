"""
Leaderboard System for Winning Strategies.

This module manages the ranking and presentation of winning strategies:
- Persistent leaderboard with configurable sorting
- Export winning strategies as presets
- Track strategy performance across different time periods
- Generate strategy recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from discovery.database import DiscoveryDatabase, BacktestRun, WinCriteria


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WinningStrategy:
    """
    A winning strategy from the leaderboard.
    
    Attributes:
        rank: Position in leaderboard
        params: Strategy parameters
        total_return: Total equity return %
        max_drawdown: Maximum drawdown %
        profit_factor: Profit factor
        win_rate: Win rate %
        sharpe_ratio: Sharpe ratio
        num_trades: Number of trades
        params_hash: Unique identifier
        discovered_at: When this strategy was found
    """
    rank: int
    params: Dict[str, Any]
    total_return: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    sharpe_ratio: float
    num_trades: int
    params_hash: str
    discovered_at: str
    
    def to_preset(self, name: str, description: str = "") -> Dict[str, Any]:
        """
        Convert this winning strategy to a preset format.
        
        Args:
            name: Name for the preset
            description: Optional description
        
        Returns:
            Preset dictionary compatible with STRATEGY_PRESETS
        """
        preset = {
            "name": name,
            "description": description or f"Auto-discovered winning strategy (Return: {self.total_return:.2f}%)",
            "category": "discovered",
        }
        
        # Add all strategy parameters
        for key, value in self.params.items():
            if key not in {"symbol", "timeframe", "start_ts", "end_ts", "exchange"}:
                preset[key] = value
        
        return preset
    
    def summary(self) -> str:
        """Generate a one-line summary of this strategy."""
        return (
            f"#{self.rank}: Return {self.total_return:.2f}% | "
            f"DD {self.max_drawdown:.2f}% | "
            f"PF {self.profit_factor:.2f} | "
            f"WR {self.win_rate:.1f}% | "
            f"{self.num_trades} trades"
        )


@dataclass
class LeaderboardStats:
    """Statistics about the current leaderboard."""
    total_winners: int
    avg_return: float
    avg_drawdown: float
    avg_profit_factor: float
    avg_win_rate: float
    best_return: float
    worst_drawdown: float
    top_strategy_params: Dict[str, Any]


# =============================================================================
# Leaderboard Class
# =============================================================================

class Leaderboard:
    """
    Manages ranking and presentation of winning strategies.
    
    Provides methods for:
    - Retrieving ranked lists of winners
    - Sorting by different metrics
    - Exporting winners as presets
    - Generating recommendations
    
    Example:
        >>> db = DiscoveryDatabase("data/discovery.db")
        >>> lb = Leaderboard(db)
        >>> top_10 = lb.get_top(10)
        >>> for strategy in top_10:
        ...     print(strategy.summary())
    """
    
    def __init__(self, db: DiscoveryDatabase):
        """
        Initialize leaderboard with database connection.
        
        Args:
            db: DiscoveryDatabase instance
        """
        self.db = db
    
    def get_top(
        self,
        n: int = 10,
        sort_by: str = "total_return",
        min_trades: int = 10,
    ) -> List[WinningStrategy]:
        """
        Get top N winning strategies.
        
        Args:
            n: Number of strategies to return
            sort_by: Metric to sort by ('total_return', 'profit_factor', 
                    'sharpe_ratio', 'win_rate', 'max_drawdown')
            min_trades: Minimum trades required
        
        Returns:
            List of WinningStrategy objects
        """
        # Map sort_by to ascending/descending
        ascending = sort_by == "max_drawdown"  # Lower DD is better
        
        winners = self.db.get_winners(
            limit=n * 2,  # Get extra to filter
            order_by=sort_by,
            ascending=ascending
        )
        
        # Filter by min_trades
        winners = [w for w in winners if w.num_trades >= min_trades]
        
        # Convert to WinningStrategy
        strategies = []
        for rank, run in enumerate(winners[:n], 1):
            strategies.append(WinningStrategy(
                rank=rank,
                params=run.params,
                total_return=run.total_return,
                max_drawdown=run.max_drawdown,
                profit_factor=run.profit_factor,
                win_rate=run.win_rate,
                sharpe_ratio=run.sharpe_ratio,
                num_trades=run.num_trades,
                params_hash=run.params_hash,
                discovered_at=run.run_timestamp,
            ))
        
        return strategies
    
    def get_by_criteria(
        self,
        min_return: float = 0.0,
        max_drawdown: float = 100.0,
        min_profit_factor: float = 0.0,
        min_win_rate: float = 0.0,
        min_trades: int = 10,
        limit: int = 50,
    ) -> List[WinningStrategy]:
        """
        Get strategies matching specific criteria.
        
        Args:
            min_return: Minimum total return %
            max_drawdown: Maximum drawdown %
            min_profit_factor: Minimum profit factor
            min_win_rate: Minimum win rate %
            min_trades: Minimum number of trades
            limit: Maximum strategies to return
        
        Returns:
            List of matching WinningStrategy objects
        """
        winners = self.db.get_winners(limit=limit * 2)
        
        filtered = []
        for run in winners:
            if (
                run.total_return >= min_return
                and run.max_drawdown <= max_drawdown
                and run.profit_factor >= min_profit_factor
                and run.win_rate >= min_win_rate
                and run.num_trades >= min_trades
            ):
                filtered.append(run)
        
        # Convert and rank
        strategies = []
        for rank, run in enumerate(filtered[:limit], 1):
            strategies.append(WinningStrategy(
                rank=rank,
                params=run.params,
                total_return=run.total_return,
                max_drawdown=run.max_drawdown,
                profit_factor=run.profit_factor,
                win_rate=run.win_rate,
                sharpe_ratio=run.sharpe_ratio,
                num_trades=run.num_trades,
                params_hash=run.params_hash,
                discovered_at=run.run_timestamp,
            ))
        
        return strategies
    
    def get_stats(self) -> LeaderboardStats:
        """
        Get summary statistics for the leaderboard.
        
        Returns:
            LeaderboardStats with averages and extremes
        """
        winners = self.db.get_winners(limit=1000)
        
        if not winners:
            return LeaderboardStats(
                total_winners=0,
                avg_return=0,
                avg_drawdown=0,
                avg_profit_factor=0,
                avg_win_rate=0,
                best_return=0,
                worst_drawdown=0,
                top_strategy_params={},
            )
        
        returns = [w.total_return for w in winners]
        drawdowns = [w.max_drawdown for w in winners]
        pfs = [w.profit_factor for w in winners if w.profit_factor < 999]  # Exclude inf
        wrs = [w.win_rate for w in winners]
        
        return LeaderboardStats(
            total_winners=len(winners),
            avg_return=sum(returns) / len(returns),
            avg_drawdown=sum(drawdowns) / len(drawdowns),
            avg_profit_factor=sum(pfs) / len(pfs) if pfs else 0,
            avg_win_rate=sum(wrs) / len(wrs),
            best_return=max(returns),
            worst_drawdown=max(drawdowns),
            top_strategy_params=winners[0].params if winners else {},
        )
    
    def export_as_presets(
        self,
        top_n: int = 5,
        name_prefix: str = "discovered",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export top strategies as preset configurations.
        
        Args:
            top_n: Number of strategies to export
            name_prefix: Prefix for preset names
        
        Returns:
            Dictionary of preset configurations
        """
        top = self.get_top(top_n)
        
        presets = {}
        for strategy in top:
            name = f"{name_prefix}_{strategy.rank}"
            description = (
                f"Discovered strategy #{strategy.rank}: "
                f"{strategy.total_return:.2f}% return, "
                f"{strategy.max_drawdown:.2f}% max DD, "
                f"PF {strategy.profit_factor:.2f}"
            )
            
            preset = strategy.to_preset(name, description)
            presets[name] = preset
        
        return presets
    
    def get_strategy_by_hash(self, params_hash: str) -> Optional[WinningStrategy]:
        """
        Retrieve a specific strategy by its parameter hash.
        
        Args:
            params_hash: Unique hash identifier
        
        Returns:
            WinningStrategy if found, None otherwise
        """
        run = self.db.get_run_by_hash(params_hash)
        
        if not run or not run.is_winner:
            return None
        
        return WinningStrategy(
            rank=0,  # Not ranked in this context
            params=run.params,
            total_return=run.total_return,
            max_drawdown=run.max_drawdown,
            profit_factor=run.profit_factor,
            win_rate=run.win_rate,
            sharpe_ratio=run.sharpe_ratio,
            num_trades=run.num_trades,
            params_hash=run.params_hash,
            discovered_at=run.run_timestamp,
        )
    
    def compare_strategies(
        self,
        hash1: str,
        hash2: str,
    ) -> Dict[str, Any]:
        """
        Compare two strategies side by side.
        
        Args:
            hash1: First strategy hash
            hash2: Second strategy hash
        
        Returns:
            Comparison dictionary
        """
        s1 = self.get_strategy_by_hash(hash1)
        s2 = self.get_strategy_by_hash(hash2)
        
        if not s1 or not s2:
            return {"error": "One or both strategies not found"}
        
        # Find parameter differences
        param_diffs = {}
        all_params = set(s1.params.keys()) | set(s2.params.keys())
        
        for param in all_params:
            v1 = s1.params.get(param)
            v2 = s2.params.get(param)
            if v1 != v2:
                param_diffs[param] = {"strategy_1": v1, "strategy_2": v2}
        
        return {
            "strategy_1": {
                "hash": hash1,
                "total_return": s1.total_return,
                "max_drawdown": s1.max_drawdown,
                "profit_factor": s1.profit_factor,
                "win_rate": s1.win_rate,
                "num_trades": s1.num_trades,
            },
            "strategy_2": {
                "hash": hash2,
                "total_return": s2.total_return,
                "max_drawdown": s2.max_drawdown,
                "profit_factor": s2.profit_factor,
                "win_rate": s2.win_rate,
                "num_trades": s2.num_trades,
            },
            "parameter_differences": param_diffs,
            "return_diff": s1.total_return - s2.total_return,
            "drawdown_diff": s1.max_drawdown - s2.max_drawdown,
        }
    
    def get_recommendations(
        self,
        risk_tolerance: str = "moderate",
    ) -> List[WinningStrategy]:
        """
        Get strategy recommendations based on risk tolerance.
        
        Args:
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        
        Returns:
            List of recommended strategies
        """
        if risk_tolerance == "conservative":
            # Prioritize low drawdown, decent returns
            return self.get_by_criteria(
                min_return=5.0,
                max_drawdown=10.0,
                min_profit_factor=1.5,
                min_win_rate=50.0,
                limit=5
            )
        elif risk_tolerance == "aggressive":
            # Prioritize high returns, accept higher drawdown
            return self.get_top(
                n=5,
                sort_by="total_return",
                min_trades=10
            )
        else:  # moderate
            # Balance between return and risk
            return self.get_by_criteria(
                min_return=10.0,
                max_drawdown=15.0,
                min_profit_factor=1.3,
                min_win_rate=45.0,
                limit=5
            )
    
    def format_leaderboard(
        self,
        top_n: int = 10,
        sort_by: str = "total_return",
    ) -> str:
        """
        Generate a formatted leaderboard string.
        
        Args:
            top_n: Number of strategies to show
            sort_by: Sorting metric
        
        Returns:
            Formatted leaderboard string
        """
        strategies = self.get_top(top_n, sort_by=sort_by)
        
        if not strategies:
            return "No winning strategies found yet. Run discovery to find winners."
        
        lines = [
            "=" * 80,
            "WINNING STRATEGIES LEADERBOARD",
            f"Sorted by: {sort_by}",
            "=" * 80,
            "",
            f"{'Rank':<5} {'Return':<10} {'MaxDD':<10} {'PF':<8} {'WinRate':<10} {'Trades':<8}",
            "-" * 80,
        ]
        
        for s in strategies:
            lines.append(
                f"#{s.rank:<4} {s.total_return:>8.2f}% {s.max_drawdown:>8.2f}% "
                f"{s.profit_factor:>6.2f} {s.win_rate:>8.1f}% {s.num_trades:>6}"
            )
        
        lines.append("-" * 80)
        
        # Add stats summary
        stats = self.get_stats()
        lines.extend([
            "",
            f"Total Winners: {stats.total_winners}",
            f"Average Return: {stats.avg_return:.2f}%",
            f"Average Drawdown: {stats.avg_drawdown:.2f}%",
            f"Best Return: {stats.best_return:.2f}%",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================

def export_leaderboard_to_json(
    db: DiscoveryDatabase,
    filepath: str,
    top_n: int = 50,
) -> None:
    """
    Export leaderboard to a JSON file.
    
    Args:
        db: Database instance
        filepath: Output file path
        top_n: Number of strategies to export
    """
    lb = Leaderboard(db)
    strategies = lb.get_top(top_n)
    
    data = {
        "exported_at": datetime.utcnow().isoformat(),
        "total_strategies": len(strategies),
        "strategies": [
            {
                "rank": s.rank,
                "params": s.params,
                "total_return": s.total_return,
                "max_drawdown": s.max_drawdown,
                "profit_factor": s.profit_factor,
                "win_rate": s.win_rate,
                "sharpe_ratio": s.sharpe_ratio,
                "num_trades": s.num_trades,
                "params_hash": s.params_hash,
            }
            for s in strategies
        ],
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_strategy_from_json(filepath: str, rank: int = 1) -> Optional[Dict[str, Any]]:
    """
    Load a specific strategy from exported JSON.
    
    Args:
        filepath: Path to exported JSON
        rank: Rank of strategy to load
    
    Returns:
        Strategy parameters if found
    """
    with open(filepath) as f:
        data = json.load(f)
    
    for strategy in data.get("strategies", []):
        if strategy.get("rank") == rank:
            return strategy.get("params")
    
    return None
