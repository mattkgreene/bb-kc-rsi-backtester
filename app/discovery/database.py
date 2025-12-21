"""
SQLite persistence layer for Strategy Discovery System.

This module provides database operations for storing and retrieving:
- Backtest run results with full parameters and metrics
- Winning strategies that meet configurable criteria
- Discovered rules/patterns from analysis

The database enables incremental discovery by tracking which parameter
combinations have already been tested.
"""

from __future__ import annotations

import sqlite3
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestRun:
    """
    Record of a single backtest run with parameters and results.
    
    Attributes:
        params_hash: Unique hash of parameter combination
        params: Full parameter dictionary (JSON serialized)
        symbol: Trading symbol (e.g., 'BTC/USD')
        timeframe: Candle timeframe (e.g., '30m')
        start_date: Backtest start date
        end_date: Backtest end date
        total_return: Total equity return percentage
        max_drawdown: Maximum drawdown percentage
        profit_factor: Gross profit / gross loss
        win_rate: Percentage of winning trades
        sharpe_ratio: Risk-adjusted return
        sortino_ratio: Downside-adjusted return
        calmar_ratio: Return / max drawdown
        num_trades: Total number of trades
        avg_return: Average return per trade
        run_timestamp: When this backtest was executed
        is_winner: Whether this meets winning criteria
    """
    params_hash: str
    params: Dict[str, Any]
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_return: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    num_trades: int
    avg_return: float
    run_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_winner: bool = False
    id: Optional[int] = None


@dataclass
class WinCriteria:
    """
    Configurable criteria for determining a 'winning' strategy.
    
    Attributes:
        min_total_return: Minimum total return % (default: 0 = positive)
        max_drawdown: Maximum allowed drawdown % (default: 20%)
        min_trades: Minimum trades for statistical significance (default: 10)
        min_profit_factor: Minimum profit factor (default: 1.0)
        min_win_rate: Minimum win rate % (default: 0)
        min_calmar: Minimum Calmar ratio (return/drawdown)
        max_profit_factor: Maximum profit factor (to filter overfitting)
        max_win_rate: Maximum win rate (to filter suspiciously high)
    """
    min_total_return: float = 0.0
    max_drawdown: float = 20.0
    min_trades: int = 10
    min_profit_factor: float = 1.0
    min_win_rate: float = 0.0
    min_calmar: float = 0.0  # Return / Drawdown ratio
    max_profit_factor: float = 10.0  # Filter likely overfit
    max_win_rate: float = 90.0  # Filter suspicious results
    
    def is_winner(self, run: BacktestRun) -> bool:
        """Check if a backtest run meets winning criteria."""
        # Calculate Calmar ratio
        calmar = run.total_return / run.max_drawdown if run.max_drawdown > 0 else float('inf')
        
        return (
            run.total_return >= self.min_total_return
            and run.max_drawdown <= self.max_drawdown
            and run.num_trades >= self.min_trades
            and run.profit_factor >= self.min_profit_factor
            and run.profit_factor <= self.max_profit_factor
            and run.win_rate >= self.min_win_rate
            and run.win_rate <= self.max_win_rate
            and calmar >= self.min_calmar
        )


@dataclass
class RobustWinCriteria(WinCriteria):
    """
    Enhanced win criteria focused on long-term robustness.
    
    These stricter criteria help filter out overfit strategies that
    look good in backtests but fail in live trading.
    """
    min_total_return: float = 5.0  # Must show meaningful profit
    max_drawdown: float = 25.0
    min_trades: int = 30  # Need more trades for significance
    min_profit_factor: float = 1.2  # Must have real edge
    min_win_rate: float = 35.0  # Not unrealistically low
    min_calmar: float = 0.5  # Decent risk-adjusted return
    max_profit_factor: float = 5.0  # Lower max to filter overfit
    max_win_rate: float = 80.0  # Lower max to filter suspicious


@dataclass 
class DiscoveredRule:
    """
    A pattern discovered from analyzing winning strategies.
    
    Attributes:
        rule_id: Unique identifier
        parameter: Parameter name this rule applies to
        condition: Description of the condition (e.g., "70-74")
        occurrence_pct: Percentage of winners with this pattern
        avg_return_with: Average return when pattern present
        avg_return_without: Average return when pattern absent
        confidence: Statistical confidence (0-1)
        description: Human-readable description
        discovered_at: When this rule was discovered
    """
    parameter: str
    condition: str
    occurrence_pct: float
    avg_return_with: float
    avg_return_without: float
    confidence: float
    description: str
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    rule_id: Optional[int] = None


# =============================================================================
# Database Class
# =============================================================================

class DiscoveryDatabase:
    """
    SQLite database for persisting strategy discovery results.
    
    Provides methods for:
    - Storing/retrieving backtest runs
    - Managing winning strategies
    - Storing discovered rules
    - Checking if parameter combinations have been tested
    
    Example:
        >>> db = DiscoveryDatabase("data/discovery.db")
        >>> db.initialize()
        >>> db.save_run(run)
        >>> winners = db.get_winners()
    """
    
    def __init__(self, db_path: str = "data/discovery.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize(self) -> None:
        """
        Create database tables if they don't exist.
        
        Tables:
        - backtest_runs: All individual backtest results
        - discovered_rules: Patterns found in winning strategies
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Backtest runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    params_hash TEXT UNIQUE NOT NULL,
                    params_json TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    total_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    sortino_ratio REAL NOT NULL,
                    calmar_ratio REAL NOT NULL,
                    num_trades INTEGER NOT NULL,
                    avg_return REAL NOT NULL,
                    run_timestamp TEXT NOT NULL,
                    is_winner INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # Discovered rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_rules (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    occurrence_pct REAL NOT NULL,
                    avg_return_with REAL NOT NULL,
                    avg_return_without REAL NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    UNIQUE(parameter, condition)
                )
            """)
            
            # Index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_params_hash 
                ON backtest_runs(params_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_winner 
                ON backtest_runs(is_winner)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_total_return 
                ON backtest_runs(total_return DESC)
            """)
    
    @staticmethod
    def compute_params_hash(params: Dict[str, Any]) -> str:
        """
        Compute a unique hash for a parameter combination.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            SHA256 hash of sorted, serialized parameters
        """
        # Sort keys and serialize for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
    
    def has_been_tested(self, params: Dict[str, Any]) -> bool:
        """
        Check if a parameter combination has already been tested.
        
        Args:
            params: Parameter dictionary to check
        
        Returns:
            True if this combination exists in the database
        """
        params_hash = self.compute_params_hash(params)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM backtest_runs WHERE params_hash = ?",
                (params_hash,)
            )
            return cursor.fetchone() is not None
    
    def save_run(self, run: BacktestRun, criteria: Optional[WinCriteria] = None) -> int:
        """
        Save a backtest run to the database.
        
        Args:
            run: BacktestRun object to save
            criteria: Optional WinCriteria to determine is_winner
        
        Returns:
            Database ID of the saved run
        """
        if criteria:
            run.is_winner = criteria.is_winner(run)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO backtest_runs (
                    params_hash, params_json, symbol, timeframe,
                    start_date, end_date, total_return, max_drawdown,
                    profit_factor, win_rate, sharpe_ratio, sortino_ratio,
                    calmar_ratio, num_trades, avg_return, run_timestamp, is_winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.params_hash,
                json.dumps(run.params, default=str),
                run.symbol,
                run.timeframe,
                run.start_date,
                run.end_date,
                run.total_return,
                run.max_drawdown,
                run.profit_factor,
                run.win_rate,
                run.sharpe_ratio,
                run.sortino_ratio,
                run.calmar_ratio,
                run.num_trades,
                run.avg_return,
                run.run_timestamp,
                int(run.is_winner)
            ))
            return cursor.lastrowid
    
    def save_runs_batch(
        self, 
        runs: List[BacktestRun], 
        criteria: Optional[WinCriteria] = None
    ) -> int:
        """
        Save multiple backtest runs efficiently.
        
        Args:
            runs: List of BacktestRun objects
            criteria: Optional WinCriteria to determine winners
        
        Returns:
            Number of runs saved
        """
        if not runs:
            return 0
        
        if criteria:
            for run in runs:
                run.is_winner = criteria.is_winner(run)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO backtest_runs (
                    params_hash, params_json, symbol, timeframe,
                    start_date, end_date, total_return, max_drawdown,
                    profit_factor, win_rate, sharpe_ratio, sortino_ratio,
                    calmar_ratio, num_trades, avg_return, run_timestamp, is_winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    run.params_hash,
                    json.dumps(run.params, default=str),
                    run.symbol,
                    run.timeframe,
                    run.start_date,
                    run.end_date,
                    run.total_return,
                    run.max_drawdown,
                    run.profit_factor,
                    run.win_rate,
                    run.sharpe_ratio,
                    run.sortino_ratio,
                    run.calmar_ratio,
                    run.num_trades,
                    run.avg_return,
                    run.run_timestamp,
                    int(run.is_winner)
                )
                for run in runs
            ])
            return len(runs)
    
    def get_winners(
        self, 
        limit: int = 100,
        order_by: str = "total_return",
        ascending: bool = False
    ) -> List[BacktestRun]:
        """
        Get winning strategies from the database.
        
        Args:
            limit: Maximum number of results
            order_by: Column to sort by
            ascending: Sort direction
        
        Returns:
            List of BacktestRun objects meeting win criteria
        """
        direction = "ASC" if ascending else "DESC"
        valid_columns = {
            "total_return", "max_drawdown", "profit_factor",
            "win_rate", "sharpe_ratio", "num_trades", "run_timestamp"
        }
        if order_by not in valid_columns:
            order_by = "total_return"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM backtest_runs 
                WHERE is_winner = 1
                ORDER BY {order_by} {direction}
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_run(row) for row in rows]
    
    def get_all_runs(
        self,
        limit: int = 1000,
        winners_only: bool = False
    ) -> List[BacktestRun]:
        """
        Get all backtest runs from the database.
        
        Args:
            limit: Maximum number of results
            winners_only: If True, only return winning strategies
        
        Returns:
            List of BacktestRun objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if winners_only:
                cursor.execute("""
                    SELECT * FROM backtest_runs 
                    WHERE is_winner = 1
                    ORDER BY total_return DESC
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT * FROM backtest_runs 
                    ORDER BY total_return DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_run(row) for row in rows]
    
    def get_run_by_hash(self, params_hash: str) -> Optional[BacktestRun]:
        """
        Get a specific run by its parameter hash.
        
        Args:
            params_hash: Hash of the parameter combination
        
        Returns:
            BacktestRun if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM backtest_runs WHERE params_hash = ?",
                (params_hash,)
            )
            row = cursor.fetchone()
            return self._row_to_run(row) if row else None
    
    def _row_to_run(self, row: sqlite3.Row) -> BacktestRun:
        """Convert a database row to a BacktestRun object."""
        return BacktestRun(
            id=row["id"],
            params_hash=row["params_hash"],
            params=json.loads(row["params_json"]),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            total_return=row["total_return"],
            max_drawdown=row["max_drawdown"],
            profit_factor=row["profit_factor"],
            win_rate=row["win_rate"],
            sharpe_ratio=row["sharpe_ratio"],
            sortino_ratio=row["sortino_ratio"],
            calmar_ratio=row["calmar_ratio"],
            num_trades=row["num_trades"],
            avg_return=row["avg_return"],
            run_timestamp=row["run_timestamp"],
            is_winner=bool(row["is_winner"])
        )
    
    def save_rule(self, rule: DiscoveredRule) -> int:
        """
        Save a discovered rule to the database.
        
        Args:
            rule: DiscoveredRule object
        
        Returns:
            Database ID of the saved rule
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO discovered_rules (
                    parameter, condition, occurrence_pct,
                    avg_return_with, avg_return_without,
                    confidence, description, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.parameter,
                rule.condition,
                rule.occurrence_pct,
                rule.avg_return_with,
                rule.avg_return_without,
                rule.confidence,
                rule.description,
                rule.discovered_at
            ))
            return cursor.lastrowid
    
    def get_rules(self, min_confidence: float = 0.0) -> List[DiscoveredRule]:
        """
        Get discovered rules from the database.
        
        Args:
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of DiscoveredRule objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM discovered_rules
                WHERE confidence >= ?
                ORDER BY confidence DESC, occurrence_pct DESC
            """, (min_confidence,))
            
            rows = cursor.fetchall()
            return [
                DiscoveredRule(
                    rule_id=row["rule_id"],
                    parameter=row["parameter"],
                    condition=row["condition"],
                    occurrence_pct=row["occurrence_pct"],
                    avg_return_with=row["avg_return_with"],
                    avg_return_without=row["avg_return_without"],
                    confidence=row["confidence"],
                    description=row["description"],
                    discovered_at=row["discovered_at"]
                )
                for row in rows
            ]
    
    def clear_rules(self) -> None:
        """Clear all discovered rules (for re-analysis)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM discovered_rules")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about the discovery database.
        
        Returns:
            Dictionary with counts, averages, and distributions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total runs
            cursor.execute("SELECT COUNT(*) FROM backtest_runs")
            total_runs = cursor.fetchone()[0]
            
            # Winners count
            cursor.execute("SELECT COUNT(*) FROM backtest_runs WHERE is_winner = 1")
            winners_count = cursor.fetchone()[0]
            
            # Best performer
            cursor.execute("""
                SELECT total_return, params_json FROM backtest_runs
                ORDER BY total_return DESC LIMIT 1
            """)
            best = cursor.fetchone()
            best_return = best[0] if best else 0
            
            # Average metrics for winners
            cursor.execute("""
                SELECT 
                    AVG(total_return) as avg_return,
                    AVG(max_drawdown) as avg_drawdown,
                    AVG(profit_factor) as avg_pf,
                    AVG(win_rate) as avg_wr
                FROM backtest_runs WHERE is_winner = 1
            """)
            winner_stats = cursor.fetchone()
            
            # Rules count
            cursor.execute("SELECT COUNT(*) FROM discovered_rules")
            rules_count = cursor.fetchone()[0]
            
            return {
                "total_runs": total_runs,
                "winners_count": winners_count,
                "win_rate_pct": (winners_count / total_runs * 100) if total_runs > 0 else 0,
                "best_return": best_return,
                "avg_winner_return": winner_stats[0] if winner_stats[0] else 0,
                "avg_winner_drawdown": winner_stats[1] if winner_stats[1] else 0,
                "avg_winner_pf": winner_stats[2] if winner_stats[2] else 0,
                "avg_winner_win_rate": winner_stats[3] if winner_stats[3] else 0,
                "rules_discovered": rules_count,
            }
    
    def get_tested_hashes(self) -> set:
        """
        Get all parameter hashes that have been tested.
        
        Returns:
            Set of parameter hash strings
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT params_hash FROM backtest_runs")
            return {row[0] for row in cursor.fetchall()}
    
    def update_winner_status(self, criteria: WinCriteria) -> int:
        """
        Re-evaluate all runs against new win criteria.
        
        Args:
            criteria: New WinCriteria to apply
        
        Returns:
            Number of winners after re-evaluation
        """
        runs = self.get_all_runs(limit=100000, winners_only=False)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for run in runs:
                is_winner = criteria.is_winner(run)
                cursor.execute(
                    "UPDATE backtest_runs SET is_winner = ? WHERE params_hash = ?",
                    (int(is_winner), run.params_hash)
                )
        
        return len([r for r in runs if criteria.is_winner(r)])
