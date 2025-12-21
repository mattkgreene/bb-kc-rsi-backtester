"""
Discovery Engine for systematic strategy parameter search.

This module provides the core discovery functionality:
- Exhaustive parameter sweep across all strategy variations
- Incremental testing (skips already-tested combinations)
- Progress tracking and callbacks
- Results persistence to database

The engine builds on the existing grid_search module but adds:
- Persistence of all results
- Incremental search capability
- Broader parameter space exploration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import math
import pandas as pd
import numpy as np

from ..backtest.engine import run_backtest
from .database import (
    DiscoveryDatabase, 
    BacktestRun, 
    WinCriteria
)


# =============================================================================
# Worker Functions for Parallel Execution
# =============================================================================

def _run_single_backtest(args: Tuple[Dict, Dict, str]) -> Optional[Dict]:
    """
    Worker function for parallel backtest execution.
    
    This runs in a separate process, so we can't pass complex objects.
    Instead, we pass serializable data and return a dict result.
    
    Args:
        args: Tuple of (params_dict, combo_dict, params_hash)
    
    Returns:
        Dictionary with backtest results, or None if failed
    """
    params, combo, params_hash = args
    
    try:
        # Import here to avoid pickle issues in multiprocessing
        from ..backtest.engine import run_backtest
        import pandas as pd
        import math
        
        # Reconstruct DataFrame from serialized data
        df_data = params.pop("_df_data", {})
        index_data = df_data.pop("_index", None)
        df = pd.DataFrame(df_data)
        if index_data is not None:
            df.index = pd.to_datetime(index_data)
        
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
            rsi_max=params.get("rsi_max"),
            rsi_ma_max=params.get("rsi_ma_max"),
            use_rsi_relation=params.get("use_rsi_relation", True),
            rsi_relation=params.get("rsi_relation", ">="),
            entry_band_mode=params.get("entry_band_mode", "Either"),
            trade_direction=params.get("trade_direction", "Short"),
            exit_channel=params.get("exit_channel", "BB"),
            exit_level=params.get("exit_level", "mid"),
            cash=params.get("cash", 10000),
            commission=params.get("commission", 0.001),
            trade_mode=params.get("trade_mode", "Margin / Futures"),
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
        
        # Extract metrics
        num_trades = int(stats.get("trades", 0))
        total_return = float(stats.get("total_equity_return_pct", 0))
        profit_factor = float(stats.get("profit_factor", 0))
        
        # Cap infinity profit factor
        if math.isinf(profit_factor):
            profit_factor = 999.0
        
        return {
            "params_hash": params_hash,
            "combo": combo,
            "symbol": params.get("symbol", "BTC/USD"),
            "timeframe": params.get("timeframe", "30m"),
            "start_date": str(params.get("start_ts", "")),
            "end_date": str(params.get("end_ts", "")),
            "total_return": total_return,
            "max_drawdown": float(stats.get("max_drawdown_pct", 0)),
            "profit_factor": profit_factor,
            "win_rate": float(stats.get("win_rate", 0)),
            "sharpe_ratio": float(stats.get("sharpe_ratio", 0)),
            "sortino_ratio": float(stats.get("sortino_ratio", 0)),
            "calmar_ratio": float(stats.get("calmar_ratio", 0)),
            "num_trades": num_trades,
            "avg_return": float(stats.get("avg_return_pct", 0)),
        }
        
    except Exception as e:
        return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiscoveryConfig:
    """
    Configuration for a discovery run.
    
    Attributes:
        param_grid: Dictionary mapping parameter names to lists of values
        win_criteria: Criteria for determining winners
        skip_tested: Whether to skip already-tested combinations
        batch_size: Number of results to batch before saving
        max_combinations: Maximum combinations to test (None = unlimited)
    """
    param_grid: Dict[str, List[Any]]
    win_criteria: WinCriteria = field(default_factory=WinCriteria)
    skip_tested: bool = True
    batch_size: int = 50
    max_combinations: Optional[int] = None


# =============================================================================
# Default Parameter Grids
# =============================================================================

# Comprehensive grid for full discovery
DISCOVERY_PARAM_GRID: Dict[str, List[Any]] = {
    # RSI Entry Thresholds (most impactful)
    "rsi_min": [62, 64, 66, 68, 70, 72, 74, 76, 78, 80],
    "rsi_ma_min": [60, 62, 64, 66, 68, 70, 72, 74, 76, 78],
    
    # Band Entry Mode
    "entry_band_mode": ["Either", "KC", "BB", "Both"],
    
    # RSI Relation
    "use_rsi_relation": [True, False],
    "rsi_relation": [">=", ">"],
    
    # Exit Configuration
    "exit_channel": ["BB", "KC"],
    "exit_level": ["mid", "lower"],
    
    # Band Multipliers
    "bb_std": [1.6, 1.8, 2.0, 2.2, 2.4],
    "kc_mult": [1.6, 1.8, 2.0, 2.2, 2.4],
    
    # Stop Loss
    "use_stop": [True],
    "stop_pct": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    
    # Trailing Stop
    "use_trailing": [True, False],
    "trail_pct": [1.0, 1.5, 2.0, 2.5],
}
# Estimated combinations: ~1.4 million (use sampling or incremental)


# Quick discovery grid for faster iteration
QUICK_DISCOVERY_GRID: Dict[str, List[Any]] = {
    "rsi_min": [66, 68, 70, 72, 74],
    "rsi_ma_min": [64, 66, 68, 70, 72],
    "entry_band_mode": ["Either", "KC", "BB", "Both"],
    "exit_level": ["mid", "lower"],
    "bb_std": [1.8, 2.0, 2.2],
    "kc_mult": [1.8, 2.0, 2.2],
    "stop_pct": [1.5, 2.0, 2.5, 3.0],
}
# Estimated combinations: 5 * 5 * 4 * 2 * 3 * 3 * 4 = 7,200


# Focused grid targeting promising ranges
FOCUSED_DISCOVERY_GRID: Dict[str, List[Any]] = {
    "rsi_min": [68, 70, 72, 74],
    "rsi_ma_min": [66, 68, 70, 72],
    "entry_band_mode": ["Either", "KC", "Both"],
    "exit_level": ["mid", "lower"],
    "bb_std": [1.9, 2.0, 2.1],
    "kc_mult": [1.9, 2.0, 2.1],
    "stop_pct": [1.5, 2.0, 2.5],
    "use_trailing": [True, False],
    "trail_pct": [1.0, 1.5, 2.0],
}
# Estimated combinations: 4 * 4 * 3 * 2 * 3 * 3 * 3 * 2 * 3 = 15,552


# =============================================================================
# Margin/Futures Parameter Grids
# =============================================================================

# Margin-specific parameters
MARGIN_PARAM_GRID: Dict[str, List[Any]] = {
    # Trade Mode
    "trade_mode": ["Margin / Futures"],
    
    # Leverage options
    "max_leverage": [2.0, 3.0, 5.0, 10.0],
    
    # Risk per trade (% of equity)
    "risk_per_trade_pct": [0.5, 1.0, 1.5, 2.0],
    
    # Maintenance margin
    "maintenance_margin_pct": [0.5, 1.0],
    
    # Stop modes
    "stop_mode": ["Fixed %", "ATR"],
    
    # ATR stop multiplier (for ATR mode)
    "stop_atr_mult": [1.5, 2.0, 2.5],
    
    # Fixed stop % (for Fixed % mode)  
    "stop_pct": [1.5, 2.0, 2.5, 3.0],
    
    # Trailing
    "use_trailing": [True, False],
    "trail_pct": [1.0, 1.5, 2.0],
}


# Combined grid: Base strategy + Margin params
FULL_MARGIN_DISCOVERY_GRID: Dict[str, List[Any]] = {
    # RSI Entry
    "rsi_min": [68, 70, 72, 74],
    "rsi_ma_min": [66, 68, 70, 72],
    
    # Entry/Exit
    "entry_band_mode": ["Either", "KC", "Both"],
    "exit_level": ["mid", "lower"],
    
    # Bands
    "bb_std": [1.9, 2.0, 2.1],
    "kc_mult": [1.9, 2.0, 2.1],
    
    # Trade Mode
    "trade_mode": ["Margin / Futures"],
    
    # Leverage (margin mode only)
    "max_leverage": [2.0, 5.0, 10.0],
    
    # Risk sizing
    "risk_per_trade_pct": [0.5, 1.0, 2.0],
    
    # Stop configuration
    "use_stop": [True],
    "stop_mode": ["Fixed %", "ATR"],
    "stop_pct": [1.5, 2.0, 2.5],
    "stop_atr_mult": [1.5, 2.0, 2.5],
    
    # Trailing
    "use_trailing": [True, False],
    "trail_pct": [1.0, 1.5, 2.0],
}


# =============================================================================
# Helper Functions
# =============================================================================

def count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Calculate total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def generate_combinations(
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


def filter_invalid_combinations(
    combinations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filter out invalid parameter combinations.
    
    Removes combinations that don't make sense, like:
    - use_trailing=False with specific trail_pct
    - use_stop=False with specific stop_pct
    - use_rsi_relation=False with specific rsi_relation
    - Fixed % stop mode with ATR multiplier
    - ATR stop mode with fixed stop %
    """
    valid = []
    seen_hashes = set()
    
    for combo in combinations:
        # Skip duplicates that arise from irrelevant param values
        # Create a normalized version for deduplication
        normalized = combo.copy()
        
        # If trailing disabled, trail_pct doesn't matter
        if not combo.get("use_trailing", True):
            normalized.pop("trail_pct", None)
        
        # If stop disabled, stop_pct and stop_atr_mult don't matter
        if not combo.get("use_stop", True):
            normalized.pop("stop_pct", None)
            normalized.pop("stop_atr_mult", None)
            normalized.pop("stop_mode", None)
        
        # If RSI relation disabled, rsi_relation doesn't matter
        if not combo.get("use_rsi_relation", True):
            normalized.pop("rsi_relation", None)
        
        # Handle stop mode specifics
        stop_mode = combo.get("stop_mode", "Fixed %")
        if stop_mode == "Fixed %":
            # Fixed % mode: stop_atr_mult doesn't matter
            normalized.pop("stop_atr_mult", None)
        elif stop_mode == "ATR":
            # ATR mode: stop_pct doesn't matter
            normalized.pop("stop_pct", None)
        
        # Create hash for deduplication
        combo_hash = str(sorted(normalized.items()))
        if combo_hash not in seen_hashes:
            seen_hashes.add(combo_hash)
            valid.append(combo)
    
    return valid


# =============================================================================
# Discovery Engine
# =============================================================================

@dataclass
class DiscoveryResult:
    """
    Result of a discovery run.
    
    Attributes:
        total_tested: Total combinations tested
        new_tested: Combinations that were new (not skipped)
        winners_found: Number of winning strategies found
        best_return: Best total return found
        best_params: Parameters of best performer
        duration_seconds: Time taken for discovery
    """
    total_tested: int
    new_tested: int
    winners_found: int
    best_return: float
    best_params: Dict[str, Any]
    duration_seconds: float


def run_discovery(
    df: pd.DataFrame,
    base_params: Dict[str, Any],
    db: DiscoveryDatabase,
    config: Optional[DiscoveryConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DiscoveryResult:
    """
    Run strategy discovery with exhaustive parameter search.
    
    This function:
    1. Generates all parameter combinations from the grid
    2. Skips already-tested combinations (if configured)
    3. Runs backtests for each combination
    4. Saves results to database
    5. Tracks and returns winning strategies
    
    Args:
        df: OHLCV DataFrame for backtesting
        base_params: Base parameter dictionary (exchange, symbol, dates, etc.)
        db: DiscoveryDatabase instance for persistence
        config: DiscoveryConfig with param_grid and criteria
        progress_callback: Optional callback(current, total, status_msg)
    
    Returns:
        DiscoveryResult with summary of findings
    
    Example:
        >>> db = DiscoveryDatabase("data/discovery.db")
        >>> db.initialize()
        >>> config = DiscoveryConfig(param_grid=QUICK_DISCOVERY_GRID)
        >>> result = run_discovery(df, base_params, db, config)
        >>> print(f"Found {result.winners_found} winners")
    """
    start_time = datetime.now()
    
    # Use default config if not provided
    if config is None:
        config = DiscoveryConfig(param_grid=QUICK_DISCOVERY_GRID)
    
    # Generate and filter combinations
    all_combinations = generate_combinations(config.param_grid)
    combinations = filter_invalid_combinations(all_combinations)
    
    # Apply max_combinations limit if set
    if config.max_combinations and len(combinations) > config.max_combinations:
        # Randomly sample to stay within limit
        np.random.shuffle(combinations)
        combinations = combinations[:config.max_combinations]
    
    total = len(combinations)
    
    # Get already-tested hashes for skip functionality
    tested_hashes = db.get_tested_hashes() if config.skip_tested else set()
    
    # Track results
    results_batch: List[BacktestRun] = []
    tested_count = 0
    new_count = 0
    winners_count = 0
    best_return = float("-inf")
    best_params = {}
    
    for i, combo in enumerate(combinations):
        # Merge with base params
        params = base_params.copy()
        params.update(combo)
        
        # Compute hash for this combination
        params_hash = DiscoveryDatabase.compute_params_hash(params)
        
        # Skip if already tested
        if config.skip_tested and params_hash in tested_hashes:
            tested_count += 1
            if progress_callback and i % 100 == 0:
                progress_callback(i + 1, total, f"Skipped {tested_count} tested")
            continue
        
        new_count += 1
        
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
                rsi_max=params.get("rsi_max"),
                rsi_ma_max=params.get("rsi_ma_max"),
                use_rsi_relation=params.get("use_rsi_relation", True),
                rsi_relation=params.get("rsi_relation", ">="),
                entry_band_mode=params.get("entry_band_mode", "Either"),
                trade_direction=params.get("trade_direction", "Short"),
                exit_channel=params.get("exit_channel", "BB"),
                exit_level=params.get("exit_level", "mid"),
                cash=params.get("cash", 10000),
                commission=params.get("commission", 0.001),
                trade_mode=params.get("trade_mode", "Margin / Futures"),
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
            
            # Extract metrics
            num_trades = int(stats.get("trades", 0))
            total_return = float(stats.get("total_equity_return_pct", 0))
            profit_factor = float(stats.get("profit_factor", 0))
            
            # Cap infinity profit factor
            if math.isinf(profit_factor):
                profit_factor = 999.0
            
            # Create run record
            run = BacktestRun(
                params_hash=params_hash,
                params=combo,  # Store only the grid params, not full params
                symbol=params.get("symbol", "BTC/USD"),
                timeframe=params.get("timeframe", "30m"),
                start_date=str(params.get("start_ts", "")),
                end_date=str(params.get("end_ts", "")),
                total_return=total_return,
                max_drawdown=float(stats.get("max_drawdown_pct", 0)),
                profit_factor=profit_factor,
                win_rate=float(stats.get("win_rate", 0)),
                sharpe_ratio=float(stats.get("sharpe_ratio", 0)),
                sortino_ratio=float(stats.get("sortino_ratio", 0)),
                calmar_ratio=float(stats.get("calmar_ratio", 0)),
                num_trades=num_trades,
                avg_return=float(stats.get("avg_return_pct", 0)),
            )
            
            # Check if winner
            run.is_winner = config.win_criteria.is_winner(run)
            if run.is_winner:
                winners_count += 1
            
            # Track best
            if total_return > best_return:
                best_return = total_return
                best_params = combo.copy()
            
            # Add to batch
            results_batch.append(run)
            
            # Save batch periodically
            if len(results_batch) >= config.batch_size:
                db.save_runs_batch(results_batch)
                results_batch = []
            
        except Exception as e:
            # Skip failed backtests
            continue
        
        # Progress callback
        if progress_callback and new_count % 10 == 0:
            progress_callback(
                i + 1, 
                total, 
                f"Tested {new_count} new | {winners_count} winners | Best: {best_return:.2f}%"
            )
    
    # Save remaining batch
    if results_batch:
        db.save_runs_batch(results_batch)
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    
    return DiscoveryResult(
        total_tested=total,
        new_tested=new_count,
        winners_found=winners_count,
        best_return=best_return if best_return > float("-inf") else 0,
        best_params=best_params,
        duration_seconds=duration
    )


def run_discovery_parallel(
    df: pd.DataFrame,
    base_params: Dict[str, Any],
    db: DiscoveryDatabase,
    config: Optional[DiscoveryConfig] = None,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DiscoveryResult:
    """
    Run strategy discovery with parallel execution for faster processing.
    
    Uses ProcessPoolExecutor to run multiple backtests concurrently,
    providing ~Nx speedup where N is the number of CPU cores.
    
    Args:
        df: OHLCV DataFrame for backtesting
        base_params: Base parameter dictionary (exchange, symbol, dates, etc.)
        db: DiscoveryDatabase instance for persistence
        config: DiscoveryConfig with param_grid and criteria
        n_workers: Number of parallel workers (default: CPU count)
        progress_callback: Optional callback(current, total, status_msg)
    
    Returns:
        DiscoveryResult with summary of findings
    
    Example:
        >>> result = run_discovery_parallel(df, base_params, db, config, n_workers=8)
        >>> print(f"Found {result.winners_found} winners in {result.duration_seconds:.1f}s")
    """
    start_time = datetime.now()
    
    # Use default config if not provided
    if config is None:
        config = DiscoveryConfig(param_grid=QUICK_DISCOVERY_GRID)
    
    # Default to CPU count minus 1 (leave one for main process)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # Generate and filter combinations
    all_combinations = generate_combinations(config.param_grid)
    combinations = filter_invalid_combinations(all_combinations)
    
    # Apply max_combinations limit if set
    if config.max_combinations and len(combinations) > config.max_combinations:
        np.random.shuffle(combinations)
        combinations = combinations[:config.max_combinations]
    
    total = len(combinations)
    
    # Get already-tested hashes for skip functionality
    tested_hashes = db.get_tested_hashes() if config.skip_tested else set()
    
    # Prepare DataFrame data for serialization (workers can't receive DataFrame directly)
    df_data = df.to_dict('list')
    df_data['_index'] = df.index.tolist()
    
    # Build work items (skip already tested)
    work_items = []
    skipped_count = 0
    
    for combo in combinations:
        params = base_params.copy()
        params.update(combo)
        params_hash = DiscoveryDatabase.compute_params_hash(params)
        
        if config.skip_tested and params_hash in tested_hashes:
            skipped_count += 1
            continue
        
        # Include DataFrame data in params for worker
        params["_df_data"] = df_data
        work_items.append((params, combo, params_hash))
    
    if progress_callback:
        progress_callback(0, total, f"Skipped {skipped_count} already tested. Starting {len(work_items)} tests with {n_workers} workers...")
    
    if not work_items:
        return DiscoveryResult(
            total_tested=total,
            new_tested=0,
            winners_found=0,
            best_return=0,
            best_params={},
            duration_seconds=(datetime.now() - start_time).total_seconds()
        )
    
    # Track results
    results_batch: List[BacktestRun] = []
    completed_count = 0
    winners_count = 0
    best_return = float("-inf")
    best_params = {}
    
    # Run parallel execution
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all work
        futures = {executor.submit(_run_single_backtest, item): item for item in work_items}
        
        # Process results as they complete
        for future in as_completed(futures):
            completed_count += 1
            
            try:
                result = future.result()
                
                if result is not None:
                    # Create BacktestRun from result dict
                    run = BacktestRun(
                        params_hash=result["params_hash"],
                        params=result["combo"],
                        symbol=result["symbol"],
                        timeframe=result["timeframe"],
                        start_date=result["start_date"],
                        end_date=result["end_date"],
                        total_return=result["total_return"],
                        max_drawdown=result["max_drawdown"],
                        profit_factor=result["profit_factor"],
                        win_rate=result["win_rate"],
                        sharpe_ratio=result["sharpe_ratio"],
                        sortino_ratio=result["sortino_ratio"],
                        calmar_ratio=result["calmar_ratio"],
                        num_trades=result["num_trades"],
                        avg_return=result["avg_return"],
                    )
                    
                    # Check if winner
                    run.is_winner = config.win_criteria.is_winner(run)
                    if run.is_winner:
                        winners_count += 1
                    
                    # Track best
                    if result["total_return"] > best_return:
                        best_return = result["total_return"]
                        best_params = result["combo"].copy()
                    
                    # Add to batch
                    results_batch.append(run)
                    
                    # Save batch periodically (in main process - SQLite safe)
                    if len(results_batch) >= config.batch_size:
                        db.save_runs_batch(results_batch)
                        results_batch = []
                
            except Exception as e:
                # Worker error - skip this result
                pass
            
            # Progress callback
            if progress_callback and completed_count % 20 == 0:
                progress_callback(
                    skipped_count + completed_count,
                    total,
                    f"Tested {completed_count}/{len(work_items)} | {winners_count} winners | Best: {best_return:.2f}%"
                )
    
    # Save remaining batch
    if results_batch:
        db.save_runs_batch(results_batch)
    
    # Final progress
    if progress_callback:
        progress_callback(total, total, f"Complete! {winners_count} winners found")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    return DiscoveryResult(
        total_tested=total,
        new_tested=completed_count,
        winners_found=winners_count,
        best_return=best_return if best_return > float("-inf") else 0,
        best_params=best_params,
        duration_seconds=duration
    )


def run_incremental_discovery(
    df: pd.DataFrame,
    base_params: Dict[str, Any],
    db: DiscoveryDatabase,
    max_new_tests: int = 1000,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DiscoveryResult:
    """
    Run incremental discovery, only testing unexplored combinations.
    
    This is useful for continuing discovery after previous runs,
    automatically skipping already-tested parameters.
    
    Args:
        df: OHLCV DataFrame
        base_params: Base parameters
        db: Database instance
        max_new_tests: Maximum new combinations to test
        progress_callback: Progress callback function
    
    Returns:
        DiscoveryResult with findings
    """
    config = DiscoveryConfig(
        param_grid=QUICK_DISCOVERY_GRID,
        skip_tested=True,
        max_combinations=max_new_tests
    )
    
    return run_discovery(df, base_params, db, config, progress_callback)


def create_discovery_grid(
    rsi_range: Tuple[int, int] = (66, 76),
    rsi_ma_range: Tuple[int, int] = (64, 74),
    band_mult_range: Tuple[float, float] = (1.8, 2.2),
    stop_range: Tuple[float, float] = (1.5, 3.0),
    rsi_step: int = 2,
    mult_step: float = 0.1,
    stop_step: float = 0.5,
    include_all_modes: bool = True,
) -> Dict[str, List[Any]]:
    """
    Create a custom discovery grid based on ranges.
    
    Args:
        rsi_range: (min, max) for RSI minimum threshold
        rsi_ma_range: (min, max) for RSI MA threshold
        band_mult_range: (min, max) for BB std and KC multiplier
        stop_range: (min, max) for stop loss percentage
        rsi_step: Step size for RSI values
        mult_step: Step size for multipliers
        stop_step: Step size for stop percentage
        include_all_modes: Include all entry band modes
    
    Returns:
        Parameter grid dictionary
    """
    def arange(start, end, step):
        """Generate range including endpoint."""
        values = []
        current = start
        while current <= end + step/2:  # +step/2 to handle float precision
            values.append(round(current, 2) if isinstance(step, float) else int(current))
            current += step
        return values
    
    grid = {
        "rsi_min": arange(rsi_range[0], rsi_range[1], rsi_step),
        "rsi_ma_min": arange(rsi_ma_range[0], rsi_ma_range[1], rsi_step),
        "bb_std": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "kc_mult": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "stop_pct": arange(stop_range[0], stop_range[1], stop_step),
        "exit_level": ["mid", "lower"],
    }
    
    if include_all_modes:
        grid["entry_band_mode"] = ["Either", "KC", "BB", "Both"]
    else:
        grid["entry_band_mode"] = ["Either", "Both"]
    
    return grid


def estimate_discovery_time(
    param_grid: Dict[str, List[Any]],
    tested_count: int = 0,
    seconds_per_test: float = 0.1,
    n_workers: int = 1
) -> Dict[str, Any]:
    """
    Estimate time required for a discovery run.
    
    Args:
        param_grid: Parameter grid to estimate
        tested_count: Number of already-tested combinations
        seconds_per_test: Estimated time per backtest
        n_workers: Number of parallel workers
    
    Returns:
        Dictionary with time estimates
    """
    total = count_combinations(param_grid)
    remaining = max(0, total - tested_count)
    
    # Account for parallel execution
    effective_workers = max(1, n_workers)
    seconds = (remaining * seconds_per_test) / effective_workers
    minutes = seconds / 60
    hours = minutes / 60
    
    return {
        "total_combinations": total,
        "already_tested": tested_count,
        "remaining": remaining,
        "n_workers": effective_workers,
        "estimated_seconds": round(seconds, 1),
        "estimated_minutes": round(minutes, 1),
        "estimated_hours": round(hours, 2),
        "human_readable": (
            f"{hours:.1f} hours" if hours >= 1 
            else f"{minutes:.0f} minutes" if minutes >= 1
            else f"{seconds:.0f} seconds"
        )
    }


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    return mp.cpu_count()


def create_margin_discovery_grid(
    rsi_range: Tuple[int, int] = (68, 74),
    rsi_ma_range: Tuple[int, int] = (66, 72),
    band_mult_range: Tuple[float, float] = (1.9, 2.1),
    leverage_options: List[float] = None,
    risk_pct_options: List[float] = None,
    include_atr_stops: bool = True,
    include_trailing: bool = True,
    rsi_step: int = 2,
    mult_step: float = 0.1,
) -> Dict[str, List[Any]]:
    """
    Create a discovery grid that includes margin/futures parameters.
    
    Args:
        rsi_range: (min, max) for RSI minimum threshold
        rsi_ma_range: (min, max) for RSI MA threshold
        band_mult_range: (min, max) for BB std and KC multiplier
        leverage_options: List of leverage values to test
        risk_pct_options: List of risk per trade % values
        include_atr_stops: Include ATR-based stop mode
        include_trailing: Include trailing stop variations
        rsi_step: Step size for RSI values
        mult_step: Step size for multipliers
    
    Returns:
        Parameter grid dictionary with margin parameters
    """
    def arange(start, end, step):
        """Generate range including endpoint."""
        values = []
        current = start
        while current <= end + step/2:
            values.append(round(current, 2) if isinstance(step, float) else int(current))
            current += step
        return values
    
    # Default leverage and risk options
    if leverage_options is None:
        leverage_options = [2.0, 5.0, 10.0]
    if risk_pct_options is None:
        risk_pct_options = [0.5, 1.0, 2.0]
    
    grid = {
        # RSI params
        "rsi_min": arange(rsi_range[0], rsi_range[1], rsi_step),
        "rsi_ma_min": arange(rsi_ma_range[0], rsi_ma_range[1], rsi_step),
        
        # Band params
        "bb_std": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "kc_mult": arange(band_mult_range[0], band_mult_range[1], mult_step),
        
        # Entry/Exit
        "entry_band_mode": ["Either", "KC", "Both"],
        "exit_level": ["mid", "lower"],
        
        # Trade mode
        "trade_mode": ["Margin / Futures"],
        
        # Margin params
        "max_leverage": leverage_options,
        "risk_per_trade_pct": risk_pct_options,
        "maintenance_margin_pct": [0.5],  # Standard maintenance margin
        
        # Stop configuration
        "use_stop": [True],
        "stop_pct": [1.5, 2.0, 2.5],
    }
    
    # Add ATR stops if enabled
    if include_atr_stops:
        grid["stop_mode"] = ["Fixed %", "ATR"]
        grid["stop_atr_mult"] = [1.5, 2.0, 2.5]
    else:
        grid["stop_mode"] = ["Fixed %"]
    
    # Add trailing if enabled
    if include_trailing:
        grid["use_trailing"] = [True, False]
        grid["trail_pct"] = [1.0, 1.5, 2.0]
    else:
        grid["use_trailing"] = [False]
    
    return grid


def estimate_filtered_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """
    Estimate the number of valid combinations after filtering.
    
    This accounts for combinations that will be filtered out
    (e.g., ATR mode with fixed stop %).
    
    Args:
        param_grid: Parameter grid to estimate
    
    Returns:
        Estimated number of valid combinations
    """
    # Generate a sample and see reduction ratio
    raw_count = count_combinations(param_grid)
    
    # Quick estimate based on known filters
    reduction_factor = 1.0
    
    # If both stop modes, params don't cross-apply (~50% reduction)
    if "stop_mode" in param_grid:
        modes = param_grid["stop_mode"]
        if len(modes) == 2:  # Both Fixed % and ATR
            reduction_factor *= 0.7
    
    # If trailing options, some become duplicates
    if "use_trailing" in param_grid and False in param_grid["use_trailing"]:
        if "trail_pct" in param_grid and len(param_grid["trail_pct"]) > 1:
            reduction_factor *= 0.85
    
    return int(raw_count * reduction_factor)
