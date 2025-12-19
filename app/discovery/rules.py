"""
Rule Discovery Module for Strategy Pattern Analysis.

This module analyzes winning strategies to discover common patterns
and rules that correlate with success. It provides:

- Statistical analysis of parameter distributions
- Comparison between winners and losers
- Automatic rule generation with confidence scores
- Pattern clustering and identification

The discovered rules help identify what parameter ranges and
combinations tend to produce winning strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import statistics
import math

from discovery.database import (
    DiscoveryDatabase,
    BacktestRun,
    DiscoveredRule,
)


# =============================================================================
# Analysis Helpers
# =============================================================================

def _safe_mean(values: List[float]) -> float:
    """Calculate mean, returning 0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: List[float]) -> float:
    """Calculate stdev, returning 0 for lists with < 2 elements."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _calculate_confidence(
    with_count: int,
    without_count: int,
    with_return: float,
    without_return: float,
) -> float:
    """
    Calculate confidence score for a discovered rule.
    
    Confidence is based on:
    - Sample size (more data = higher confidence)
    - Effect size (bigger difference = higher confidence)
    - Statistical significance approximation
    
    Returns:
        Confidence score between 0 and 1
    """
    # Minimum samples for any confidence
    if with_count < 5 or without_count < 5:
        return 0.0
    
    # Sample size factor (0.3-1.0 based on total samples)
    total = with_count + without_count
    sample_factor = min(1.0, 0.3 + (total / 500) * 0.7)
    
    # Effect size factor (0-1 based on return difference)
    if without_return == 0:
        effect_factor = 0.5 if with_return > 0 else 0.0
    else:
        # How much better is "with" compared to "without"
        improvement = (with_return - without_return) / max(abs(without_return), 1)
        effect_factor = min(1.0, max(0.0, improvement / 2 + 0.5))
    
    # Combine factors
    confidence = sample_factor * 0.4 + effect_factor * 0.6
    
    return round(confidence, 3)


# =============================================================================
# Pattern Discovery Functions
# =============================================================================

@dataclass
class ParameterStats:
    """Statistics for a parameter across winners and all runs."""
    parameter: str
    winner_values: List[Any] = field(default_factory=list)
    loser_values: List[Any] = field(default_factory=list)
    all_values: List[Any] = field(default_factory=list)
    winner_returns: Dict[Any, List[float]] = field(default_factory=dict)
    loser_returns: Dict[Any, List[float]] = field(default_factory=dict)


def analyze_parameter_distributions(
    runs: List[BacktestRun],
) -> Dict[str, ParameterStats]:
    """
    Analyze parameter distributions across all runs.
    
    Args:
        runs: List of backtest runs to analyze
    
    Returns:
        Dictionary mapping parameter names to ParameterStats
    """
    # Collect all parameter names from runs
    param_names = set()
    for run in runs:
        param_names.update(run.params.keys())
    
    # Initialize stats for each parameter
    stats: Dict[str, ParameterStats] = {
        name: ParameterStats(parameter=name) for name in param_names
    }
    
    # Populate stats from runs
    for run in runs:
        for param, value in run.params.items():
            if param not in stats:
                continue
            
            ps = stats[param]
            ps.all_values.append(value)
            
            if run.is_winner:
                ps.winner_values.append(value)
                if value not in ps.winner_returns:
                    ps.winner_returns[value] = []
                ps.winner_returns[value].append(run.total_return)
            else:
                ps.loser_values.append(value)
                if value not in ps.loser_returns:
                    ps.loser_returns[value] = []
                ps.loser_returns[value].append(run.total_return)
    
    return stats


def find_value_patterns(
    stats: ParameterStats,
    min_occurrence: int = 5,
) -> List[Tuple[Any, float, int]]:
    """
    Find which parameter values appear more frequently in winners.
    
    Args:
        stats: ParameterStats for a single parameter
        min_occurrence: Minimum occurrences to consider
    
    Returns:
        List of (value, winner_rate, count) tuples sorted by winner_rate
    """
    # Count occurrences of each value
    all_counts: Dict[Any, int] = defaultdict(int)
    winner_counts: Dict[Any, int] = defaultdict(int)
    
    for v in stats.all_values:
        all_counts[v] += 1
    for v in stats.winner_values:
        winner_counts[v] += 1
    
    # Calculate winner rate for each value
    patterns = []
    for value, total in all_counts.items():
        if total < min_occurrence:
            continue
        
        winners = winner_counts.get(value, 0)
        rate = winners / total if total > 0 else 0
        patterns.append((value, rate, total))
    
    # Sort by winner rate descending
    patterns.sort(key=lambda x: x[1], reverse=True)
    
    return patterns


def find_range_patterns(
    stats: ParameterStats,
    num_buckets: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find which parameter ranges correlate with winning.
    
    For numeric parameters, divides into buckets and analyzes
    winner distribution across ranges.
    
    Args:
        stats: ParameterStats for a single parameter
        num_buckets: Number of range buckets to create
    
    Returns:
        List of range pattern dictionaries
    """
    # Only works for numeric values
    numeric_values = [v for v in stats.all_values if isinstance(v, (int, float))]
    if len(numeric_values) < 10:
        return []
    
    # Calculate bucket boundaries
    min_val = min(numeric_values)
    max_val = max(numeric_values)
    
    if min_val == max_val:
        return []
    
    bucket_size = (max_val - min_val) / num_buckets
    
    # Assign values to buckets
    all_buckets: Dict[int, List[float]] = defaultdict(list)
    winner_buckets: Dict[int, List[float]] = defaultdict(list)
    
    for run_idx, value in enumerate(stats.all_values):
        if not isinstance(value, (int, float)):
            continue
        
        bucket = min(int((value - min_val) / bucket_size), num_buckets - 1)
        all_buckets[bucket].append(value)
    
    for value in stats.winner_values:
        if not isinstance(value, (int, float)):
            continue
        bucket = min(int((value - min_val) / bucket_size), num_buckets - 1)
        winner_buckets[bucket].append(value)
    
    # Calculate stats per bucket
    patterns = []
    for bucket in range(num_buckets):
        bucket_start = min_val + bucket * bucket_size
        bucket_end = bucket_start + bucket_size
        
        total = len(all_buckets[bucket])
        winners = len(winner_buckets[bucket])
        
        if total < 5:
            continue
        
        patterns.append({
            "bucket": bucket,
            "range_start": round(bucket_start, 2),
            "range_end": round(bucket_end, 2),
            "total_count": total,
            "winner_count": winners,
            "winner_rate": round(winners / total, 3) if total > 0 else 0,
        })
    
    return patterns


# =============================================================================
# Main Rule Discovery
# =============================================================================

def find_winning_patterns(
    db: DiscoveryDatabase,
    min_confidence: float = 0.3,
    min_occurrence_pct: float = 10.0,
) -> List[DiscoveredRule]:
    """
    Analyze winning strategies to find common patterns.
    
    This is the main entry point for rule discovery. It:
    1. Loads all backtest runs from the database
    2. Analyzes parameter distributions
    3. Identifies patterns that correlate with winning
    4. Creates and stores discovered rules
    
    Args:
        db: DiscoveryDatabase instance
        min_confidence: Minimum confidence for a rule to be stored
        min_occurrence_pct: Minimum % of winners with pattern
    
    Returns:
        List of DiscoveredRule objects
    
    Example:
        >>> db = DiscoveryDatabase("data/discovery.db")
        >>> rules = find_winning_patterns(db)
        >>> for rule in rules:
        ...     print(f"{rule.parameter}: {rule.description}")
    """
    # Load all runs
    all_runs = db.get_all_runs(limit=100000, winners_only=False)
    winners = [r for r in all_runs if r.is_winner]
    losers = [r for r in all_runs if not r.is_winner]
    
    if len(winners) < 10:
        # Not enough winners to analyze
        return []
    
    # Analyze parameter distributions
    stats = analyze_parameter_distributions(all_runs)
    
    discovered_rules: List[DiscoveredRule] = []
    
    # Clear existing rules before adding new ones
    db.clear_rules()
    
    # Analyze each parameter
    for param_name, param_stats in stats.items():
        # Skip metadata-like parameters
        if param_name in {"symbol", "timeframe", "start_date", "end_date"}:
            continue
        
        # Check if parameter is categorical or numeric
        sample_values = param_stats.all_values[:10]
        is_numeric = all(isinstance(v, (int, float)) for v in sample_values if v is not None)
        
        if is_numeric:
            rules = _discover_numeric_rules(
                param_name, 
                param_stats, 
                winners, 
                losers,
                min_confidence,
                min_occurrence_pct
            )
        else:
            rules = _discover_categorical_rules(
                param_name,
                param_stats,
                winners,
                losers,
                min_confidence,
                min_occurrence_pct
            )
        
        for rule in rules:
            db.save_rule(rule)
            discovered_rules.append(rule)
    
    # Sort by confidence
    discovered_rules.sort(key=lambda r: r.confidence, reverse=True)
    
    return discovered_rules


def _discover_numeric_rules(
    param_name: str,
    stats: ParameterStats,
    winners: List[BacktestRun],
    losers: List[BacktestRun],
    min_confidence: float,
    min_occurrence_pct: float,
) -> List[DiscoveredRule]:
    """Discover rules for numeric parameters."""
    rules = []
    
    # Get winner and loser values
    winner_values = [
        run.params.get(param_name) 
        for run in winners 
        if param_name in run.params and isinstance(run.params.get(param_name), (int, float))
    ]
    loser_values = [
        run.params.get(param_name)
        for run in losers
        if param_name in run.params and isinstance(run.params.get(param_name), (int, float))
    ]
    
    if len(winner_values) < 5:
        return rules
    
    # Calculate winner statistics
    winner_mean = _safe_mean(winner_values)
    winner_stdev = _safe_stdev(winner_values)
    loser_mean = _safe_mean(loser_values) if loser_values else winner_mean
    
    # Find optimal range (within 1 stdev of winner mean)
    if winner_stdev > 0:
        range_low = round(winner_mean - winner_stdev, 2)
        range_high = round(winner_mean + winner_stdev, 2)
    else:
        range_low = round(winner_mean * 0.9, 2)
        range_high = round(winner_mean * 1.1, 2)
    
    # Count winners in this range
    in_range = [v for v in winner_values if range_low <= v <= range_high]
    occurrence_pct = len(in_range) / len(winner_values) * 100 if winner_values else 0
    
    if occurrence_pct < min_occurrence_pct:
        return rules
    
    # Calculate returns for in-range vs out-of-range
    in_range_returns = [
        run.total_return for run in winners
        if param_name in run.params 
        and isinstance(run.params.get(param_name), (int, float))
        and range_low <= run.params[param_name] <= range_high
    ]
    out_range_returns = [
        run.total_return for run in winners
        if param_name in run.params
        and isinstance(run.params.get(param_name), (int, float))
        and not (range_low <= run.params[param_name] <= range_high)
    ]
    
    avg_return_with = _safe_mean(in_range_returns)
    avg_return_without = _safe_mean(out_range_returns) if out_range_returns else 0
    
    # Calculate confidence
    confidence = _calculate_confidence(
        len(in_range_returns),
        len(out_range_returns),
        avg_return_with,
        avg_return_without
    )
    
    if confidence < min_confidence:
        return rules
    
    # Create rule
    condition = f"{range_low}-{range_high}"
    description = (
        f"{param_name} in range {range_low}-{range_high} appears in "
        f"{occurrence_pct:.0f}% of winners (avg return: {avg_return_with:.2f}%)"
    )
    
    rule = DiscoveredRule(
        parameter=param_name,
        condition=condition,
        occurrence_pct=occurrence_pct,
        avg_return_with=avg_return_with,
        avg_return_without=avg_return_without,
        confidence=confidence,
        description=description,
    )
    
    rules.append(rule)
    
    return rules


def _discover_categorical_rules(
    param_name: str,
    stats: ParameterStats,
    winners: List[BacktestRun],
    losers: List[BacktestRun],
    min_confidence: float,
    min_occurrence_pct: float,
) -> List[DiscoveredRule]:
    """Discover rules for categorical parameters."""
    rules = []
    
    # Count value occurrences in winners vs losers
    winner_counts: Dict[Any, int] = defaultdict(int)
    loser_counts: Dict[Any, int] = defaultdict(int)
    winner_returns: Dict[Any, List[float]] = defaultdict(list)
    
    for run in winners:
        if param_name in run.params:
            value = run.params[param_name]
            winner_counts[value] += 1
            winner_returns[value].append(run.total_return)
    
    for run in losers:
        if param_name in run.params:
            value = run.params[param_name]
            loser_counts[value] += 1
    
    total_winners = len(winners)
    
    # Analyze each value
    for value, count in winner_counts.items():
        occurrence_pct = count / total_winners * 100 if total_winners > 0 else 0
        
        if occurrence_pct < min_occurrence_pct:
            continue
        
        # Calculate winner rate for this value
        total_with_value = count + loser_counts.get(value, 0)
        winner_rate = count / total_with_value if total_with_value > 0 else 0
        
        # Calculate returns
        avg_return_with = _safe_mean(winner_returns[value])
        other_returns = [
            run.total_return for run in winners
            if param_name in run.params and run.params[param_name] != value
        ]
        avg_return_without = _safe_mean(other_returns) if other_returns else 0
        
        # Confidence
        confidence = _calculate_confidence(
            count,
            total_winners - count,
            avg_return_with,
            avg_return_without
        )
        
        if confidence < min_confidence:
            continue
        
        # Create rule
        condition = str(value)
        description = (
            f"{param_name}={value} appears in {occurrence_pct:.0f}% of winners "
            f"(win rate: {winner_rate*100:.1f}%, avg return: {avg_return_with:.2f}%)"
        )
        
        rule = DiscoveredRule(
            parameter=param_name,
            condition=condition,
            occurrence_pct=occurrence_pct,
            avg_return_with=avg_return_with,
            avg_return_without=avg_return_without,
            confidence=confidence,
            description=description,
        )
        
        rules.append(rule)
    
    return rules


# =============================================================================
# Advanced Analysis
# =============================================================================

def find_parameter_interactions(
    db: DiscoveryDatabase,
    param1: str,
    param2: str,
) -> Dict[str, Any]:
    """
    Analyze how two parameters interact in winning strategies.
    
    Args:
        db: Database instance
        param1: First parameter name
        param2: Second parameter name
    
    Returns:
        Dictionary with interaction analysis
    """
    winners = db.get_winners(limit=10000)
    
    if len(winners) < 20:
        return {"error": "Not enough winners to analyze"}
    
    # Build interaction matrix
    interactions: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
    
    for run in winners:
        if param1 in run.params and param2 in run.params:
            key = (run.params[param1], run.params[param2])
            interactions[key].append(run.total_return)
    
    # Find best combinations
    best_combos = []
    for (v1, v2), returns in interactions.items():
        if len(returns) >= 3:  # Minimum samples
            best_combos.append({
                param1: v1,
                param2: v2,
                "count": len(returns),
                "avg_return": _safe_mean(returns),
                "best_return": max(returns),
            })
    
    # Sort by average return
    best_combos.sort(key=lambda x: x["avg_return"], reverse=True)
    
    return {
        "param1": param1,
        "param2": param2,
        "total_combinations": len(interactions),
        "best_combinations": best_combos[:10],
    }


def get_rule_summary(rules: List[DiscoveredRule]) -> str:
    """
    Generate a human-readable summary of discovered rules.
    
    Args:
        rules: List of discovered rules
    
    Returns:
        Formatted summary string
    """
    if not rules:
        return "No rules discovered yet. Run more discovery tests to find patterns."
    
    lines = [
        "=" * 60,
        "DISCOVERED WINNING PATTERNS",
        "=" * 60,
        "",
    ]
    
    # Group rules by confidence level
    high_conf = [r for r in rules if r.confidence >= 0.7]
    med_conf = [r for r in rules if 0.4 <= r.confidence < 0.7]
    low_conf = [r for r in rules if r.confidence < 0.4]
    
    if high_conf:
        lines.append("HIGH CONFIDENCE PATTERNS:")
        lines.append("-" * 40)
        for rule in high_conf[:5]:
            lines.append(f"  • {rule.description}")
            lines.append(f"    Confidence: {rule.confidence:.0%}")
        lines.append("")
    
    if med_conf:
        lines.append("MEDIUM CONFIDENCE PATTERNS:")
        lines.append("-" * 40)
        for rule in med_conf[:5]:
            lines.append(f"  • {rule.description}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"Total rules discovered: {len(rules)}")
    lines.append(f"High confidence: {len(high_conf)}")
    lines.append(f"Medium confidence: {len(med_conf)}")
    lines.append(f"Low confidence: {len(low_conf)}")
    
    return "\n".join(lines)
