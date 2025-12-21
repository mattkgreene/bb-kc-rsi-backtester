"""
Robustness Scoring Module for Strategy Evaluation.

This module provides tools for evaluating whether a strategy is likely
to perform well in the long run, rather than just in-sample backtests.

Key Concepts:
1. OVERFITTING DETECTION: Strategies that are too perfectly tuned to 
   historical data often fail in live trading.
   
2. STABILITY SCORING: Strategies that perform consistently across different
   time periods and market conditions are more reliable.
   
3. PARAMETER SENSITIVITY: Strategies that only work with very specific
   parameters are fragile. Robust strategies work across a range.

4. OUT-OF-SAMPLE VALIDATION: The gold standard - performance on data
   the strategy wasn't optimized on.

Why This Matters:
Most "winning" strategies from backtests fail in live trading because they're
overfit to the specific historical period. This module helps identify strategies
that are genuinely robust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import math
from datetime import datetime


# =============================================================================
# Robustness Metrics
# =============================================================================

@dataclass
class RobustnessScore:
    """
    Comprehensive robustness score for a strategy.
    
    Attributes:
        overall_score: 0-100, higher = more robust
        consistency_score: How consistent are returns across time
        stability_score: How stable is performance across periods
        regime_score: Performance across different market regimes
        parameter_score: Sensitivity to parameter changes
        oos_score: Out-of-sample performance ratio
        warnings: List of potential issues
    """
    overall_score: float
    consistency_score: float
    stability_score: float
    regime_score: float
    parameter_score: float
    oos_score: float
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_robust(self, threshold: float = 60) -> bool:
        """Returns True if overall score meets threshold."""
        return self.overall_score >= threshold
    
    def get_grade(self) -> str:
        """Returns letter grade for the strategy."""
        score = self.overall_score
        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"


def calculate_consistency_score(
    trades: pd.DataFrame,
    equity_curve: np.ndarray,
    min_trades_per_period: int = 5
) -> Tuple[float, Dict]:
    """
    Calculate consistency of returns over time.
    
    A consistent strategy has similar performance across different
    time periods rather than making all profits in one lucky stretch.
    
    Args:
        trades: DataFrame with trade records
        equity_curve: Array of equity values
        min_trades_per_period: Minimum trades per period for analysis
    
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if trades is None or trades.empty or len(trades) < min_trades_per_period:
        return 0.0, {"error": "Not enough trades"}
    
    # Get returns per trade
    if 'RealizedPnL' in trades.columns and 'EquityBefore' in trades.columns:
        returns = trades['RealizedPnL'] / trades['EquityBefore'].replace(0, np.nan)
    elif 'ReturnPct' in trades.columns:
        returns = trades['ReturnPct']
    else:
        return 50.0, {"error": "No return data available"}
    
    returns = returns.dropna()
    
    if len(returns) < 10:
        return 50.0, {"warning": "Too few trades for consistency analysis"}
    
    # Split into periods (roughly equal chunks)
    n_periods = min(5, len(returns) // min_trades_per_period)
    if n_periods < 2:
        return 50.0, {"warning": "Not enough periods for analysis"}
    
    period_size = len(returns) // n_periods
    period_returns = []
    
    for i in range(n_periods):
        start = i * period_size
        end = start + period_size if i < n_periods - 1 else len(returns)
        period_ret = returns.iloc[start:end].sum()
        period_returns.append(period_ret)
    
    period_returns = np.array(period_returns)
    
    # Calculate metrics
    mean_return = np.mean(period_returns)
    std_return = np.std(period_returns)
    
    # Count profitable periods
    profitable_periods = (period_returns > 0).sum()
    pct_profitable = profitable_periods / n_periods * 100
    
    # Coefficient of variation (lower = more consistent)
    if abs(mean_return) > 0.001:
        cv = abs(std_return / mean_return)
    else:
        cv = float('inf') if std_return > 0 else 0
    
    # Score based on profitable periods and consistency
    # Ideal: high % profitable periods and low CV
    period_score = pct_profitable
    consistency_bonus = max(0, 50 - cv * 25)  # Lower CV = higher bonus
    
    score = min(100, period_score * 0.6 + consistency_bonus * 0.8)
    
    # Check for "one good period" problem
    if n_periods >= 3:
        sorted_returns = sorted(period_returns, reverse=True)
        top_period_contribution = sorted_returns[0] / max(sum(sorted_returns), 0.0001)
        if top_period_contribution > 0.7:
            score *= 0.7  # Penalize if one period dominates
    
    details = {
        "periods_analyzed": n_periods,
        "profitable_periods": profitable_periods,
        "pct_profitable": pct_profitable,
        "period_returns": period_returns.tolist(),
        "coefficient_of_variation": cv if not math.isinf(cv) else None,
    }
    
    return max(0, min(100, score)), details


def calculate_stability_score(
    equity_curve: np.ndarray,
    trades: pd.DataFrame
) -> Tuple[float, Dict]:
    """
    Calculate stability score based on equity curve smoothness.
    
    A stable strategy has smooth equity growth without large drawdowns
    or wild swings.
    
    Args:
        equity_curve: Array of equity values
        trades: DataFrame with trade records
    
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if equity_curve is None or len(equity_curve) < 10:
        return 50.0, {"error": "Not enough equity data"}
    
    equity = pd.Series(equity_curve)
    
    # Calculate maximum drawdown
    running_max = equity.expanding().max()
    drawdown = (running_max - equity) / running_max * 100
    max_dd = drawdown.max()
    
    # Calculate average drawdown
    avg_dd = drawdown.mean()
    
    # Calculate equity curve smoothness (linear regression R²)
    x = np.arange(len(equity))
    if len(equity) > 1:
        correlation = np.corrcoef(x, equity)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
    else:
        r_squared = 0
    
    # Calculate underwater periods
    underwater = drawdown > 0
    underwater_pct = underwater.mean() * 100
    
    # Score components
    dd_score = max(0, 100 - max_dd * 3)  # Penalize high drawdown
    smoothness_score = r_squared * 100
    underwater_score = max(0, 100 - underwater_pct)
    
    # Weighted combination
    score = dd_score * 0.4 + smoothness_score * 0.4 + underwater_score * 0.2
    
    details = {
        "max_drawdown_pct": max_dd,
        "avg_drawdown_pct": avg_dd,
        "r_squared": r_squared,
        "underwater_pct": underwater_pct,
    }
    
    return max(0, min(100, score)), details


def calculate_regime_score(
    trades: pd.DataFrame,
    regime_data: Optional[pd.DataFrame] = None
) -> Tuple[float, Dict]:
    """
    Calculate performance consistency across different market regimes.
    
    A robust strategy performs reasonably in all conditions, not just
    one specific type of market.
    
    Args:
        trades: DataFrame with trade records
        regime_data: DataFrame with regime classifications
    
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if trades is None or trades.empty:
        return 50.0, {"error": "No trades to analyze"}
    
    # If no regime data provided, use proxy based on trade outcomes
    if regime_data is None:
        # Analyze win rate across different periods as a proxy
        if 'EntryBar' in trades.columns:
            # Split by entry time
            trades = trades.sort_values('EntryBar')
            n_trades = len(trades)
            
            if n_trades < 10:
                return 50.0, {"warning": "Too few trades for regime analysis"}
            
            # Split into "regimes" by time quartiles
            quartiles = pd.qcut(trades['EntryBar'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Calculate win rate per quartile
            if 'RealizedPnL' in trades.columns:
                trades_copy = trades.copy()
                trades_copy['quartile'] = quartiles
                wins_by_q = trades_copy.groupby('quartile')['RealizedPnL'].apply(
                    lambda x: (x > 0).sum() / len(x) * 100
                )
            else:
                return 50.0, {"warning": "No PnL data for regime analysis"}
            
            # Score based on consistency across quartiles
            win_rates = wins_by_q.values
            mean_wr = np.mean(win_rates)
            std_wr = np.std(win_rates)
            min_wr = np.min(win_rates)
            
            # Penalize if any quartile has very low win rate
            min_penalty = max(0, 40 - min_wr) * 1.5
            
            # Penalize high variance
            variance_penalty = std_wr * 2
            
            score = max(0, 100 - min_penalty - variance_penalty)
            
            details = {
                "win_rate_by_quartile": dict(zip(['Q1', 'Q2', 'Q3', 'Q4'], win_rates.tolist())),
                "mean_win_rate": mean_wr,
                "std_win_rate": std_wr,
                "min_win_rate": min_wr,
            }
            
            return max(0, min(100, score)), details
    
    return 50.0, {"note": "Regime analysis requires regime data"}


def calculate_parameter_sensitivity(
    base_performance: Dict[str, float],
    neighbor_performances: List[Dict[str, float]],
    key_metric: str = "total_return"
) -> Tuple[float, Dict]:
    """
    Calculate how sensitive the strategy is to parameter changes.
    
    A robust strategy performs similarly with slightly different parameters.
    A fragile strategy only works with very specific settings.
    
    Args:
        base_performance: Performance metrics of the base strategy
        neighbor_performances: Performance metrics of nearby parameter combos
        key_metric: Metric to compare
    
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if not neighbor_performances:
        return 50.0, {"warning": "No neighbor data for sensitivity analysis"}
    
    base_value = base_performance.get(key_metric, 0)
    neighbor_values = [p.get(key_metric, 0) for p in neighbor_performances]
    
    if not neighbor_values:
        return 50.0, {"warning": "No valid neighbor performances"}
    
    # Calculate how different neighbors are from base
    differences = [abs(nv - base_value) for nv in neighbor_values]
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    # Calculate coefficient of variation across all (base + neighbors)
    all_values = [base_value] + neighbor_values
    mean_perf = np.mean(all_values)
    std_perf = np.std(all_values)
    
    if abs(mean_perf) > 0.001:
        cv = std_perf / abs(mean_perf)
    else:
        cv = float('inf') if std_perf > 0 else 0
    
    # Check if base is outlier (best by far)
    if base_value > 0:
        sorted_values = sorted(all_values, reverse=True)
        base_rank = sorted_values.index(base_value) + 1
        is_outlier_best = base_rank == 1 and (base_value > np.mean(neighbor_values) * 1.5)
    else:
        is_outlier_best = False
    
    # Score: lower variance = more robust
    # Penalize if base is outlier (likely overfit)
    base_score = max(0, 100 - cv * 50)
    
    if is_outlier_best:
        base_score *= 0.6  # Heavy penalty for being an outlier
    
    # Count neighbors that are also profitable (if base is)
    if base_value > 0:
        profitable_neighbors = sum(1 for nv in neighbor_values if nv > 0)
        neighbor_quality = profitable_neighbors / len(neighbor_values) * 100
        base_score = base_score * 0.6 + neighbor_quality * 0.4
    
    details = {
        "base_performance": base_value,
        "mean_neighbor_performance": np.mean(neighbor_values),
        "std_neighbor_performance": std_perf,
        "coefficient_of_variation": cv if not math.isinf(cv) else None,
        "is_outlier": is_outlier_best,
        "neighbors_analyzed": len(neighbor_performances),
    }
    
    return max(0, min(100, base_score)), details


def calculate_oos_score(
    in_sample_return: float,
    out_of_sample_return: float,
    in_sample_sharpe: float = 0,
    out_of_sample_sharpe: float = 0
) -> Tuple[float, Dict]:
    """
    Calculate out-of-sample performance score.
    
    The best test of a strategy is performance on data it wasn't optimized on.
    A big drop from in-sample to out-of-sample indicates overfitting.
    
    Args:
        in_sample_return: Return % on training data
        out_of_sample_return: Return % on test data
        in_sample_sharpe: Sharpe ratio on training data
        out_of_sample_sharpe: Sharpe ratio on test data
    
    Returns:
        Tuple of (score 0-100, details dict)
    """
    # Calculate degradation
    if in_sample_return > 0:
        return_ratio = out_of_sample_return / in_sample_return
    elif in_sample_return < 0:
        # If IS was negative and OOS is less negative or positive, that's good
        return_ratio = 1 if out_of_sample_return >= in_sample_return else 0.5
    else:
        return_ratio = 1 if out_of_sample_return >= 0 else 0.5
    
    # Sharpe degradation
    if in_sample_sharpe > 0:
        sharpe_ratio = out_of_sample_sharpe / in_sample_sharpe
    else:
        sharpe_ratio = 1
    
    # Base score on return ratio
    if return_ratio >= 0.8:
        return_score = 100  # OOS is 80%+ of IS
    elif return_ratio >= 0.6:
        return_score = 80   # OOS is 60-80% of IS
    elif return_ratio >= 0.4:
        return_score = 60   # OOS is 40-60% of IS
    elif return_ratio >= 0.2:
        return_score = 40   # OOS is 20-40% of IS
    elif return_ratio > 0:
        return_score = 20   # OOS positive but much lower
    else:
        return_score = 0    # OOS negative when IS positive
    
    # Bonus/penalty for OOS profitability
    if out_of_sample_return > 0:
        return_score = min(100, return_score + 10)
    elif out_of_sample_return < -5:
        return_score = max(0, return_score - 20)
    
    # Consider Sharpe ratio degradation
    sharpe_score = max(0, min(100, sharpe_ratio * 100))
    
    # Weighted score
    score = return_score * 0.7 + sharpe_score * 0.3
    
    details = {
        "in_sample_return": in_sample_return,
        "out_of_sample_return": out_of_sample_return,
        "return_ratio": return_ratio,
        "in_sample_sharpe": in_sample_sharpe,
        "out_of_sample_sharpe": out_of_sample_sharpe,
        "sharpe_ratio": sharpe_ratio,
    }
    
    return max(0, min(100, score)), details


# =============================================================================
# Comprehensive Robustness Evaluation
# =============================================================================

def evaluate_robustness(
    trades: pd.DataFrame,
    equity_curve: np.ndarray,
    in_sample_stats: Dict[str, float],
    out_of_sample_stats: Optional[Dict[str, float]] = None,
    neighbor_performances: Optional[List[Dict[str, float]]] = None,
    regime_data: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None
) -> RobustnessScore:
    """
    Perform comprehensive robustness evaluation of a strategy.
    
    This is the main function to call to get a full robustness assessment.
    
    Args:
        trades: DataFrame with trade records
        equity_curve: Array of equity values
        in_sample_stats: Performance stats from backtest
        out_of_sample_stats: Optional OOS performance stats
        neighbor_performances: Optional nearby parameter performances
        regime_data: Optional market regime data
        weights: Custom weights for each component
    
    Returns:
        RobustnessScore object with overall and component scores
    
    Example:
        >>> score = evaluate_robustness(trades, equity_curve, stats)
        >>> print(f"Overall: {score.overall_score:.1f} ({score.get_grade()})")
        >>> if not score.is_robust():
        ...     print("Warnings:", score.warnings)
    """
    # Default weights
    if weights is None:
        weights = {
            "consistency": 0.25,
            "stability": 0.25,
            "regime": 0.15,
            "parameter": 0.15,
            "oos": 0.20,
        }
    
    warnings = []
    details = {}
    
    # 1. Consistency Score
    consistency_score, consistency_details = calculate_consistency_score(trades, equity_curve)
    details["consistency"] = consistency_details
    
    # 2. Stability Score
    stability_score, stability_details = calculate_stability_score(equity_curve, trades)
    details["stability"] = stability_details
    
    if stability_details.get("max_drawdown_pct", 0) > 30:
        warnings.append(f"High max drawdown: {stability_details['max_drawdown_pct']:.1f}%")
    
    # 3. Regime Score
    regime_score, regime_details = calculate_regime_score(trades, regime_data)
    details["regime"] = regime_details
    
    # 4. Parameter Sensitivity Score
    if neighbor_performances:
        parameter_score, param_details = calculate_parameter_sensitivity(
            in_sample_stats, neighbor_performances
        )
        details["parameter"] = param_details
        
        if param_details.get("is_outlier", False):
            warnings.append("Strategy appears to be an outlier - likely overfit")
    else:
        parameter_score = 50  # Neutral if no data
        details["parameter"] = {"warning": "No neighbor data provided"}
    
    # 5. Out-of-Sample Score
    if out_of_sample_stats:
        oos_score, oos_details = calculate_oos_score(
            in_sample_stats.get("total_equity_return_pct", 0),
            out_of_sample_stats.get("total_equity_return_pct", 0),
            in_sample_stats.get("sharpe_ratio", 0),
            out_of_sample_stats.get("sharpe_ratio", 0)
        )
        details["oos"] = oos_details
        
        if oos_details.get("return_ratio", 1) < 0.5:
            warnings.append("Large performance degradation in out-of-sample test")
    else:
        oos_score = 50  # Neutral if no OOS data
        details["oos"] = {"warning": "No out-of-sample data provided"}
        warnings.append("No out-of-sample validation performed")
    
    # Calculate weighted overall score
    overall_score = (
        consistency_score * weights["consistency"] +
        stability_score * weights["stability"] +
        regime_score * weights["regime"] +
        parameter_score * weights["parameter"] +
        oos_score * weights["oos"]
    )
    
    # Additional warnings based on metrics
    if trades is not None and len(trades) < 30:
        warnings.append(f"Low trade count ({len(trades)}) - results may not be statistically significant")
    
    in_sample_return = in_sample_stats.get("total_equity_return_pct", 0)
    if in_sample_return > 100:
        warnings.append(f"Very high return ({in_sample_return:.1f}%) - verify not overfit")
    
    return RobustnessScore(
        overall_score=overall_score,
        consistency_score=consistency_score,
        stability_score=stability_score,
        regime_score=regime_score,
        parameter_score=parameter_score,
        oos_score=oos_score,
        warnings=warnings,
        details=details
    )


# =============================================================================
# Quick Checks
# =============================================================================

def quick_robustness_check(
    num_trades: int,
    win_rate: float,
    profit_factor: float,
    max_drawdown: float,
    total_return: float
) -> Tuple[str, List[str]]:
    """
    Quick robustness check based on key metrics.
    
    Returns a grade (A-F) and list of concerns.
    
    Args:
        num_trades: Total number of trades
        win_rate: Win rate percentage (e.g., 55.0)
        profit_factor: Profit factor
        max_drawdown: Maximum drawdown percentage
        total_return: Total return percentage
    
    Returns:
        Tuple of (grade, list of concerns)
    """
    concerns = []
    score = 0
    
    # Trade count
    if num_trades < 20:
        concerns.append(f"Very low trade count ({num_trades}) - not statistically significant")
    elif num_trades < 50:
        concerns.append(f"Low trade count ({num_trades}) - use caution")
        score += 5
    elif num_trades >= 100:
        score += 15
    else:
        score += 10
    
    # Win rate
    if win_rate < 30:
        concerns.append(f"Very low win rate ({win_rate:.1f}%)")
    elif win_rate < 40:
        concerns.append(f"Low win rate ({win_rate:.1f}%) - needs high reward/risk")
        score += 5
    elif win_rate > 80:
        concerns.append(f"Suspiciously high win rate ({win_rate:.1f}%) - verify not overfit")
        score += 10
    elif win_rate >= 50:
        score += 20
    else:
        score += 10
    
    # Profit factor
    if profit_factor < 1.0:
        concerns.append(f"Profit factor < 1.0 ({profit_factor:.2f}) - strategy loses money")
    elif profit_factor < 1.2:
        concerns.append(f"Low profit factor ({profit_factor:.2f}) - minimal edge")
        score += 5
    elif profit_factor > 5.0:
        concerns.append(f"Very high profit factor ({profit_factor:.2f}) - likely overfit")
        score += 10
    elif profit_factor >= 1.5:
        score += 25
    else:
        score += 15
    
    # Max drawdown
    if max_drawdown > 50:
        concerns.append(f"Extreme drawdown ({max_drawdown:.1f}%) - high risk of ruin")
    elif max_drawdown > 30:
        concerns.append(f"High drawdown ({max_drawdown:.1f}%) - difficult to recover")
        score += 5
    elif max_drawdown > 20:
        score += 10
    elif max_drawdown <= 10:
        score += 25
    else:
        score += 15
    
    # Total return vs drawdown (simple Calmar check)
    if max_drawdown > 0:
        calmar = total_return / max_drawdown
        if calmar < 0.5:
            concerns.append(f"Poor risk-adjusted return (Calmar: {calmar:.2f})")
        elif calmar >= 2.0:
            score += 15
        elif calmar >= 1.0:
            score += 10
    
    # Grade based on score
    if score >= 75:
        grade = "A"
    elif score >= 60:
        grade = "B"
    elif score >= 45:
        grade = "C"
    elif score >= 30:
        grade = "D"
    else:
        grade = "F"
    
    return grade, concerns


# =============================================================================
# Overfitting Detection
# =============================================================================

def detect_overfitting_signals(
    in_sample_stats: Dict[str, float],
    out_of_sample_stats: Optional[Dict[str, float]] = None,
    num_parameters_optimized: int = 0,
    num_trades: int = 0
) -> List[str]:
    """
    Detect common signs of overfitting in a strategy.
    
    Returns a list of warning signs found.
    
    Args:
        in_sample_stats: In-sample performance stats
        out_of_sample_stats: Out-of-sample stats (if available)
        num_parameters_optimized: Number of parameters that were optimized
        num_trades: Total number of trades
    
    Returns:
        List of overfitting warning strings
    """
    warnings = []
    
    is_return = in_sample_stats.get("total_equity_return_pct", 0)
    is_pf = in_sample_stats.get("profit_factor", 0)
    is_wr = in_sample_stats.get("win_rate", 0)
    is_sharpe = in_sample_stats.get("sharpe_ratio", 0)
    
    # 1. Unrealistically good performance
    if is_return > 200:
        warnings.append("In-sample return >200% is suspicious")
    if is_pf > 5.0 and is_pf != float('inf'):
        warnings.append("Profit factor >5.0 is unusually high")
    if is_wr > 85:
        warnings.append("Win rate >85% is suspicious for this strategy type")
    if is_sharpe > 5.0:
        warnings.append("Sharpe ratio >5.0 is unrealistic for live trading")
    
    # 2. Low trade count with good results
    if num_trades < 30 and is_return > 50:
        warnings.append(f"High return ({is_return:.1f}%) with only {num_trades} trades - likely noise")
    
    # 3. Too many parameters relative to trades
    if num_parameters_optimized > 0 and num_trades > 0:
        trades_per_param = num_trades / num_parameters_optimized
        if trades_per_param < 10:
            warnings.append(f"Only {trades_per_param:.1f} trades per optimized parameter - high overfit risk")
    
    # 4. OOS degradation
    if out_of_sample_stats:
        oos_return = out_of_sample_stats.get("total_equity_return_pct", 0)
        oos_pf = out_of_sample_stats.get("profit_factor", 0)
        
        if is_return > 0 and oos_return < 0:
            warnings.append("Strategy profitable in-sample but loses money out-of-sample")
        
        if is_return > 0:
            degradation = (is_return - oos_return) / is_return
            if degradation > 0.5:
                warnings.append(f"Performance dropped {degradation*100:.1f}% from IS to OOS")
        
        if is_pf > 1.5 and oos_pf < 1.0:
            warnings.append("Profit factor dropped below 1.0 in out-of-sample")
    
    return warnings
