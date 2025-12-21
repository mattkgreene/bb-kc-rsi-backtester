from __future__ import annotations

from typing import Any, Dict, List, Tuple
import multiprocessing as mp


def count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Calculate total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def estimate_discovery_time(
    param_grid: Dict[str, List[Any]],
    tested_count: int = 0,
    seconds_per_test: float = 0.1,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimate time required for a discovery run."""
    total = count_combinations(param_grid)
    remaining = max(0, total - tested_count)

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
        ),
    }


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    return mp.cpu_count()


def create_margin_discovery_grid(
    rsi_range: Tuple[int, int] = (68, 74),
    rsi_ma_range: Tuple[int, int] = (66, 72),
    band_mult_range: Tuple[float, float] = (1.9, 2.1),
    leverage_options: List[float] | None = None,
    risk_pct_options: List[float] | None = None,
    include_atr_stops: bool = True,
    include_trailing: bool = True,
    rsi_step: int = 2,
    mult_step: float = 0.1,
) -> Dict[str, List[Any]]:
    """Create a discovery grid that includes margin/futures parameters."""
    def arange(start, end, step):
        values = []
        current = start
        while current <= end + step / 2:
            values.append(round(current, 2) if isinstance(step, float) else int(current))
            current += step
        return values

    if leverage_options is None:
        leverage_options = [2.0, 5.0, 10.0]
    if risk_pct_options is None:
        risk_pct_options = [0.5, 1.0, 2.0]

    grid = {
        "rsi_min": arange(rsi_range[0], rsi_range[1], rsi_step),
        "rsi_ma_min": arange(rsi_ma_range[0], rsi_ma_range[1], rsi_step),
        "bb_std": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "kc_mult": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "entry_band_mode": ["Either", "KC", "Both"],
        "exit_level": ["mid", "lower"],
        "trade_mode": ["Margin / Futures"],
        "max_leverage": leverage_options,
        "risk_per_trade_pct": risk_pct_options,
        "maintenance_margin_pct": [0.5],
        "use_stop": [True],
        "stop_pct": [1.5, 2.0, 2.5],
    }

    if include_atr_stops:
        grid["stop_mode"] = ["Fixed %", "ATR"]
        grid["stop_atr_mult"] = [1.5, 2.0, 2.5]
    else:
        grid["stop_mode"] = ["Fixed %"]

    if include_trailing:
        grid["use_trailing"] = [True, False]
        grid["trail_pct"] = [1.0, 1.5, 2.0]
    else:
        grid["use_trailing"] = [False]

    return grid
