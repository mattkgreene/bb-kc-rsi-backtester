from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from backend.config import Settings
from backend.processing.spark import get_spark_session

from core.data import fetch_ohlcv_range_db_cached
from discovery.database import DiscoveryDatabase, WinCriteria
from discovery.engine import DiscoveryConfig, run_discovery, run_discovery_parallel


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    exchange = payload["exchange"]
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]
    start_ts = pd.Timestamp(payload["start_ts"])
    end_ts = pd.Timestamp(payload["end_ts"])

    report_progress(0, 1, "Loading OHLCV data")
    df = fetch_ohlcv_range_db_cached(
        exchange,
        symbol,
        timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        db_path=settings.market_db_path,
    )
    if df is None or df.empty:
        raise ValueError("No OHLCV data available for requested range")

    # Optional Spark: just validate we can create a session for larger workloads.
    if settings.use_spark:
        spark = get_spark_session(app_name="bbkc-discovery", master=settings.spark_master)
        if spark is None:
            log("warn", "USE_SPARK enabled but SparkSession unavailable; falling back to multiprocessing.")
        else:
            log("info", "SparkSession available (discovery still runs via multiprocessing in this version).")

    base_params = dict(payload.get("base_params") or {})
    base_params.update(
        {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
        }
    )

    win_criteria_dict = payload.get("win_criteria") or {}
    win_criteria = WinCriteria(
        min_total_return=float(win_criteria_dict.get("min_total_return", 0.0)),
        max_drawdown=float(win_criteria_dict.get("max_drawdown", 20.0)),
        min_trades=int(win_criteria_dict.get("min_trades", 10)),
        min_profit_factor=float(win_criteria_dict.get("min_profit_factor", 1.0)),
        min_win_rate=float(win_criteria_dict.get("min_win_rate", 0.0)),
    )

    cfg = DiscoveryConfig(
        param_grid=dict(payload.get("param_grid") or {}),
        win_criteria=win_criteria,
        skip_tested=bool(payload.get("skip_tested", True)),
        batch_size=int(payload.get("batch_size", 50)),
        max_combinations=(int(payload["max_combinations"]) if payload.get("max_combinations") is not None else None),
    )

    discovery_db = DiscoveryDatabase(str(settings.discovery_db_path))
    discovery_db.initialize()

    use_parallel = bool(payload.get("use_parallel", True))
    n_workers = payload.get("n_workers")

    def _progress(current: int, total: int, message: str):
        # Update at most ~100 times to keep DB writes modest.
        step = max(1, total // 100)
        if current == total or current % step == 0:
            report_progress(int(current), int(total), message)

    log("info", f"Starting discovery for {exchange} {symbol} {timeframe}")
    if use_parallel:
        result = run_discovery_parallel(
            df=df,
            base_params=base_params,
            db=discovery_db,
            config=cfg,
            n_workers=(int(n_workers) if n_workers is not None else None),
            progress_callback=_progress,
        )
    else:
        result = run_discovery(
            df=df,
            base_params=base_params,
            db=discovery_db,
            config=cfg,
            progress_callback=_progress,
        )

    report_progress(1, 1, "Discovery complete")
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "summary": {
            "total_tested": result.total_tested,
            "new_tested": result.new_tested,
            "winners_found": result.winners_found,
            "best_return": result.best_return,
            "best_params": result.best_params,
            "duration_seconds": result.duration_seconds,
        },
        "discovery_db_path": str(settings.discovery_db_path),
    }

