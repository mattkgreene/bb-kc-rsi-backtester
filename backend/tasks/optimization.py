from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from backend.config import Settings
from backend.processing.spark import get_spark_session

from core.data import fetch_ohlcv_range_db_cached
from optimization.grid_search import (
    ResultConstraints,
    WalkForwardConfig,
    run_grid_search,
    run_walk_forward_grid_search,
)


def _df_to_records(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"columns": [], "records": []}
    out = df.copy()
    for col in out.columns:
        # Ensure JSON-serializable values.
        out[col] = out[col].apply(lambda v: v.item() if hasattr(v, "item") else v)
    return {"columns": list(out.columns), "records": out.to_dict("records")}


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

    if settings.use_spark:
        spark = get_spark_session(app_name="bbkc-optimization", master=settings.spark_master)
        if spark is None:
            log("warn", "USE_SPARK enabled but SparkSession unavailable; falling back to pandas.")
        else:
            log("info", "SparkSession available (optimization runs in pandas in this version).")

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

    param_grid = dict(payload.get("param_grid") or {})
    metric = str(payload.get("metric") or "profit_factor")
    min_trades = int(payload.get("min_trades") or 5)
    top_n = int(payload.get("top_n") or 20)

    constraints_dict = payload.get("constraints") or {}
    constraints = ResultConstraints(
        min_trades=int(constraints_dict.get("min_trades", min_trades)),
        min_win_rate=(float(constraints_dict["min_win_rate"]) if constraints_dict.get("min_win_rate") is not None else None),
        min_profit_factor=(float(constraints_dict["min_profit_factor"]) if constraints_dict.get("min_profit_factor") is not None else None),
        max_drawdown=(float(constraints_dict["max_drawdown"]) if constraints_dict.get("max_drawdown") is not None else None),
        min_total_return=(float(constraints_dict["min_total_return"]) if constraints_dict.get("min_total_return") is not None else None),
    )

    validation_mode = str(payload.get("validation_mode") or "in_sample")

    if validation_mode == "walk":
        wf_dict = payload.get("walk_forward") or {}
        wf_cfg = WalkForwardConfig(
            train_days=int(wf_dict.get("train_days", 180)),
            test_days=int(wf_dict.get("test_days", 30)),
            step_days=(int(wf_dict["step_days"]) if wf_dict.get("step_days") is not None else None),
            max_folds=(int(wf_dict["max_folds"]) if wf_dict.get("max_folds") is not None else None),
        )

        def _wf_progress(current: int, total: int):
            report_progress(int(current), int(total), f"Walk-forward {current}/{total}")

        fold_results, wf_summary = run_walk_forward_grid_search(
            df=df,
            param_grid=param_grid,
            base_params=base_params,
            metric=metric,
            constraints=constraints,
            wf=wf_cfg,
            top_n_train=int(payload.get("top_n_train") or 20),
            progress_callback=_wf_progress,
        )

        report_progress(1, 1, "Optimization complete")
        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "kind": "walk_forward",
            "summary": wf_summary,
            "results": _df_to_records(fold_results),
        }

    def _progress(current: int, total: int):
        step = max(1, total // 100)
        if current == total or current % step == 0:
            report_progress(int(current), int(total), f"Grid search {current}/{total}")

    log("info", f"Starting optimization for {exchange} {symbol} {timeframe}")
    results = run_grid_search(
        df=df,
        param_grid=param_grid,
        base_params=base_params,
        metric=metric,
        min_trades=min_trades,
        constraints=constraints,
        top_n=top_n,
        progress_callback=_progress,
    )

    report_progress(1, 1, "Optimization complete")
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "kind": "grid_search",
        "metric": metric,
        "results": _df_to_records(results),
    }

