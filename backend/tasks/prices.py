from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from backend.config import Settings
from backend.processing.spark import get_spark_session
from backend.features.market.service import MarketDataService


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
    page_limit = int(payload.get("page_limit") or 1000)
    sleep_mult = float(payload.get("sleep_mult") or 1.0)

    market = MarketDataService(settings.market_db_path)

    report_progress(0, 1, "Fetching + caching OHLCV")
    df = market.fetch_range(
        exchange,
        symbol,
        timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        page_limit=page_limit,
        sleep_mult=sleep_mult,
    )

    bounds = market.coverage_bounds(exchange, symbol, timeframe)
    result: dict[str, Any] = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested": {"start_ts": start_ts.isoformat(), "end_ts": end_ts.isoformat()},
        "cached_bounds": {
            "min_ts": (bounds.min_ts.isoformat() if bounds.min_ts is not None else None),
            "max_ts": (bounds.max_ts.isoformat() if bounds.max_ts is not None else None),
        },
        "rows_returned": int(len(df) if df is not None else 0),
        "market_db_path": str(settings.market_db_path),
    }

    if settings.use_spark and df is not None and not df.empty:
        spark = get_spark_session(app_name="bbkc-prices", master=settings.spark_master)
        if spark is None:
            log("warn", "USE_SPARK enabled but SparkSession unavailable; skipping Spark stats.")
        else:
            try:
                from pyspark.sql import functions as F  # type: ignore

                sdf = spark.createDataFrame(df.reset_index().rename(columns={"index": "ts"}))
                stats = sdf.agg(
                    F.count("*").alias("rows"),
                    F.min("ts").alias("min_ts"),
                    F.max("ts").alias("max_ts"),
                ).collect()[0]
                result["spark_stats"] = {
                    "rows": int(stats["rows"]),
                    "min_ts": str(stats["min_ts"]),
                    "max_ts": str(stats["max_ts"]),
                }
            except Exception as exc:
                log("warn", f"Spark stats failed: {exc}")

    report_progress(1, 1, "Ingestion complete")
    return result
