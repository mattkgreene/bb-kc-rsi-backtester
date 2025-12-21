from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import fetch_ohlcv_range_db_cached
from .ohlcv_cache import CacheBounds, DbRef, get_cache_bounds


@dataclass(frozen=True)
class MarketDataService:
    db_ref: DbRef

    def fetch_range(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        *,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        page_limit: int = 1000,
        sleep_mult: float = 1.0,
    ) -> pd.DataFrame:
        return fetch_ohlcv_range_db_cached(
            exchange,
            symbol,
            timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            page_limit=page_limit,
            sleep_mult=sleep_mult,
            db_path=self.db_ref,
        )

    def coverage_bounds(self, exchange: str, symbol: str, timeframe: str) -> CacheBounds:
        return get_cache_bounds(exchange, symbol, timeframe, self.db_ref)
