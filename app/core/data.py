"""
Data fetching module for OHLCV market data.

This module provides functions to fetch historical candlestick (OHLCV) data
from various cryptocurrency exchanges using the CCXT library. It supports
paginated fetching for large date ranges and handles rate limiting.

Supported exchanges: coinbase, kraken, gemini, bitstamp, binanceus, and
any other exchange supported by CCXT.
"""

import time
import ccxt
import pandas as pd
import os
import re
import hashlib
import pickle
import gzip
from pathlib import Path
from typing import Optional, Tuple, List

from core.ohlcv_cache import get_cache_bounds, read_ohlcv_range, upsert_ohlcv, init_ohlcv_cache

def _tf_ms(tf: str) -> int:
    """
    Convert a timeframe string to milliseconds.
    
    Args:
        tf: Timeframe string (e.g., '30m', '1h', '4h', '1d', '1w').
            Supported suffixes: ms, s, m, h, d, w
    
    Returns:
        Duration of one candle in milliseconds.
    
    Examples:
        >>> _tf_ms('30m')
        1800000
        >>> _tf_ms('1h')
        3600000
        >>> _tf_ms('1d')
        86400000
    """
    tf = tf.strip().lower()
    if tf.endswith("ms"): return int(tf[:-2])
    if tf.endswith("s"):  return int(tf[:-1]) * 1000
    if tf.endswith("m"):  return int(tf[:-1]) * 60_000
    if tf.endswith("h"):  return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):  return int(tf[:-1]) * 86_400_000
    if tf.endswith("w"):  return int(tf[:-1]) * 7 * 86_400_000
    # default 30m
    return 1_800_000


def _project_root() -> Path:
    """
    Best-effort project root resolver.

    Assumes this file lives at app/core/data.py.
    """
    return Path(__file__).resolve().parents[2]


def _default_ohlcv_cache_dir() -> Path:
    """
    Default on-disk cache directory for OHLCV data.

    Stored under <project>/data/ohlcv_cache so it persists across Streamlit reruns
    and can be mounted as a volume in deployments.
    """
    return _project_root() / "data" / "ohlcv_cache"


def _safe_slug(s: str) -> str:
    """Filesystem-safe slug for exchange/symbol/timeframe identifiers."""
    s = str(s).strip()
    s = s.replace("/", "-").replace(":", "-")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "x"


def _ohlcv_cache_path(exchange: str, symbol: str, timeframe: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Return deterministic cache file path for a market series.

    We keep one file per (exchange, symbol, timeframe) and slice ranges from it.
    """
    base = cache_dir or _default_ohlcv_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    key = f"{_safe_slug(exchange)}__{_safe_slug(symbol)}__{_safe_slug(timeframe)}"
    return base / f"{key}.pkl.gz"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic file write (best-effort) via temp file + replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame shape and index."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLCV DataFrame must have a DatetimeIndex")

    # Ensure monotonic, de-duped, UTC-naive timestamps (exchange data is UTC)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def _infer_expected_freq(timeframe: str) -> pd.Timedelta:
    """Expected bar spacing for gap detection."""
    ms = _tf_ms(timeframe)
    return pd.to_timedelta(ms, unit="ms")


def _find_gaps(idx: pd.DatetimeIndex, timeframe: str, max_gaps: int = 25) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find large gaps in an OHLCV index and return missing windows.

    Returns list of (gap_start, gap_end) where data is missing in between.
    """
    if idx is None or len(idx) < 2:
        return []
    expected = _infer_expected_freq(timeframe)
    deltas = idx.to_series().diff()
    # Anything bigger than 1.5x expected frequency is considered a gap.
    gap_mask = deltas > (expected * 1.5)
    gaps = []
    gap_points = idx[gap_mask.values]
    for t in gap_points[:max_gaps]:
        prev = idx[idx.get_loc(t) - 1]
        # We fetch from prev+expected to t-expected (inclusive-ish)
        gaps.append((prev + expected, t - expected))
    # Filter tiny/invalid windows
    gaps = [(a, b) for (a, b) in gaps if a < b]
    return gaps


def fetch_ohlcv(exchange: str, symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    """
    Fetch a single page of OHLCV data from an exchange.
    
    This function retrieves the most recent candles up to the specified limit.
    For historical data over a specific date range, use fetch_ohlcv_range() instead.
    
    Args:
        exchange: Exchange identifier (e.g., 'bitstamp', 'kraken', 'coinbase').
                  Must be a valid CCXT exchange name.
        symbol: Trading pair symbol (e.g., 'BTC/USD', 'ETH/USDT').
        timeframe: Candle interval (e.g., '30m', '1h', '4h', '1d').
        limit: Maximum number of candles to fetch. Default is 1500.
               Actual limit depends on exchange capabilities.
    
    Returns:
        DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        and a DatetimeIndex named 'Time'.
    
    Raises:
        ccxt.ExchangeError: If the exchange request fails.
        ccxt.NetworkError: If there's a network connectivity issue.
        AttributeError: If the exchange name is invalid.
    
    Example:
        >>> df = fetch_ohlcv('bitstamp', 'BTC/USD', '1h', limit=100)
        >>> df.columns.tolist()
        ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    return _normalize_ohlcv_df(df)


def fetch_ohlcv_range(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    page_limit: int = 1000,
    sleep_mult: float = 1.0
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a specific date range with pagination.
    
    This function handles the pagination required to fetch large amounts of
    historical data. It respects exchange rate limits and implements retry
    logic for transient errors.
    
    Args:
        exchange: Exchange identifier (e.g., 'bitstamp', 'kraken', 'coinbase').
        symbol: Trading pair symbol (e.g., 'BTC/USD', 'ETH/USDT').
        timeframe: Candle interval (e.g., '30m', '1h', '4h', '1d').
        start_ts: Start timestamp for the data range (inclusive).
        end_ts: End timestamp for the data range (inclusive).
        page_limit: Maximum candles per request. Default 1000.
                    Will be capped at 1000 to respect exchange limits.
        sleep_mult: Multiplier for rate limit pause duration. Default 1.0.
                    Increase if experiencing rate limit errors.
    
    Returns:
        DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        and a DatetimeIndex named 'Time'. Data is trimmed to exact range
        and deduplicated.
    
    Raises:
        ccxt.ExchangeError: If exchange requests persistently fail.
        ccxt.NetworkError: If network issues persist after retry.
    
    Notes:
        - Uses exchange's rateLimit to avoid hitting API limits
        - Implements single retry on transient errors
        - Caps total data at 1,000,000 rows to prevent memory issues
        - Falls back to single-page fetch if pagination returns no data
    
    Example:
        >>> start = pd.Timestamp('2024-01-01')
        >>> end = pd.Timestamp('2024-01-31')
        >>> df = fetch_ohlcv_range('bitstamp', 'BTC/USD', '1h', start, end)
        >>> len(df)
        744  # ~31 days * 24 hours
    """
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tfms = _tf_ms(timeframe)

    # Treat naive timestamps as UTC (UI inputs are UTC dates).
    s = pd.Timestamp(start_ts)
    e = pd.Timestamp(end_ts)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")

    since = int(s.value // 1_000_000)   # ns -> ms
    end_ms = int(e.value // 1_000_000)  # ns -> ms
    out = []

    per_page = min(page_limit, 1000)
    # Respect rate limit - ensure minimum pause between requests
    pause = max(0.05, (ex.rateLimit or 250) / 1000.0) * sleep_mult

    last_guard = -1
    while since < end_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # Brief backoff and single retry on transient errors
            time.sleep(pause * 2)
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)

        if not batch:
            break

        out.extend(batch)

        # Advance cursor to next page
        latest = batch[-1][0]
        if latest <= last_guard:
            # Safety: if we're not making progress, advance by one candle
            since += tfms
        else:
            since = latest + tfms
            last_guard = latest

        # Stop if we've reached the end window
        if latest >= end_ms:
            break

        time.sleep(pause)

        # Memory safety: don't over-accumulate
        if len(out) > 1_000_000:
            break

    if not out:
        # Fallback: single recent page if pagination returned nothing
        data = fetch_ohlcv(exchange, symbol, timeframe, limit=page_limit)
        return data.loc[(data.index >= start_ts) & (data.index <= end_ts)]

    df = pd.DataFrame(out, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.drop_duplicates(subset=['Time'], inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    df = _normalize_ohlcv_df(df)
    # Trim to exact requested window (normalize to UTC-naive)
    s = pd.Timestamp(start_ts)
    e = pd.Timestamp(end_ts)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    else:
        s = s.tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    else:
        e = e.tz_localize(None)
    return df.loc[(df.index >= s) & (df.index <= e)]


def fetch_ohlcv_range_cached(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    page_limit: int = 1000,
    sleep_mult: float = 1.0,
    cache_dir: Optional[str | Path] = None,
    repair_gaps: bool = True,
    max_gap_repairs: int = 10,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a date range with a persistent on-disk cache.

    Why this exists:
    - CCXT range fetching is slow on cold start due to pagination + rate limiting.
    - This cache stores the full series per (exchange, symbol, timeframe) and only
      fetches missing edges (and optionally fills large internal gaps).

    Cache behavior:
    - One file per (exchange, symbol, timeframe).
    - When requesting a range, we will:
      1) Load cached data (if present)
      2) Fetch missing [start..cache_min) and/or (cache_max..end] edges
      3) Optionally repair large gaps inside the cached window
      4) Merge, de-dup, sort, and persist back to disk
      5) Return the requested slice
    """
    cache_path = _ohlcv_cache_path(exchange, symbol, timeframe, Path(cache_dir) if cache_dir else None)

    s = pd.Timestamp(start_ts)
    e = pd.Timestamp(end_ts)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    else:
        s = s.tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    else:
        e = e.tz_localize(None)

    cached: Optional[pd.DataFrame] = None
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path, compression="gzip")
            cached = _normalize_ohlcv_df(cached)
        except Exception:
            # Corrupt/old cache - ignore and rebuild
            cached = None

    pieces: List[pd.DataFrame] = []
    if cached is not None and not cached.empty:
        pieces.append(cached)

        cache_min = cached.index.min()
        cache_max = cached.index.max()

        # Fetch missing leading edge
        if s < cache_min:
            lead_end = min(e, cache_min)
            lead = fetch_ohlcv_range(
                exchange, symbol, timeframe,
                start_ts=s, end_ts=lead_end,
                page_limit=page_limit, sleep_mult=sleep_mult
            )
            if lead is not None and not lead.empty:
                pieces.append(lead)

        # Fetch missing trailing edge
        if e > cache_max:
            trail_start = max(s, cache_max)
            trail = fetch_ohlcv_range(
                exchange, symbol, timeframe,
                start_ts=trail_start, end_ts=e,
                page_limit=page_limit, sleep_mult=sleep_mult
            )
            if trail is not None and not trail.empty:
                pieces.append(trail)

        # Repair gaps within the overlapping window (optional, capped)
        if repair_gaps:
            window = cached.loc[(cached.index >= s) & (cached.index <= e)]
            gaps = _find_gaps(window.index, timeframe, max_gaps=max_gap_repairs)
            for g0, g1 in gaps:
                if g0 >= g1:
                    continue
                gap_df = fetch_ohlcv_range(
                    exchange, symbol, timeframe,
                    start_ts=g0, end_ts=g1,
                    page_limit=page_limit, sleep_mult=sleep_mult
                )
                if gap_df is not None and not gap_df.empty:
                    pieces.append(gap_df)
    else:
        # No cache yet - fetch full range and seed cache
        fresh = fetch_ohlcv_range(
            exchange, symbol, timeframe,
            start_ts=s, end_ts=e,
            page_limit=page_limit, sleep_mult=sleep_mult
        )
        if fresh is not None and not fresh.empty:
            pieces.append(fresh)

    if not pieces:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    merged = _normalize_ohlcv_df(pd.concat(pieces, axis=0, ignore_index=False))

    # Persist merged cache (store full series, not just requested slice)
    try:
        payload = gzip.compress(pickle.dumps(merged, protocol=pickle.HIGHEST_PROTOCOL))
        _atomic_write_bytes(cache_path, payload)
    except Exception:
        # Best-effort cache; never fail the fetch because disk write failed.
        pass

    return merged.loc[(merged.index >= s) & (merged.index <= e)]


def fetch_ohlcv_range_db_cached(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    page_limit: int = 1000,
    sleep_mult: float = 1.0,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a date range using a SQLite cache.

    Strategy:
    - Read bounds from cache.
    - Fetch missing leading/trailing edges via CCXT.
    - Upsert fetched data into cache.
    - Return cached slice for requested range.
    """
    init_ohlcv_cache(Path(db_path) if db_path else None)

    s = pd.Timestamp(start_ts)
    e = pd.Timestamp(end_ts)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)

    bounds = get_cache_bounds(exchange, symbol, timeframe, Path(db_path) if db_path else None)

    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if bounds.min_ts is None or bounds.max_ts is None:
        missing_ranges.append((s, e))
    else:
        if s < bounds.min_ts:
            missing_ranges.append((s, min(bounds.min_ts, e)))
        if e > bounds.max_ts:
            missing_ranges.append((max(bounds.max_ts, s), e))

    for m_start, m_end in missing_ranges:
        if m_start >= m_end:
            continue
        fetched = fetch_ohlcv_range(
            exchange,
            symbol,
            timeframe,
            start_ts=m_start,
            end_ts=m_end,
            page_limit=page_limit,
            sleep_mult=sleep_mult,
        )
        if fetched is None or fetched.empty:
            continue
        upsert_ohlcv(exchange, symbol, timeframe, fetched, Path(db_path) if db_path else None)

    return read_ohlcv_range(exchange, symbol, timeframe, s, e, Path(db_path) if db_path else None)
