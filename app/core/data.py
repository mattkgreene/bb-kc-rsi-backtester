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
    ex = getattr(ccxt, exchange)()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    return df


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
    ex = getattr(ccxt, exchange)()
    tfms = _tf_ms(timeframe)
    since = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
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
    # Trim to exact requested window
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df
