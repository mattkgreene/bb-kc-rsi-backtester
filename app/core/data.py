import time
import ccxt
import pandas as pd

def _tf_ms(tf: str) -> int:
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
    ex = getattr(ccxt, exchange)()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Time','Open','High','Low','Close','Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    return df

def fetch_ohlcv_range(exchange: str, symbol: str, timeframe: str,
                      start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                      page_limit: int = 1000, sleep_mult: float = 1.0) -> pd.DataFrame:
    ex = getattr(ccxt, exchange)()
    tfms = _tf_ms(timeframe)
    since = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    out = []

    per_page = min(page_limit, 1000)
    # Respect rate limit
    pause = max(0.05, (ex.rateLimit or 250) / 1000.0) * sleep_mult

    last_guard = -1
    while since < end_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # brief backoff and retry
            time.sleep(pause * 2)
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)

        if not batch:
            break

        out.extend(batch)

        latest = batch[-1][0]
        if latest <= last_guard:
            since += tfms
        else:
            since = latest + tfms
            last_guard = latest

        # stop if we’ve reached the end window
        if latest >= end_ms:
            break

        time.sleep(pause)

        # don’t over-accumulate
        if len(out) > 1_000_000:
            break

    if not out:
        # Fallback: single recent page
        data = fetch_ohlcv(exchange, symbol, timeframe, limit=page_limit)
        return data.loc[(data.index >= start_ts) & (data.index <= end_ts)]

    df = pd.DataFrame(out, columns=['Time','Open','High','Low','Close','Volume'])
    df.drop_duplicates(subset=['Time'], inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    # Trim to exact window
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df
