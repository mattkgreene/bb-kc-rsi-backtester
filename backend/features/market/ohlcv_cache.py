from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd


@dataclass
class CacheBounds:
    min_ts: Optional[pd.Timestamp]
    max_ts: Optional[pd.Timestamp]


def _default_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "market_data.db"


DbRef = str | Path


def _is_postgres(db: Optional[DbRef]) -> bool:
    if not isinstance(db, str):
        return False
    v = db.strip().lower()
    return v.startswith("postgres://") or v.startswith("postgresql://")


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLCV DataFrame must have a DatetimeIndex")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


@contextmanager
def _get_conn(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _get_pg_conn(database_url: str):
    try:
        import psycopg2  # type: ignore
        from psycopg2.extras import DictCursor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Postgres support requires psycopg2-binary") from exc

    conn = psycopg2.connect(str(database_url), cursor_factory=DictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_ohlcv_cache(db_path: Optional[DbRef] = None) -> DbRef:
    if _is_postgres(db_path):
        with _get_pg_conn(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts BIGINT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (exchange, symbol, timeframe, ts)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
                ON ohlcv_cache(exchange, symbol, timeframe, ts)
                """
            )
        return str(db_path)

    path = Path(db_path) if isinstance(db_path, str) else (db_path or _default_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (exchange, symbol, timeframe, ts)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
            ON ohlcv_cache(exchange, symbol, timeframe, ts)
            """
        )
    return path


def get_cache_bounds(
    exchange: str,
    symbol: str,
    timeframe: str,
    db_path: Optional[DbRef] = None,
) -> CacheBounds:
    if _is_postgres(db_path):
        init_ohlcv_cache(db_path)
        with _get_pg_conn(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts
                FROM ohlcv_cache
                WHERE exchange = %s AND symbol = %s AND timeframe = %s
                """,
                (exchange, symbol, timeframe),
            )
            row = cur.fetchone()
    else:
        path = init_ohlcv_cache(db_path)
        with _get_conn(Path(path)) as conn:
            row = conn.execute(
                """
                SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts
                FROM ohlcv_cache
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                """,
                (exchange, symbol, timeframe),
            ).fetchone()

    if row is None or row["min_ts"] is None:
        return CacheBounds(None, None)
    return CacheBounds(
        pd.to_datetime(row["min_ts"], unit="ms"),
        pd.to_datetime(row["max_ts"], unit="ms"),
    )


def upsert_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    db_path: Optional[DbRef] = None,
) -> int:
    normalized = _normalize_ohlcv_df(df)
    if normalized.empty:
        return 0

    rows = [
        (
            exchange,
            symbol,
            timeframe,
            int(ts.timestamp() * 1000),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
        )
        for ts, row in normalized.iterrows()
    ]

    if _is_postgres(db_path):
        init_ohlcv_cache(db_path)
        with _get_pg_conn(str(db_path)) as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO ohlcv_cache
                (exchange, symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (exchange, symbol, timeframe, ts)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                rows,
            )
        return len(rows)

    path = init_ohlcv_cache(db_path)
    with _get_conn(Path(path)) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_cache
            (exchange, symbol, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def read_ohlcv_range(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    db_path: Optional[DbRef] = None,
) -> pd.DataFrame:
    s = pd.Timestamp(start_ts)
    e = pd.Timestamp(end_ts)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)

    if _is_postgres(db_path):
        init_ohlcv_cache(db_path)
        with _get_pg_conn(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM ohlcv_cache
                WHERE exchange = %s AND symbol = %s AND timeframe = %s
                  AND ts BETWEEN %s AND %s
                ORDER BY ts ASC
                """,
                (
                    exchange,
                    symbol,
                    timeframe,
                    int(s.timestamp() * 1000),
                    int(e.timestamp() * 1000),
                ),
            )
            rows = cur.fetchall()
    else:
        path = init_ohlcv_cache(db_path)
        with _get_conn(Path(path)) as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM ohlcv_cache
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                  AND ts BETWEEN ? AND ?
                ORDER BY ts ASC
                """,
                (
                    exchange,
                    symbol,
                    timeframe,
                    int(s.timestamp() * 1000),
                    int(e.timestamp() * 1000),
                ),
            ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(
        rows,
        columns=["ts", "Open", "High", "Low", "Close", "Volume"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    return _normalize_ohlcv_df(df)


def get_coverage(
    *,
    limit: int = 500,
    db_path: Optional[DbRef] = None,
) -> list[dict[str, Any]]:
    """
    Summarize cached OHLCV coverage for display (exchange/symbol/timeframe bounds).
    """
    if _is_postgres(db_path):
        init_ohlcv_cache(db_path)
        with _get_pg_conn(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT exchange, symbol, timeframe,
                       MIN(ts) AS min_ts, MAX(ts) AS max_ts,
                       COUNT(*) AS rows_count
                FROM ohlcv_cache
                GROUP BY exchange, symbol, timeframe
                ORDER BY rows_count DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
            return [
                {
                    "exchange": str(r["exchange"]),
                    "symbol": str(r["symbol"]),
                    "timeframe": str(r["timeframe"]),
                    "min_ts_ms": (int(r["min_ts"]) if r["min_ts"] is not None else None),
                    "max_ts_ms": (int(r["max_ts"]) if r["max_ts"] is not None else None),
                    "rows": int(r["rows_count"] or 0),
                }
                for r in rows
            ]

    path = init_ohlcv_cache(db_path)
    with _get_conn(Path(path)) as conn:
        rows = conn.execute(
            """
            SELECT exchange, symbol, timeframe,
                   MIN(ts) AS min_ts, MAX(ts) AS max_ts,
                   COUNT(*) AS rows_count
            FROM ohlcv_cache
            GROUP BY exchange, symbol, timeframe
            ORDER BY rows_count DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [
            {
                "exchange": str(r["exchange"]),
                "symbol": str(r["symbol"]),
                "timeframe": str(r["timeframe"]),
                "min_ts_ms": (int(r["min_ts"]) if r["min_ts"] is not None else None),
                "max_ts_ms": (int(r["max_ts"]) if r["max_ts"] is not None else None),
                "rows": int(r["rows_count"] or 0),
            }
            for r in rows
        ]
