from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any, Iterator


def is_postgres_url(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v.startswith("postgres://") or v.startswith("postgresql://")


@contextmanager
def pg_conn(database_url: str):
    """
    Best-effort Postgres connection context manager.

    Import psycopg2 lazily so the codebase can be imported without Postgres deps
    when SQLite is used.
    """
    try:
        import psycopg2  # type: ignore
        from psycopg2.extras import DictCursor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Postgres support requires psycopg2-binary") from exc

    retries = int(os.getenv("PG_CONNECT_RETRIES", "8"))
    delay_seconds = float(os.getenv("PG_CONNECT_DELAY_SECONDS", "1.0"))
    connect_timeout = int(os.getenv("PG_CONNECT_TIMEOUT", "5"))

    last_exc: Exception | None = None
    conn = None
    for attempt in range(max(1, retries)):
        try:
            conn = psycopg2.connect(
                str(database_url),
                cursor_factory=DictCursor,
                connect_timeout=connect_timeout,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, retries) - 1:
                raise
            time.sleep(delay_seconds)
    if conn is None and last_exc is not None:
        raise last_exc
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def pg_execute(conn: Any, sql: str, params: tuple[Any, ...] | list[Any] | None = None):
    cur = conn.cursor()
    cur.execute(sql, params or ())
    return cur
