from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.db.pg import is_postgres_url, pg_conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class LeaderboardSnapshotRow:
    id: int
    created_at: str
    sort_by: str
    min_trades: int
    top_n: int
    payload: dict[str, Any]


DbRef = str | Path


def _is_pg(db: DbRef) -> bool:
    return isinstance(db, str) and is_postgres_url(db)


def _sqlite_path(db: DbRef) -> Path:
    return db if isinstance(db, Path) else Path(db)


@contextmanager
def _get_conn(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
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


def init_leaderboard_db(db: DbRef) -> None:
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    sort_by TEXT NOT NULL,
                    min_trades INTEGER NOT NULL,
                    top_n INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_lb_created_at ON leaderboard_snapshots(created_at)")
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sort_by TEXT NOT NULL,
                min_trades INTEGER NOT NULL,
                top_n INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lb_created_at ON leaderboard_snapshots(created_at)")


def save_snapshot(
    db: DbRef,
    *,
    sort_by: str,
    min_trades: int,
    top_n: int,
    payload: dict[str, Any],
) -> int:
    init_leaderboard_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO leaderboard_snapshots (
                    created_at, sort_by, min_trades, top_n, payload_json
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (_utc_now_iso(), str(sort_by), int(min_trades), int(top_n), json.dumps(payload, default=str)),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else 0

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO leaderboard_snapshots (
                created_at, sort_by, min_trades, top_n, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (_utc_now_iso(), str(sort_by), int(min_trades), int(top_n), json.dumps(payload, default=str)),
        )
        return int(cur.lastrowid)


def _row_to_snapshot(row: Any) -> LeaderboardSnapshotRow:
    return LeaderboardSnapshotRow(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        sort_by=str(row["sort_by"]),
        min_trades=int(row["min_trades"]),
        top_n=int(row["top_n"]),
        payload=json.loads(row["payload_json"] or "{}"),
    )


def get_latest_snapshot(db: DbRef) -> Optional[LeaderboardSnapshotRow]:
    init_leaderboard_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM leaderboard_snapshots
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return _row_to_snapshot(row) if row else None

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM leaderboard_snapshots
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        return _row_to_snapshot(row) if row else None
