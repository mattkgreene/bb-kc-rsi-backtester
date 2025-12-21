from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.db.pg import is_postgres_url, pg_conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class JobRow:
    id: int
    job_type: str
    status: str
    priority: int
    payload: dict[str, Any]
    result: Optional[dict[str, Any]]
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    heartbeat_at: Optional[str]
    worker_id: Optional[str]
    progress_current: int
    progress_total: int
    progress_message: Optional[str]


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


def init_jobs_db(db: DbRef) -> None:
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id BIGSERIAL PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT,
                    worker_id TEXT,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    progress_message TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    id BIGSERIAL PRIMARY KEY,
                    job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id)")
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error_text TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                heartbeat_at TEXT,
                worker_id TEXT,
                progress_current INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0,
                progress_message TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")


def enqueue_job(
    db: DbRef,
    *,
    job_type: str,
    payload: dict[str, Any],
    priority: int = 0,
) -> int:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO jobs (
                    job_type, status, priority, payload_json, created_at
                ) VALUES (%s, 'queued', %s, %s, %s)
                RETURNING id
                """,
                (str(job_type), int(priority), json.dumps(payload, default=str), _utc_now_iso()),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else 0

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO jobs (
                job_type, status, priority, payload_json, created_at
            ) VALUES (?, 'queued', ?, ?, ?)
            """,
            (job_type, int(priority), json.dumps(payload, default=str), _utc_now_iso()),
        )
        return int(cur.lastrowid)


def _row_to_job(row: Any) -> JobRow:
    return JobRow(
        id=int(row["id"]),
        job_type=str(row["job_type"]),
        status=str(row["status"]),
        priority=int(row["priority"]),
        payload=json.loads(row["payload_json"] or "{}"),
        result=(json.loads(row["result_json"]) if row["result_json"] else None),
        error=(str(row["error_text"]) if row["error_text"] else None),
        created_at=str(row["created_at"]),
        started_at=(str(row["started_at"]) if row["started_at"] else None),
        finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
        heartbeat_at=(str(row["heartbeat_at"]) if row["heartbeat_at"] else None),
        worker_id=(str(row["worker_id"]) if row["worker_id"] else None),
        progress_current=int(row["progress_current"] or 0),
        progress_total=int(row["progress_total"] or 0),
        progress_message=(str(row["progress_message"]) if row["progress_message"] else None),
    )


def get_job(db: DbRef, job_id: int) -> Optional[JobRow]:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE id = %s", (int(job_id),))
            row = cur.fetchone()
            return _row_to_job(row) if row else None

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
        return _row_to_job(row) if row else None


def get_job_status(db: DbRef, job_id: int) -> Optional[str]:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM jobs WHERE id = %s", (int(job_id),))
            row = cur.fetchone()
            return str(row["status"]) if row else None

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
        return str(row["status"]) if row else None


def is_job_canceled(db: DbRef, *, job_id: int) -> bool:
    return get_job_status(db, job_id) == "canceled"


def list_jobs(
    db: DbRef,
    *,
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[JobRow]:
    init_jobs_db(db)
    if _is_pg(db):
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = %s")
            params.append(status)
        if job_type:
            where.append("job_type = %s")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT * FROM jobs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([int(limit), int(offset)])
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [_row_to_job(r) for r in rows]

    db_path = _sqlite_path(db)
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if job_type:
        where.append("job_type = ?")
        params.append(job_type)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT * FROM jobs
        {where_sql}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([int(limit), int(offset)])
    with _get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_job(r) for r in rows]


def claim_next_job(db: DbRef, *, worker_id: str) -> Optional[JobRow]:
    """
    Atomically claim the next queued job.

    SQLite has limited concurrency primitives, but BEGIN IMMEDIATE ensures only
    one worker can claim at a time.
    """
    init_jobs_db(db)
    now = _utc_now_iso()

    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                WITH next AS (
                    SELECT id FROM jobs
                    WHERE status = 'queued'
                    ORDER BY priority DESC, created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE jobs j
                SET status = 'running',
                    started_at = %s,
                    heartbeat_at = %s,
                    worker_id = %s
                FROM next
                WHERE j.id = next.id
                RETURNING j.*
                """,
                (now, now, str(worker_id)),
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None

        job_id = int(row["id"])
        updated = conn.execute(
            """
            UPDATE jobs
            SET status = 'running',
                started_at = ?,
                heartbeat_at = ?,
                worker_id = ?
            WHERE id = ? AND status = 'queued'
            """,
            (now, now, worker_id, job_id),
        )
        if updated.rowcount != 1:
            return None

        claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(claimed) if claimed else None


def update_progress(
    db: DbRef,
    *,
    job_id: int,
    current: int,
    total: int,
    message: Optional[str] = None,
) -> None:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs
                SET progress_current = %s,
                    progress_total = %s,
                    progress_message = %s,
                    heartbeat_at = %s
                WHERE id = %s AND status = 'running'
                """,
                (int(current), int(total), message, _utc_now_iso(), int(job_id)),
            )
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET progress_current = ?,
                progress_total = ?,
                progress_message = ?,
                heartbeat_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (int(current), int(total), message, _utc_now_iso(), int(job_id)),
        )


def mark_succeeded(db: DbRef, *, job_id: int, result: dict[str, Any]) -> None:
    init_jobs_db(db)
    now = _utc_now_iso()
    payload = json.dumps(result, default=str)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    result_json = %s,
                    finished_at = %s,
                    heartbeat_at = %s,
                    error_text = NULL
                WHERE id = %s AND status = 'running'
                """,
                (payload, now, now, int(job_id)),
            )
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'succeeded',
                result_json = ?,
                finished_at = ?,
                heartbeat_at = ?,
                error_text = NULL
            WHERE id = ? AND status = 'running'
            """,
            (payload, now, now, int(job_id)),
        )


def mark_failed(db: DbRef, *, job_id: int, error: str) -> None:
    init_jobs_db(db)
    now = _utc_now_iso()
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_text = %s,
                    finished_at = %s,
                    heartbeat_at = %s
                WHERE id = %s AND status = 'running'
                """,
                (str(error), now, now, int(job_id)),
            )
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error_text = ?,
                finished_at = ?,
                heartbeat_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (error, now, now, int(job_id)),
        )


def mark_canceled(db: DbRef, *, job_id: int) -> None:
    init_jobs_db(db)
    now = _utc_now_iso()
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    finished_at = COALESCE(finished_at, %s),
                    heartbeat_at = %s
                WHERE id = %s AND status IN ('queued', 'running')
                """,
                (now, now, int(job_id)),
            )
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'canceled',
                finished_at = COALESCE(finished_at, ?),
                heartbeat_at = ?
            WHERE id = ? AND status IN ('queued', 'running')
            """,
            (now, now, int(job_id)),
        )


def append_event(
    db: DbRef,
    *,
    job_id: int,
    level: str,
    message: str,
) -> None:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO job_events (job_id, ts, level, message)
                VALUES (%s, %s, %s, %s)
                """,
                (int(job_id), _utc_now_iso(), str(level), str(message)),
            )
        return

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id)",
        )
        conn.execute(
            """
            INSERT INTO job_events (job_id, ts, level, message)
            VALUES (?, ?, ?, ?)
            """,
            (int(job_id), _utc_now_iso(), str(level), str(message)),
        )


def get_events(db: DbRef, *, job_id: int, limit: int = 200) -> list[dict[str, Any]]:
    init_jobs_db(db)
    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, ts, level, message
                FROM job_events
                WHERE job_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (int(job_id), int(limit)),
            )
            rows = cur.fetchall()
            return [
                {"id": int(r["id"]), "ts": str(r["ts"]), "level": str(r["level"]), "message": str(r["message"])}
                for r in rows
            ][::-1]

    db_path = _sqlite_path(db)
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        rows = conn.execute(
            """
            SELECT id, ts, level, message
            FROM job_events
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(job_id), int(limit)),
        ).fetchall()
        return [
            {"id": int(r["id"]), "ts": str(r["ts"]), "level": str(r["level"]), "message": str(r["message"])}
            for r in rows
        ][::-1]


def delete_jobs(
    db: DbRef,
    *,
    job_ids: Iterable[int],
) -> int:
    ids = [int(i) for i in job_ids]
    if not ids:
        return 0
    init_jobs_db(db)

    if _is_pg(db):
        with pg_conn(str(db)) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM jobs WHERE id = ANY(%s)", (ids,))
            return int(cur.rowcount or 0)

    db_path = _sqlite_path(db)
    marks = ",".join(["?"] * len(ids))
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        conn.execute(f"DELETE FROM job_events WHERE job_id IN ({marks})", ids)
        cur = conn.execute(f"DELETE FROM jobs WHERE id IN ({marks})", ids)
        return int(cur.rowcount or 0)
