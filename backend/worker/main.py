from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from backend.config import get_settings
from backend.db.jobs import (
    append_event,
    claim_next_job,
    init_jobs_db,
    is_job_canceled,
    mark_failed,
    mark_succeeded,
    update_progress,
)


def _ensure_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app_dir = repo_root / "app"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))


def _dispatch(job_type: str):
    # Import lazily so sys.path bootstrapping happens before importing app/* modules.
    from backend.tasks import discovery, leaderboard, optimization, patterns, prices

    if job_type == "prices_ingest":
        return prices.run_job
    if job_type == "optimize":
        return optimization.run_job
    if job_type == "discover":
        return discovery.run_job
    if job_type == "leaderboard_refresh":
        return leaderboard.run_job
    if job_type == "patterns_refresh":
        return patterns.run_job
    return None


class JobCanceledError(RuntimeError):
    pass


def main() -> int:
    _ensure_import_paths()
    settings = get_settings()

    init_jobs_db(settings.backend_db_path)

    print(f"[worker] id={settings.worker_id} db={settings.backend_db_path}")
    poll = max(0.2, float(settings.worker_poll_seconds))

    while True:
        try:
            job = claim_next_job(settings.backend_db_path, worker_id=settings.worker_id)
            if job is None:
                time.sleep(poll)
                continue

            append_event(
                settings.backend_db_path,
                job_id=job.id,
                level="info",
                message=f"Claimed job type={job.job_type} worker_id={settings.worker_id}",
            )

            if is_job_canceled(settings.backend_db_path, job_id=job.id):
                append_event(
                    settings.backend_db_path,
                    job_id=job.id,
                    level="info",
                    message="Job was canceled before execution started",
                )
                continue

            handler = _dispatch(job.job_type)
            if handler is None:
                raise ValueError(f"Unknown job_type: {job.job_type}")

            def _progress(current: int, total: int, message: Optional[str] = None) -> None:
                update_progress(
                    settings.backend_db_path,
                    job_id=job.id,
                    current=current,
                    total=total,
                    message=message,
                )
                if is_job_canceled(settings.backend_db_path, job_id=job.id):
                    raise JobCanceledError("Job canceled")

            def _log(level: str, message: str) -> None:
                append_event(settings.backend_db_path, job_id=job.id, level=level, message=message)

            try:
                _log("info", "Starting execution")
                result = handler(job.payload, settings=settings, report_progress=_progress, log=_log)
                if is_job_canceled(settings.backend_db_path, job_id=job.id):
                    append_event(
                        settings.backend_db_path,
                        job_id=job.id,
                        level="info",
                        message="Job canceled; result discarded",
                    )
                    continue
                mark_succeeded(settings.backend_db_path, job_id=job.id, result=result)
                _log("info", "Completed successfully")
            except JobCanceledError:
                append_event(settings.backend_db_path, job_id=job.id, level="info", message="Job canceled")
                continue

        except KeyboardInterrupt:
            print("[worker] shutdown")
            return 0
        except Exception as exc:
            try:
                job_id = job.id if "job" in locals() and job is not None else None
                err = f"{exc}\n{traceback.format_exc()}"
                if job_id is not None:
                    if is_job_canceled(settings.backend_db_path, job_id=job_id):
                        append_event(
                            settings.backend_db_path,
                            job_id=job_id,
                            level="info",
                            message="Job canceled; skipping failure mark",
                        )
                    else:
                        append_event(settings.backend_db_path, job_id=job_id, level="error", message=str(exc))
                        mark_failed(settings.backend_db_path, job_id=job_id, error=err)
            except Exception:
                pass
            time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
