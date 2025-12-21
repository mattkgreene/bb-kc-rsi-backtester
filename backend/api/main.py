from __future__ import annotations

import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.cache import get_redis_client, redis_get_json, redis_set_json
from backend.config import get_settings
from backend.db.jobs import (
    append_event,
    enqueue_job,
    get_events,
    get_job,
    init_jobs_db,
    list_jobs,
    mark_canceled,
)
from backend.db.leaderboard import get_latest_snapshot, init_leaderboard_db

from backend.db.pg import is_postgres_url, pg_conn
from backend.features.backtest.service import BacktestService
from backend.features.market.service import MarketDataService

from backend.api.models import (
    BacktestRequest,
    BacktestResponse,
    DiscoveryStatsResponse,
    EnqueueResponse,
    JobCreateRequest,
    JobDetail,
    JobEventsResponse,
    JobsListResponse,
    JobSummary,
    LeaderboardResponse,
    PatternsResponse,
    PricesCoverageResponse,
)


def _ensure_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_import_paths()


def _job_to_summary(job) -> JobSummary:
    return JobSummary(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        heartbeat_at=job.heartbeat_at,
        worker_id=job.worker_id,
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        progress_message=job.progress_message,
    )


def _job_to_detail(job) -> JobDetail:
    return JobDetail(
        **_job_to_summary(job).model_dump(),
        payload=job.payload,
        result=job.result,
        error=job.error,
    )


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_jobs_db(settings.backend_db_path)
    init_leaderboard_db(settings.backend_db_path)
    app.state.redis = get_redis_client(settings.redis_url)
    yield
    try:
        if getattr(app.state, "redis", None) is not None:
            app.state.redis.close()
    except Exception:
        pass


app = FastAPI(title="BBKC Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/jobs", response_model=EnqueueResponse)
def create_job(req: JobCreateRequest):
    job_id = enqueue_job(
        settings.backend_db_path,
        job_type=req.job_type,
        payload=req.payload,
        priority=req.priority,
    )
    append_event(settings.backend_db_path, job_id=job_id, level="info", message="Job enqueued")
    job = get_job(settings.backend_db_path, job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to load newly created job")
    return EnqueueResponse(job=_job_to_detail(job))


@app.get("/v1/jobs", response_model=JobsListResponse)
def jobs_list(
    status: Optional[str] = Query(default=None),
    job_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    jobs = list_jobs(settings.backend_db_path, status=status, job_type=job_type, limit=limit, offset=offset)
    return JobsListResponse(jobs=[_job_to_summary(j) for j in jobs])


@app.get("/v1/jobs/{job_id}", response_model=JobDetail)
def job_detail(job_id: int):
    job = get_job(settings.backend_db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_detail(job)


@app.get("/v1/jobs/{job_id}/events", response_model=JobEventsResponse)
def job_events(job_id: int, limit: int = Query(default=200, ge=1, le=500)):
    job = get_job(settings.backend_db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobEventsResponse(job_id=job_id, events=get_events(settings.backend_db_path, job_id=job_id, limit=limit))


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobDetail)
def cancel_job(job_id: int):
    job = get_job(settings.backend_db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    mark_canceled(settings.backend_db_path, job_id=job_id)
    append_event(settings.backend_db_path, job_id=job_id, level="info", message="Cancel requested")
    job = get_job(settings.backend_db_path, job_id)
    return _job_to_detail(job)


@app.post("/v1/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    try:
        service = BacktestService(MarketDataService(settings.market_db_path))
        results = service.run(req.params)
        return BacktestResponse(**results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/prices/ingest", response_model=EnqueueResponse)
def prices_ingest(payload: dict):
    req = JobCreateRequest(job_type="prices_ingest", payload=payload)
    return create_job(req)


@app.get("/v1/prices/coverage", response_model=PricesCoverageResponse)
def prices_coverage(limit: int = Query(default=500, ge=1, le=5000)):
    db_ref = settings.market_db_path
    if isinstance(db_ref, str) and is_postgres_url(db_ref):
        with pg_conn(db_ref) as conn:
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
    else:
        db_path = Path(db_ref) if isinstance(db_ref, str) else db_ref
        if not db_path.exists():
            return PricesCoverageResponse(coverage=[])

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
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
        finally:
            conn.close()

    coverage = []
    for r in rows:
        coverage.append(
            {
                "exchange": r["exchange"],
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "min_ts_ms": int(r["min_ts"]) if r["min_ts"] is not None else None,
                "max_ts_ms": int(r["max_ts"]) if r["max_ts"] is not None else None,
                "rows": int(r["rows_count"]),
            }
        )
    return PricesCoverageResponse(coverage=coverage)


@app.post("/v1/optimize", response_model=EnqueueResponse)
def optimize(payload: dict):
    req = JobCreateRequest(job_type="optimize", payload=payload)
    return create_job(req)


@app.post("/v1/discover", response_model=EnqueueResponse)
def discover(payload: dict):
    req = JobCreateRequest(job_type="discover", payload=payload)
    return create_job(req)


@app.post("/v1/leaderboard/refresh", response_model=EnqueueResponse)
def leaderboard_refresh(payload: Optional[dict] = None):
    req = JobCreateRequest(job_type="leaderboard_refresh", payload=(payload or {}))
    return create_job(req)


@app.get("/v1/leaderboard", response_model=LeaderboardResponse)
def leaderboard_latest():
    cached = redis_get_json(getattr(app.state, "redis", None), "leaderboard:latest")
    if isinstance(cached, dict) and cached.get("payload") is not None:
        return cached

    snapshot = get_latest_snapshot(settings.backend_db_path)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No leaderboard snapshot available yet")

    resp = LeaderboardResponse(snapshot_id=snapshot.id, created_at=snapshot.created_at, payload=snapshot.payload).model_dump()
    redis_set_json(getattr(app.state, "redis", None), "leaderboard:latest", resp, ttl_seconds=10)
    return resp


@app.post("/v1/patterns/refresh", response_model=EnqueueResponse)
def patterns_refresh(payload: Optional[dict] = None):
    req = JobCreateRequest(job_type="patterns_refresh", payload=(payload or {}))
    return create_job(req)


@app.get("/v1/patterns", response_model=PatternsResponse)
def patterns(min_confidence: float = Query(default=0.3, ge=0.0, le=1.0)):
    cache_key = f"patterns:min_confidence={float(min_confidence):.4f}"
    cached = redis_get_json(getattr(app.state, "redis", None), cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("rules"), list):
        return cached

    from backend.features.discovery.database import DiscoveryDatabase

    db = DiscoveryDatabase(str(settings.discovery_db_path))
    db.initialize()
    rules = db.get_rules(min_confidence=float(min_confidence))
    resp = PatternsResponse(
        rules=[
            {
                "rule_id": r.rule_id,
                "parameter": r.parameter,
                "condition": r.condition,
                "occurrence_pct": r.occurrence_pct,
                "avg_return_with": r.avg_return_with,
                "avg_return_without": r.avg_return_without,
                "confidence": r.confidence,
                "description": r.description,
                "discovered_at": r.discovered_at,
            }
            for r in rules
        ]
    )
    redis_set_json(getattr(app.state, "redis", None), cache_key, resp.model_dump(), ttl_seconds=60)
    return resp


@app.get("/v1/discovery/stats", response_model=DiscoveryStatsResponse)
def discovery_stats():
    cached = redis_get_json(getattr(app.state, "redis", None), "discovery:stats")
    if isinstance(cached, dict) and cached.get("stats") is not None:
        return cached

    from backend.features.discovery.database import DiscoveryDatabase

    db = DiscoveryDatabase(str(settings.discovery_db_path))
    db.initialize()
    resp = DiscoveryStatsResponse(stats=db.get_statistics()).model_dump()
    redis_set_json(getattr(app.state, "redis", None), "discovery:stats", resp, ttl_seconds=30)
    return resp
