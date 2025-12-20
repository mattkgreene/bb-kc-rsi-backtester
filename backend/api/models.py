from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


JobType = Literal[
    "prices_ingest",
    "optimize",
    "discover",
    "leaderboard_refresh",
    "patterns_refresh",
]


class JobCreateRequest(BaseModel):
    job_type: JobType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class JobSummary(BaseModel):
    id: int
    job_type: str
    status: str
    priority: int
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    worker_id: Optional[str] = None
    progress_current: int = 0
    progress_total: int = 0
    progress_message: Optional[str] = None


class JobDetail(JobSummary):
    payload: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class EnqueueResponse(BaseModel):
    job: JobDetail


class JobsListResponse(BaseModel):
    jobs: list[JobSummary]


class JobEventsResponse(BaseModel):
    job_id: int
    events: list[dict[str, Any]]


class LeaderboardResponse(BaseModel):
    snapshot_id: int
    created_at: str
    payload: dict[str, Any]


class PatternsResponse(BaseModel):
    rules: list[dict[str, Any]]


class PricesCoverageResponse(BaseModel):
    coverage: list[dict[str, Any]]


class DiscoveryStatsResponse(BaseModel):
    stats: dict[str, Any]

