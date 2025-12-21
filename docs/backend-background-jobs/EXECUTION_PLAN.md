> **Project Title:** Backend Background Jobs (API + Queue)
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Execution Plan

## 1) Objective & Scope
- **Objective:** Add a backend API + durable queue + worker processes for optimization, discovery, leaderboard refresh, pattern analysis, and price ingestion.
- **Out of Scope:** Full frontend polling/UX integration and auth/multi-tenant isolation.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** API can enqueue jobs and return job status/events.
- **AC-2:** Worker can claim queued jobs and persist results/progress to DB.
- **AC-3:** Leaderboard snapshots and discovered patterns are retrievable from DB.
- **Monitoring/Telemetry to validate:** `GET /health`, job status progression, worker event logs.

## 3) Preconditions & Dependencies
- Preconditions: Python 3.11, write access to `data/`.
- Dependencies: `fastapi`, `uvicorn`, optional `pyspark`.
- Backout window: immediate | Freeze rules: none.

## 4) Step-by-Step Plan (each is testable)

### S-101: Add job DB schema + helpers
- Goal: Durable queue with progress and results.
- Commands/Actions: implement `backend/db/jobs.py`.
- Expected Result: enqueue/claim/update/succeed/fail supported.
- **Test T-101:** run DB unit test (T-201 in log).
- Evidence: `docs/backend-background-jobs/artifacts/test-backend-jobs-db.txt`
- Rollback: remove `backend/db/jobs.py` and related callers.

### S-102: Add worker loop + job handlers
- Goal: Execute jobs out-of-process, update DB as it runs.
- Commands/Actions: implement `backend/worker/main.py` and `backend/tasks/*`.
- Expected Result: worker claims jobs and writes result JSON.
- **Test T-102:** enqueue a job and run worker locally (manual) or unit-test job claiming.
- Evidence: `docs/backend-background-jobs/artifacts/test-backend-jobs-db.txt`
- Rollback: disable worker service.

### S-103: Add FastAPI service to enqueue + query jobs
- Goal: Frontend calls API, not compute code.
- Commands/Actions: implement `backend/api/main.py`.
- Expected Result: endpoints exist for creating jobs and reading status/events.
- **Test T-103:** `py_compile` for API modules; optional runtime smoke.
- Evidence: `docs/backend-background-jobs/artifacts/py-compile.txt`
- Rollback: remove backend API files.

### S-104: Add monorepo Docker layout for Railway deployments
- Goal: Separate frontend/backend dockerfiles for independent services.
- Commands/Actions: add `frontend/Dockerfile`, `backend/Dockerfile`, `backend/Dockerfile.worker`, update `docker-compose.yml`.
- Expected Result: local compose can run `ui`, `backend`, and `worker`.
- **Test T-104:** build with Docker (manual) or config inspection.
- Evidence: N/A (no Docker build executed here).
- Rollback: restore previous compose + railway config.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Worker can’t import `app/*` modules | ImportError in worker | Ensure `PYTHONPATH` + bootstrap path injection | Codex |
| R-202 | Spark dependency causes runtime failures | Java missing | Keep Spark optional; log + fallback | mkg |

## 6) Communication Plan
- Before: confirm API contract + job payload format with frontend.
- During: incremental integration via `/v1/jobs/*`.
- After: deploy backend API + worker as separate services; point UI to backend URL.

## 7) Approval
- Change ticket: N/A
- Approvers: mkg
- Scheduled window: N/A

