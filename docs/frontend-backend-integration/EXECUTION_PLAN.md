> **Project Title:** Frontend → Backend Job API Integration
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Execution Plan

## 1) Objective & Scope
- **Objective:** Move Dash UI code to `frontend/`, call backend jobs for optimization/discovery/leaderboard/patterns, and add unit tests for the integration layer.
- **Out of Scope:** Full UI refactor to remove local backtest execution and any auth/multi-tenant backend support.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** Optimization/discovery/leaderboard/patterns no longer run inline in Dash; UI uses backend jobs.
- **AC-2:** UI displays job progress and renders results when jobs succeed.
- **AC-3:** Unit tests cover queue DB + snapshot DB + frontend API client behavior.
- **Monitoring/Telemetry to validate:** `GET /health`, `GET /v1/jobs/{id}`, UI status text updates.

## 3) Preconditions & Dependencies
- Preconditions: backend API service reachable from frontend via `BACKEND_URL`.
- Dependencies: none required beyond stdlib for the frontend API client.
- Backout window: immediate | Freeze rules: none.

## 4) Step-by-Step Plan (each is testable)

### S-101: Move Dash code into `frontend/` with compatibility shims
- Goal: monorepo clarity while preserving legacy entrypoints.
- Commands/Actions: move `app/ui/dash_app.py` → `frontend/ui/dash_app.py`; add `app/ui/dash_app.py` shim.
- Expected Result: both `python frontend/ui/dash_app.py` and `python app/ui/dash_app.py` work.
- **Test T-101:** `python3 -m py_compile frontend/ui/dash_app.py app/ui/dash_app.py`
- Evidence: `docs/frontend-backend-integration/artifacts/py-compile.txt`
- Rollback: restore original file locations; revert docker entrypoint.

### S-102: Add a stdlib HTTP client for backend API calls
- Goal: frontend can enqueue jobs and poll status without new deps.
- Commands/Actions: implement `frontend/api_client.py`.
- Expected Result: `request_json()` handles success/errors; enqueue helpers return job dict.
- **Test T-102:** `python3 test_frontend_api_client.py` passes.
- Evidence: `docs/frontend-backend-integration/artifacts/unit-tests.txt`
- Rollback: revert to inline compute (previous callbacks) and remove client.

### S-103: Wire Dash callbacks to backend jobs + polling
- Goal: replace compute-heavy callbacks with enqueue + poll + render.
- Commands/Actions: update `frontend/ui/dash_app.py` callbacks and add stores/interval.
- Expected Result: UI shows queued/running progress; renders results from job `result`.
- **Test T-103:** compile check (T-101) + unit tests (T-102/T-104).
- Evidence: `docs/frontend-backend-integration/artifacts/py-compile.txt`
- Rollback: revert callback sections to previous synchronous behavior.

### S-104: Add unit tests for DB persistence helpers
- Goal: ensure queue and leaderboard snapshot DB helpers behave correctly.
- Commands/Actions: add `test_backend_jobs_db.py`, `test_backend_leaderboard_db.py`.
- Expected Result: tests pass without external deps.
- **Test T-104:** `python3 test_backend_jobs_db.py` and `python3 test_backend_leaderboard_db.py`.
- Evidence: `docs/frontend-backend-integration/artifacts/unit-tests.txt`
- Rollback: remove the new tests.

### S-105: Update docker/compose wiring for the new frontend package
- Goal: build/run frontend from `frontend/` while sharing `app/` as the compute library.
- Commands/Actions: update `frontend/Dockerfile` and `docker-compose.yml`.
- Expected Result: `ui` service runs `frontend/ui/dash_app.py` with `BACKEND_URL` configured.
- **Test T-105:** config inspection; (optional) `docker compose up`.
- Evidence: N/A (no docker build executed here).
- Rollback: restore prior Dockerfile entrypoint.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Duplicate Dash outputs break app startup | Dash error on load | Use `allow_duplicate=True` for poll callbacks only | Codex |
| R-202 | Backend outages degrade UI | API errors | Surface errors in status text; keep prior results in place | mkg |

## 6) Communication Plan
- Before: confirm required job payload shape per callback.
- During: deploy backend API + worker; point frontend `BACKEND_URL` to backend.
- After: consider moving backtest execution into backend as a follow-up.

## 7) Approval
- Change ticket: N/A
- Approvers: mkg
- Scheduled window: N/A

