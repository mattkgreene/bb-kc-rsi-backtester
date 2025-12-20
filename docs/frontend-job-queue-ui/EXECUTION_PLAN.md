> **Project Title:** Frontend Job Queue UI
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Execution Plan

## 1) Objective & Scope
- **Objective:** Add a “Job Queue” UI surface so the frontend can list backend background jobs, inspect details/events, and request cancellation.
- **Out of Scope:** Replacing the DB-backed queue implementation or adding a distributed queue (Redis/Celery).

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** Job list renders with filters (status/type/limit) and shows progress fields.
- **AC-2:** Selecting a job shows payload/result/error and recent events.
- **AC-3:** “Cancel Selected” calls backend cancel endpoint and job transitions to `canceled` (terminal).
- **Monitoring/Telemetry to validate:** UI status strings and job events for each action.

## 3) Preconditions & Dependencies
- Preconditions: Backend deployed with jobs endpoints enabled.
- Dependencies: `frontend/api_client.py` must reach `BACKEND_URL`.
- Backout window: Immediate (revert UI/client changes).

## 4) Step-by-Step Plan (each is testable)

### S-101: Add/confirm frontend API client queue methods
- Goal: Enable frontend to query the job queue via HTTP.
- Commands/Actions:
  - Ensure `frontend/api_client.py` includes `list_jobs`, `get_job_events`, `cancel_job`.
- Expected Result: UI can call backend for job list/detail/events/cancel.
- **Test T-101:** `python3 -m unittest test_frontend_api_client.ApiClientTests`
- Evidence: `docs/frontend-job-queue-ui/artifacts/unittest_api_client.out`
- Rollback: Revert `frontend/api_client.py`; re-run T-101.

### S-102: Add Job Queue tab layout + stores
- Goal: Provide user-facing queue controls and display components.
- Commands/Actions:
  - Add stores: `store-job-queue`, `store-job-selected`, `store-job-detail`, `store-job-events`.
  - Add “Job Queue” tab with filters, `jobs-table`, and detail/events panel.
- Expected Result: UI renders and contains required IDs for callbacks.
- **Test T-102:** `python3 -m py_compile frontend/ui/dash_app.py`
- Evidence: `docs/frontend-job-queue-ui/artifacts/py_compile.out`
- Rollback: Revert `frontend/ui/dash_app.py`; re-run T-102.

### S-103: Implement callbacks for list/detail/events/cancel
- Goal: Make queue interactive and efficient.
- Commands/Actions:
  - Refresh list only when tab active and auto-refresh enabled (or manual refresh).
  - Populate table, selection store, and detail/events rendering.
  - Cancel selected job via backend API and refresh details.
- Expected Result: User can view job status and request cancellation.
- **Test T-103:** `python3 -m py_compile frontend/ui/dash_app.py`
- Evidence: `docs/frontend-job-queue-ui/artifacts/py_compile.out`
- Rollback: Revert callback additions; re-run T-103.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Refresh storms | backend-poll triggers while tab inactive | Gate by `analysis-tabs == jobs` + auto-refresh toggle | Codex |
| R-202 | Cancel overridden by worker | status flips from canceled to succeeded/failed | Treat `canceled` as terminal in backend DB updates | Backend |

## 6) Communication Plan
- Before: Confirm `BACKEND_URL` in frontend env.
- During: Validate UI renders job list and details.
- After: Run unit tests and capture artifacts.

## 7) Approval
- Change ticket: N/A
- Approvers: mkg
- Scheduled window: N/A
