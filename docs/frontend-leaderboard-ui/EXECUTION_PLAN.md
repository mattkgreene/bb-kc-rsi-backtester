> **Project Title:** Frontend Leaderboard UI
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Execution Plan

## 1) Objective & Scope
- **Objective:** Show the latest leaderboard snapshot in the frontend and make refresh + freshness information visible.
- **Out of Scope:** Changing discovery/optimization scoring logic or snapshot schema redesign.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** “Leaderboard” tab loads the latest snapshot payload via backend API.
- **AC-2:** UI shows “Last updated” using `created_at` (and snapshot id when available).
- **AC-3:** Refresh action enqueues a backend refresh job and then reloads the latest snapshot on completion.

## 3) Preconditions & Dependencies
- Preconditions: Backend exposes `GET /v1/leaderboard` and `POST /v1/leaderboard/refresh`.
- Dependencies: Frontend configured with `BACKEND_URL`.

## 4) Step-by-Step Plan (each is testable)

### S-101: Load snapshot when leaderboard tab is active
- Goal: Avoid unnecessary backend calls on app startup.
- Commands/Actions:
  - Fetch snapshot only when `analysis-tabs == leaderboard`.
  - Store response in `store-lb-snapshot`.
- Expected Result: Snapshot data is available for table rendering without startup load.
- **Test T-101:** `python3 -m py_compile frontend/ui/dash_app.py`
- Evidence: `docs/frontend-leaderboard-ui/artifacts/py_compile.out`
- Rollback: Revert tab-gated snapshot loader; re-run T-101.

### S-102: Display freshness metadata (created_at, snapshot id)
- Goal: Make leaderboard currency visible to users.
- Commands/Actions:
  - Add `lb-updated` output and render from `store-lb-snapshot`.
- Expected Result: “Last updated …” displays when snapshot exists.
- **Test T-102:** `python3 -m py_compile frontend/ui/dash_app.py`
- Evidence: `docs/frontend-leaderboard-ui/artifacts/py_compile.out`
- Rollback: Remove `lb-updated` and callback; re-run T-102.

### S-103: Refresh flow uses persisted snapshot
- Goal: Ensure UI shows the DB-backed snapshot after refresh jobs complete.
- Commands/Actions:
  - On refresh job success, call `GET /v1/leaderboard` and store full response.
- Expected Result: Table renders from persisted snapshot payload and updated metadata.
- **Test T-103:** `python3 -m unittest test_backend_leaderboard_db.LeaderboardDbTests`
- Evidence: `docs/frontend-leaderboard-ui/artifacts/unittest_leaderboard_db.out`
- Rollback: Revert refresh-to-snapshot reload; re-run T-103.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Snapshot missing (404) | no snapshot yet | UI message prompting refresh | Codex |
| R-202 | Large payload slows UI | many winners | paginate/filter in DataTable | Codex |

## 6) Communication Plan
- Before: Ensure backend has produced at least one snapshot (or user clicks Refresh).
- After: Verify last updated text appears and refresh flow updates it.

## 7) Approval
- Change ticket: N/A
- Approvers: mkg
- Scheduled window: N/A
