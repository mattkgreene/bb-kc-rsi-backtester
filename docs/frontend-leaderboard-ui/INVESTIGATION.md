> **Project Title:** Frontend Leaderboard UI
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Investigation Plan

## 1) Problem Statement
- **Context:** The frontend needs an easy, reliable way to show “winning configurations” (leaderboard) produced by background discovery/optimization jobs.
- **Desired Outcome:** Frontend reads the latest persisted leaderboard snapshot from the backend API and exposes refresh + “last updated” details to users.
- **Constraints/Assumptions:**
  - Leaderboard is stored as snapshots in the backend DB.
  - Backend exposes `GET /v1/leaderboard` for the most recent snapshot.

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Implement UI wiring + snapshot display |
| A | mkg | Approve UX + deploy |
| C | Backend | Ensure snapshot payload/metadata stored + served |
| I | Users | Browse and load winning strategies |

## 3) Hypotheses
- **H-001:** Returning snapshot metadata (`created_at`, `snapshot_id`) alongside the payload enables the UI to show freshness.
  - Evidence to confirm/refute: UI renders “Last updated …” from API response.
- **H-002:** Refreshing leaderboard via a job and then reloading the latest snapshot keeps the UI consistent with persisted state.
  - Evidence to confirm/refute: after refresh job completes, `/v1/leaderboard` returns a new snapshot id/time.

## 4) Questions to Answer
- **Q-001:** What fields must `GET /v1/leaderboard` return for the UI? → linked test(s): **T-001**
- **Q-002:** How should the UI refresh flow ensure it displays persisted data? → linked test(s): **T-002**

## 5) Investigation Tasks (test-first)
- **S-001:** Confirm backend snapshot persistence and latest snapshot read.
  - Expected signals: saving a snapshot then reading latest returns the most recent id/payload.
  - Test **T-001:** `python3 -m unittest test_backend_leaderboard_db.LeaderboardDbTests`
  - Artifacts: `docs/frontend-leaderboard-ui/artifacts/unittest_leaderboard_db.out`
  - Risk **R-001:** schema mismatch causes UI to break when parsing payload.
- **S-002:** Confirm frontend can compile with leaderboard UI changes.
  - Expected signals: `frontend/ui/dash_app.py` compiles.
  - Test **T-002:** `python3 -m py_compile frontend/ui/dash_app.py`
  - Artifacts: `docs/frontend-leaderboard-ui/artifacts/py_compile.out`
  - Risk **R-002:** UI attempts to refresh leaderboard on startup (unwanted load).

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Snapshot metadata + payload contract is defined and confirmed.
- **IC-2:** UI refresh flow reads persisted snapshot data.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Snapshot payload schema changes | Med | Med | Treat payload as dict and guard missing keys |
| R-002 | Unintended background work on UI startup | Med | Low | Only load snapshot when leaderboard tab active |

## 8) Decision Log (provisional)
- **D-001:** Display leaderboard based on backend snapshot API (not computed in UI).
  - Rationale: avoids heavy work in frontend and provides a single source of truth.
  - Alternatives considered: recompute leaderboard client-side (not scalable).

## 9) Exit Artifacts
- `docs/frontend-leaderboard-ui/artifacts/py_compile.out`
- `docs/frontend-leaderboard-ui/artifacts/unittest_leaderboard_db.out`
