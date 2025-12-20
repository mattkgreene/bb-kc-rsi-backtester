> **Project Title:** Frontend → Backend Job API Integration
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Investigation Plan

## 1) Problem Statement
- **Context:** The Dash UI currently runs optimization, strategy discovery, leaderboard generation, and pattern recognition inline, blocking the UI and duplicating backend capabilities.
- **Desired Outcome:** The Dash UI triggers those workloads via backend API jobs and polls for completion; Dash UI code lives under `frontend/` for monorepo clarity.
- **Constraints/Assumptions:**
  - Backend provides durable jobs and DB persistence (already implemented).
  - Frontend may run without non-stdlib HTTP dependencies.
  - Existing `python app/ui/dash_app.py` entrypoint should keep working (compat shim).

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Implement frontend integration + tests |
| A | mkg | Approve UI/API behavior + deployment shape |
| C | Frontend dev | Validate UX + polling behavior |
| I | Users | Use UI without blocking |

## 3) Hypotheses
- **H-001:** Replacing inline compute with backend job enqueue + polling will keep the UI responsive while preserving functionality.
  - Evidence to confirm/refute: Dash shows queued/running progress, then renders results from backend.
- **H-002:** Moving Dash UI code under `frontend/` can be done without breaking legacy imports/entrypoints by providing shims.
  - Evidence to confirm/refute: `python app/ui/dash_app.py` still runs and tests import paths remain valid.

## 4) Questions to Answer
- **Q-001:** Which Dash callbacks perform heavy compute and must be replaced with API calls? → linked test(s): T-101
- **Q-002:** What minimal API contract/payload is required for each workload? → linked test(s): T-102

## 5) Investigation Tasks (test-first)
- **S-001:** Identify callback boundaries for optimization/discovery/leaderboard/patterns.
  - Expected signals: callbacks that call `run_grid_search`, `run_discovery`, `find_winning_patterns`, `Leaderboard(...)`.
  - Test T-101: code inspection of `app/ui/dash_app.py`.
  - Artifacts: `docs/frontend-backend-integration/artifacts/callback-inventory.md`
  - Risk R-001: missing a UI path leading to inconsistent behavior.
- **S-002:** Validate backend endpoints and define payload mapping.
  - Expected signals: one-to-one mapping from UI config → API payload.
  - Test T-102: unit tests for frontend API client request/response handling.
  - Artifacts: `docs/frontend-backend-integration/artifacts/unit-tests.txt`
  - Risk R-002: payload schema drift between frontend and backend.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Clear mapping exists for each workload: UI action → `POST /v1/*` → `GET /v1/jobs/{id}` → UI render.
- **IC-2:** Plan includes compatibility approach for legacy entrypoints/imports.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | UI still runs heavy compute path | Med | Med | Replace specific callbacks; add polling + stores |
| R-002 | Backend unreachable causes confusing UX | Med | Med | Surface backend errors in UI status fields |
| R-003 | Import path regressions from moving files | High | Med | Add `app/ui/*` shims + compile/unit tests |

## 8) Decision Log (provisional)
- **D-001:** Keep backtest execution in the UI for now; move only optimization/discovery/leaderboard/patterns to jobs.
  - Rationale: smallest step to remove the biggest UI blockers.
  - Alternatives considered: move backtest to backend job as well.
- **D-002:** Use stdlib `urllib` for backend HTTP calls.
  - Rationale: avoids adding `requests` dependency to the frontend image.
  - Alternatives considered: `requests`, `httpx`.

## 9) Exit Artifacts
- `docs/frontend-backend-integration/artifacts/callback-inventory.md`
- `docs/frontend-backend-integration/artifacts/unit-tests.txt`
- `docs/frontend-backend-integration/artifacts/py-compile.txt`

