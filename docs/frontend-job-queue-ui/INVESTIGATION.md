> **Project Title:** Frontend Job Queue UI
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Investigation Plan

## 1) Problem Statement
- **Context:** Users need visibility into backend background work (optimization, discovery, leaderboard refresh, patterns) so the Dash frontend can show what is processing and why.
- **Desired Outcome:** Provide a frontend “Job Queue” view that lists jobs, shows status/progress/events, and supports canceling jobs.
- **Constraints/Assumptions:**
  - Backend provides durable job state in DB (SQLite by default).
  - Frontend should use minimal deps (stdlib HTTP client).
  - UI should not spam backend; auto-refresh must be user-controlled.

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Implement UI + client wiring |
| A | mkg | Approve UX + deploy |
| C | Backend | Ensure jobs API supports list/detail/events/cancel |
| I | Users | View job status + results |

## 3) Hypotheses
- **H-001:** Exposing `/v1/jobs` + `/v1/jobs/{id}` + events enables the UI to display job state without reading DB directly.
  - Evidence to confirm/refute: frontend can list jobs and load a selected job’s detail and events.
- **H-002:** User-controlled polling (auto-refresh toggle + interval) prevents unnecessary backend load.
  - Evidence to confirm/refute: job list/detail refresh only on tab-active + toggle/refresh actions.

## 4) Questions to Answer
- **Q-001:** What endpoints and response shapes are needed for the UI? → linked test(s): **T-001**
- **Q-002:** What minimal controls make the queue usable (filters, detail, cancel)? → linked test(s): **T-002**

## 5) Investigation Tasks (test-first)
- **S-001:** Verify backend exposes list/detail/events/cancel endpoints.
  - Expected signals: `/v1/jobs` returns a list; `/v1/jobs/{id}` returns status/progress; `/events` returns log entries; `/cancel` transitions to `canceled`.
  - Test **T-001:** code inspection of `backend/api/main.py`.
  - Artifacts: none (code inspection).
  - Risk **R-001:** response field mismatch breaks UI rendering.
- **S-002:** Verify frontend API client supports required calls.
  - Expected signals: `list_jobs`, `get_job_events`, and `cancel_job` exist and build correct URLs.
  - Test **T-002:** `python3 -m unittest test_frontend_api_client.ApiClientTests`
  - Artifacts: `docs/frontend-job-queue-ui/artifacts/unittest_api_client.out`
  - Risk **R-002:** client error handling masks backend errors.
- **S-003:** Ensure Dash UI can render a queue table and a detail view.
  - Expected signals: “Job Queue” tab exists, shows table, selection loads details/events.
  - Test **T-003:** `python3 -m py_compile frontend/ui/dash_app.py`
  - Artifacts: `docs/frontend-job-queue-ui/artifacts/py_compile.out`
  - Risk **R-003:** callback duplication/trigger logic causes refresh storms.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** The required backend endpoints and frontend client calls are identified.
- **IC-2:** A minimal UI surface is defined: filters + list + detail/events + cancel.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Field mismatch between backend and UI | Med | Med | Keep UI using backend API models; add focused unit tests for URL building |
| R-002 | Polling overload | Med | Low | Gate refresh by active tab + auto-refresh toggle |
| R-003 | Cancel semantics inconsistent (worker overwrites canceled) | Med | Med | Ensure backend treats cancel as terminal (guard success/fail updates) |

## 8) Decision Log (provisional)
- **D-001:** Implement Job Queue tab driven purely by backend API (no DB access in frontend).
  - Rationale: keeps frontend stateless and deployable separately.
  - Alternatives considered: frontend reads SQLite directly (not deploy-safe).

## 9) Exit Artifacts
- `docs/frontend-job-queue-ui/artifacts/py_compile.out`
- `docs/frontend-job-queue-ui/artifacts/unittest_api_client.out`
