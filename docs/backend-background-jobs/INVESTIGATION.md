> **Project Title:** Backend Background Jobs (API + Queue)
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Investigation Plan

## 1) Problem Statement
- **Context:** Optimization, strategy discovery, leaderboard, and pattern recognition currently run inline in the UI process, which blocks the frontend and scales poorly.
- **Desired Outcome:** Move these workloads to backend background jobs triggered by a frontend API, persisting results and market data in DB.
- **Constraints/Assumptions:**
  - A lightweight DB-first queue is acceptable (SQLite by default).
  - Spark can be optional (enabled via env) to support heavier processing paths.
  - Keep existing `app/*` backtest/discovery code as the computation engine.

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Implement API/queue/worker + wiring |
| A | mkg | Approve architecture + deploy choices |
| C | Frontend dev | Integrate UI job triggers + polling |
| I | Users | Consume job results via UI |

## 3) Hypotheses
- **H-001:** Moving heavy computations behind an API + queue will keep the UI responsive and enable horizontal scaling via multiple workers.
  - Evidence to confirm/refute: background worker can claim and execute jobs; UI can fetch job status/result.
- **H-002:** A DB-backed queue (SQLite initially) is sufficient for local/dev and can be swapped later for Redis/Postgres as scale needs grow.
  - Evidence to confirm/refute: atomic job claiming works; job lifecycle persists through process restarts.

## 4) Questions to Answer
- **Q-001:** What existing compute modules can be reused as job handlers? → linked test(s): T-101
- **Q-002:** What minimal job schema supports progress + results? → linked test(s): T-102

## 5) Investigation Tasks (test-first)
- **S-001:** Inventory current compute entry points (optimization, discovery, patterns, leaderboard).
  - Expected signals: known functions that can be called from a worker.
  - Test T-101: code inspection of `app/optimization/*` + `app/discovery/*`.
  - Artifacts: `docs/backend-background-jobs/artifacts/code-entrypoints.md`
  - Risk R-001: mismatched imports when running outside UI.
- **S-002:** Define DB schema for queued jobs and progress updates.
  - Expected signals: enqueue → claim → progress → success/fail states.
  - Test T-102: unit-level DB test using a temp SQLite file.
  - Artifacts: `docs/backend-background-jobs/artifacts/test-backend-jobs-db.txt`
  - Risk R-002: concurrency edge cases with SQLite.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Clear mapping from "UI action" → "job_type + payload" → "handler".
- **IC-2:** Minimal job lifecycle + progress model is specified.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Import path issues when running compute outside UI | Med | Med | Add deterministic `PYTHONPATH`/bootstrap in API+worker |
| R-002 | SQLite queue contention with many workers | Med | Low | Use atomic claim + allow later migration to Redis/Postgres |
| R-003 | Spark runtime complexity (Java) | Med | Med | Make Spark optional via `USE_SPARK` |

## 8) Decision Log (provisional)
- **D-001:** Use SQLite-backed queue + worker loop initially.
  - Rationale: minimal infra, consistent with existing SQLite usage.
  - Alternatives considered: Redis (RQ/Celery), Postgres (SKIP LOCKED).
- **D-002:** Expose compute as async jobs via FastAPI.
  - Rationale: simple frontend integration + clear boundaries.
  - Alternatives considered: inline FastAPI BackgroundTasks (no durable queue).

## 9) Exit Artifacts
- `docs/backend-background-jobs/artifacts/code-entrypoints.md`
- `docs/backend-background-jobs/artifacts/test-backend-jobs-db.txt`

