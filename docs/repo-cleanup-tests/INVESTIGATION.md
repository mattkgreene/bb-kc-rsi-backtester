> **Project Title:** Repo Cleanup + Test Baseline
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Investigation Plan

## 1) Problem Statement
- **Context:** The codebase includes optional heavy dependencies (dash/pandas/fastapi) and multiple moving parts (frontend UI, backend API/worker). We need a clean baseline that compiles and runs unit tests reliably.
- **Desired Outcome:** A fast, dependency-tolerant unit test suite that can run in minimal environments, plus a small set of checks that validate recent queue/leaderboard work.
- **Constraints/Assumptions:**
  - Tests must skip (not fail) when optional deps aren’t installed.
  - Use stdlib `unittest` (no new test framework).

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Make tests reliable + run them |
| A | mkg | Accept test baseline |
| C | Backend/Frontend | Keep behavior stable |
| I | CI/Deploy | Consume test results |

## 3) Hypotheses
- **H-001:** `python3 -m py_compile` over key modules catches syntax/merge issues quickly.
  - Evidence to confirm/refute: compile succeeds with no output/errors.
- **H-002:** Unit test discovery can be made resilient to missing optional deps via lazy imports and skips.
  - Evidence to confirm/refute: `python3 -m unittest discover` passes in a minimal env with skips.
- **H-003:** Cancel is a terminal state (worker cannot overwrite canceled with success/failure).
  - Evidence to confirm/refute: DB tests confirm canceled jobs remain canceled after `mark_succeeded`.

## 4) Questions to Answer
- **Q-001:** Does the current tree compile cleanly? → linked test(s): **T-001**
- **Q-002:** Do unit tests pass under minimal deps? → linked test(s): **T-002**
- **Q-003:** Is cancellation semantics safe under races? → linked test(s): **T-003**

## 5) Investigation Tasks (test-first)
- **S-001:** Run py_compile over critical modules (backend DB/worker/api + frontend client/UI).
  - Expected signals: exit code 0.
  - Test **T-001:** `python3 -m py_compile ...`
  - Artifacts: `docs/repo-cleanup-tests/artifacts/py_compile.out`
  - Risk **R-001:** syntax/indent errors in large Dash file.
- **S-002:** Run unit test discovery and record skips.
  - Expected signals: `OK` with deterministic skips (dash/pandas/fastapi missing).
  - Test **T-002:** `python3 -m unittest discover -s . -p 'test_*.py'`
  - Artifacts: `docs/repo-cleanup-tests/artifacts/unittest.out`
  - Risk **R-002:** import-time failures prevent discovery.
- **S-003:** Validate cancel terminal semantics at DB layer.
  - Expected signals: canceled jobs remain `canceled` even if worker attempts success/failure updates.
  - Test **T-003:** `python3 -m unittest test_backend_jobs_db.JobsDbTests`
  - Artifacts: `docs/repo-cleanup-tests/artifacts/unittest_after_cancel_fix.out`
  - Risk **R-003:** worker overwrites canceled status under races.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Compile checks succeed.
- **IC-2:** Unit tests pass (allowing skips for optional deps).
- **IC-3:** Cancellation cannot be overwritten by success/failure.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Large file edits cause syntax issues | High | Med | Keep py_compile checks as a gate |
| R-002 | Optional deps cause import failures | Med | Med | Lazy imports + skip tests when deps missing |
| R-003 | Cancel race leads to incorrect status | Med | Med | DB guards + worker checks |

## 8) Decision Log (provisional)
- **D-001:** Keep the default unit test suite dependency-light (stdlib `unittest`) with graceful skips.
  - Rationale: enables running tests in constrained deploy containers.
  - Alternatives considered: enforce installing dash/pandas/fastapi everywhere.

## 9) Exit Artifacts
- `docs/repo-cleanup-tests/artifacts/py_compile.out`
- `docs/repo-cleanup-tests/artifacts/unittest.out`
- `docs/repo-cleanup-tests/artifacts/py_compile_after_cancel_fix.out`
- `docs/repo-cleanup-tests/artifacts/unittest_after_cancel_fix.out`
- `docs/repo-cleanup-tests/artifacts/py_compile_final.out`
- `docs/repo-cleanup-tests/artifacts/unittest_final.out`
