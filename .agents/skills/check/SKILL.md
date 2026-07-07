---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage.
---

# Skill: Check (Integrated Validation & Triage Gatekeeper)

## Purpose
Unify static contract compliance, dynamic regression test execution, and structural test coverage auditing. Perform systematic triage and error diagnosis strictly upon failure.

## Execution Rules

### 1. Unified Validation Pipeline
Run ALL of the following in a **single batch command**:
1. **Lint (`uv run ruff check`)** — entire project (`src/` and `tests/`).
2. **Type check (`uv run mypy`)** — only source files modified by the current spec.
3. **Regression (`uv run pytest [regression_target] -v --tb=short`)** — spec-mapped test files.
4. **Coverage (`uv run pytest --cov=[module_path] [regression_target] --cov-report=term-missing`)**.
5. **Contract Review** — Verify that all TDD scenarios (Happy Path, Edge, Error) written in `docs/specs/[feature].md` are actually implemented as assertions. Check if any side-effects occur in adjacent modules.

### 2. Dynamic Verification

#### 2a. Regression Scope (Scope Isolation)
- `[regression_target]` MUST be the explicit test file paths mapped 1:1 from `docs/specs/[feature].md`.
  - **Example:** `tests/unit/domain/futures/signals/test_my_feature.py`
- If supplementary test files exist (added by `implement` for coverage gap), **include them too**.
- **Out-of-scope Error Ban**: If collection errors occur from files OUTSIDE `[regression_target]`, do NOT fix them. Use `--ignore=<path>` to skip them and report separately.
- NEVER use bare `tests/` or `tests/unit/` without precise paths.

#### 2b. Coverage Scope & Thresholds
- `[module_path]` = the **module(s) containing modified files** (e.g., `src/domain/futures/alpha_foundry`).
- Apply **tiered thresholds** from [.agents/rules/testing.md §5](../../../.agents/rules/testing.md#5-ai-coverage-driven-self-correction-loop):
  - **Core Logic (Domain, Signal, Sizing, Portfolio):** >= 90%
  - **Adapters/Runners/DTOs/Boilerplate:** >= 70%
  - **Entrypoints / CLI / `__init__.py`:** skip
- **Unchanged files in the module do NOT count toward the threshold calculation.** Only measure coverage on files that were created or modified by the current spec.

### 3. Failure Triage & Loop Circuit Breaker
If any step in the validation pipeline fails:
- **Triage**: Read only the compiled error logs and failure lines from the execution runner.
- **Diagnostics**: Determine whether the failure is a design error (requires `spec` rollback) or code bug (requires `implement` rollback).
- **Circuit Breaker**: If regression fails for **3 consecutive cycles**, STOP and request human intervention.

## Verdicts & Routing
- **PASS**: All static/dynamic criteria met AND all tiered coverage thresholds satisfied. ➔ **Transition to `sync`**
- **FAIL**: Any discrepancy found. ➔ **Transition to `implement` (or `spec`)** with clear Gap Analysis.

## Output Format
결과 보고는 마크다운 서식 및 다음 단계 지정을 배제하고 단 한 줄(Single-line)로 요약합니다.

포맷:
[결과] | [원인/상태]

예시 (FAIL):
FAIL | metrics.py:529,531 (Cov 82% < 90%)

예시 (PASS):
PASS | All checks passed (Cov 94%)
