---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage with Minimal Token Consumption.
---

# Skill: Check (Ultra-Lean Validation & Triage Gatekeeper)

## Purpose
Unify static contract compliance, dynamic regression test execution, and structural test coverage auditing. Maintain strict token efficiency by minimizing command output logs and isolating verification scope.

## Execution Rules

### 1. Minimal Output Validation Pipeline (Token Saving)
Run ALL of the following commands targeting **only the modified files and target modules**. Ensure quiet/summary flags are used to avoid stdout bloat:
1. **Lint (`uv run ruff check [modified_files] --quiet`)**: Only check currently modified source and test files.
2. **Type Check (`uv run mypy [modified_files] --ignore-missing-imports --summary-only`)**: Restrict to changed files with summarized output.
3. **Regression (`uv run pytest [regression_target] -q --tb=line`)**: Run only spec-mapped test files. `-q` (quiet) hides passing test names, and `--tb=line` limits traceback to a single line per error.
4. **Coverage (`uv run pytest --cov=[module_path] [regression_target] --cov-report=term`)**: Output summary-level coverage percentages only. **Never use `--cov-report=term-missing`** to prevent long missing-line tables.

### 2. Verification Scope

#### 2a. Target Limits
- **Precise Paths**: Always specify the exact target test file path (e.g. `tests/unit/domain/.../test_x.py`). Never run pytest on broad directories like `tests/` to prevent execution overhead and massive terminal outputs.

#### 2b. Coverage Thresholds
- Compare against target thresholds using the printed coverage summary:
  - **Core Logic (Domain, Signal, Sizing, Portfolio):** >= 90%
  - **Adapters/Runners/DTOs/Boilerplate:** >= 70%
- Measure coverage exclusively on files created or modified by the current task.

### 3. Triage & Circuit Breaker (On Failure)
If any step fails, retrieve ONLY the specific failing assertion line or traceback summary:
- **Diagnostics**: Determine if it requires a design change (`spec` rollback) or a bug fix (`implement` rollback).
- **Circuit Breaker**: If regression fails for **3 consecutive cycles**, STOP and request human intervention.

## Output Format
Write check results strictly in the compact format below:

### 🟢 PASS Format:
🟢 PASS | All checks passed (Cov [value]%)

### 🔴 FAIL Format:
🔴 FAIL | [brief failure summary]
- 🔍 Cause: [file path:line number] - [error reason]
- 🛠️ Fix: Apply [modification action] to [function/class] in [target file]
