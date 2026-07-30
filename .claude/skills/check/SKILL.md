---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check

Fast-execution model audit protocol (GPT-5.6 Luna, Sonnet 5, etc.). Independent verification of contract compliance, typing, regressions, and changed-line coverage.

## Core Rule

`/check` is an audit pass, NOT an architectural repair loop.
Execute audit command:
```bash
uv run python scripts/lean_check.py --files <modified .py files> --spec <contract.json> --skip-lint
```

## Deterministic Audit Pipeline

### Step 1: Preflight Search
- [ ] `rg` all contract symbols, callers, wiring expressions, and scenario names.
- [ ] Run `uv run ruff check <modified_files>`.
- [ ] Run `uv run pytest <touched_tests>`.
- [ ] *Exit Gate*: If contract drift is found, STOP and return to `/implement` or `/spec`. Do NOT modify contract/test names.

### Step 2: Full Audit Order
- [ ] Verify literal assertions and non-dummy implementation.
- [ ] Run `uv run mypy <modified_files>` in strict mode.
- [ ] Run test suite and inspect changed-line coverage.
- [ ] Perform vacuous-test & stale-field review.

## Output Format

### PASS Output
Return ONLY this 2-line summary:

🟢 `[CHECK AUDIT PASS] <Blueprint>`
`Spec/Wiring: <PASS> | Mypy: <PASS> | Regression: <PASS> | Cov: <FinalCov%> | Files: <FileCoverageTable>`

### FAIL Output
On failure, report ONLY:
🔴 `[CHECK AUDIT FAIL] <Blueprint>`
`Phase: <Contract/Static/Tests/Coverage> | Root Cause: <Cause> | Action: <Smallest next action>`


