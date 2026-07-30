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

Do NOT dump raw error logs or test suites in chat. Return ONLY this compact card:

### 📌 [CHECK] <Audit Target / Task Title>

- **Status**: <PASS | FAIL>
- **Checks**: Spec/Wiring: <PASS/FAIL> | Mypy: <PASS/FAIL> | Regression: <PASS/FAIL>
- **Coverage**: Total: <FinalCov%> | Modified Files: <FileCovSummary>
- **Issue**: <1-line root cause and fix plan (Only on FAIL)>


