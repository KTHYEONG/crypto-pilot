---
name: check
description: Regression Testing, Coverage Auditing, Error Triage.
---

# Check
Run: `uv run python scripts/lean_check.py --files [modified_files] [--spec docs/specs/[feature]_contract.json]`
- `--files`: include all modified .py files (source + test pairs).
- `--spec`: optional path to spec contract JSON for spec compliance verification (runs first).
Pipeline order: spec(s) → co-modification → print() → ruff → mypy → pytest+coverage (stops at first failure).

On **PASS**: report `Cov [value]%` from output.
On **FAIL**: read stderr JSON diagnostic → apply suggested fix.
If regression fails for **3 consecutive check cycles**, STOP and request human intervention.
Coverage thresholds: see testing.md §5.
