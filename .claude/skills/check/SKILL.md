---
name: check
description: Regression Testing, Coverage Auditing, Error Triage.
---

# Check (L2 Final Gatekeeper)
Verify structural type correctness, interface contracts, and coverage regressions.

Run: `uv run python scripts/lean_check.py --files [modified_files] --spec docs/specs/[feature]_contract.json --skip-lint`

- `--files`: include all modified .py files (source + test pairs).
- `--spec`: path to the spec contract JSON (must match perfectly).
- `--skip-lint`: **Mandatory**. Skips fast syntactic formatting checks (assumed completed in Implement L1.5).
- **Mypy Static Check:** Strictly enforced (do NOT pass `--skip-mypy`). Validates semantic type compliance across interfaces.

Pipeline Order:
1. Spec Compliance Verification (against contract.json: assertions, wiring & AST non-dummy implementation)
2. Co-modification mapping check (source module must have a corresponding test)
3. Strict Mypy Type Checking
4. Pytest Execution & Coverage Audit (regression prevention)

On **PASS**: report `Cov [value]%` from output.
On **FAIL**: read stderr JSON diagnostic → apply suggested fix.
If validation fails for **3 consecutive check cycles**, STOP and request human intervention (Strict budget).
Coverage thresholds: see testing.md §5.
