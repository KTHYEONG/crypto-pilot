---
name: check
description: Regression Testing, Coverage Auditing, Error Triage.
---

# Check (L2 Independent Audit Gatekeeper)
Verify structural type correctness, spec contract compliance (wiring, non-dummy AST), Mypy strict adherence, and regression coverage.

Run: `uv run python scripts/lean_check.py --files [modified_files] --spec docs/specs/[feature]_contract.json --skip-lint`

- `--files`: include all modified .py files (source, caller, and test pairs).
- `--spec`: path to the spec contract JSON (must match perfectly).
- `--skip-lint`: **Mandatory**. Skips fast syntactic formatting checks (assumed completed in Implement L1.5).
- **Mypy Static Check:** Strictly enforced (do NOT pass `--skip-mypy`). Validates semantic type compliance across interfaces.

Pipeline Order:
1. Spec Compliance Verification (against contract.json: assertions, AST non-dummy implementation, and dynamic `wiring` caller module integration)
2. Co-modification mapping check (source module must have a corresponding test, and caller module in `wiring.target_file` must be modified)
3. Strict Mypy Type Checking
4. Pytest Execution & Coverage Audit (regression prevention)
5. Per-file coverage drill-down + dead-code sweep (mandatory, not optional — see below)
6. Vacuous-test scan on any test file touched this session (see below)

`lean_check`'s PASS only proves "tests exist and exit 0 against the contract's literal assertions" — it does NOT prove the tests exercise the changed logic meaningfully. Never report its PASS as the final answer without Steps 5–6.

### 5. Per-file coverage drill-down (run immediately, not after doubting the summary number)
For every file touched in `--files`, run coverage scoped to that file's **dotted module path**, not its filesystem path:
```
uv run pytest [touched test files] -q --cov=src.domain.path.to.module --cov-report=term-missing
```
`--cov=path/with/slashes` silently collects nothing (`No data collected` warning, easy to miss) — always use dotted form. A single aggregate percentage from `lean_check` can hide a core module sitting at 50% inside an otherwise-healthy suite; check every touched file's own number, not just the total.

In the same pass, grep every function/class this session added or rewrote for callers:
```
grep -rn "OldOrNewSymbolName\b" src/ tests/ | grep -v "def OldOrNewSymbolName"
```
Any rewrite that replaces inline logic tends to strand the old helper functions as dead code with zero callers — this drags coverage down for a real reason and violates the dead-code cleanup directive in CLAUDE.md. Remove confirmed-dead functions instead of leaving them uncovered.

### 6. Vacuous-test scan
For any test asserting a regression fix or a "gate is reachable" property, check the assertion shape itself, not just whether it passes. Reject patterns like:
```python
assert real_condition or total == 0          # passes vacuously whenever total==0
assert some_count >= 0                        # always true, proves nothing
```
A test with an escape hatch that makes it pass when the exercised code path never ran is worse than no test — it reports false confidence. Reproduce the fixture manually (e.g. `uv run python -c "..."`) and print the intermediate values the assertion depends on to confirm the "real" branch, not the escape hatch, is what actually ran.

On **PASS**: report audit status with final coverage from output, and the per-file coverage table.
On **FAIL**: read stderr JSON diagnostic → apply suggested fix. Before treating any pytest failure as this session's fault, `git stash` and reproduce it against the pre-session baseline — if identical, it's a pre-existing defect, deselect it, and say so explicitly rather than folding it into the fix budget.
If validation fails for **3 consecutive check cycles**, STOP and request human intervention (Strict budget).
Coverage thresholds: see testing.md §5.

## Output Format
```md
🟢 **[CHECK AUDIT PASS]** `[Blueprint Name]`
- **Spec Compliance**: Wiring ✅ | Non-dummy AST ✅
- **Static Analysis**: Mypy Strict ✅ | Regression Test ✅ | Final Cov [value]%
```
