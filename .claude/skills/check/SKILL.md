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
- `--deselect [node_id ...]`: pass pytest node ids to exclude, for failures already confirmed pre-existing via the `git stash` reproduction in step "On FAIL" below. Never use this to hide a failure introduced this session — it must follow, not replace, the baseline reproduction.
- **Mypy Static Check:** Strictly enforced (do NOT pass `--skip-mypy`). Validates semantic type compliance across interfaces.

Pipeline Order:
1. Spec Compliance Verification against contract.json — AST non-dummy implementation, dynamic `wiring` caller integration, **and dynamic assertion execution**: for every `contracts[].assertions` entry whose `input`/`output` are JSON primitives with no prose-like strings and whose keys match the target function's real parameters, `lean_check.py` actually imports the module and calls the function, comparing the real return value/exception against the contract — not just checking that the array is non-empty. Assertions with fixture/object inputs ("300-bar synthetic fixture") or descriptive outputs ("tuple of 4 LegBook in registry order") are skipped, not failed — write those as executable literal values in the spec whenever the value is genuinely computable, since only executed assertions catch bugs like a wrong sign or an unvalidated `mode` parameter.
2. Co-modification mapping check (source module must have a corresponding test, and caller module in `wiring.target_file` must be modified). This matches literally: a scenario's `target_test_file`/`name` from the contract must exist verbatim — a passing test with a renamed function in one ad-hoc file elsewhere is a spec-compliance FAIL, not a naming nit, and must be relocated rather than reconciled after the fact.
3. Strict Mypy Type Checking
4. Pytest Execution & Coverage Audit — a `coverage-target` FAIL now returns **every** violation across every touched file in one diagnostics list, not just the first, so a single run tells you the full remediation list.
5. **Orphaned-implementation gate is now automated inside spec-compliance (step 1), not a manual step**: for every `contracts[].name` that passed the non-dummy check, `lean_check.py` greps `src/` (excluding `tests/` and the symbol's own `def`/`class` line) and FAILs if there is no caller. A function that compiles and has a passing unit test but is never invoked from production code is caught here automatically. Still do the per-file coverage drill-down below by hand — the automated gate only proves *a* caller exists, not that coverage is adequate.
6. Vacuous-test scan on any test file touched this session (see below) — still manual, the tool cannot judge whether an assertion is meaningful, only whether it ran.

`lean_check`'s PASS now proves the contract's executable assertions actually matched real behavior and every named contract symbol has a production caller — a strictly stronger guarantee than "tests exist and exit 0." It still does NOT prove non-executable assertions (fixture-based, descriptive) were honored, or that coverage is *meaningful* rather than merely present. Never report its PASS as the final answer without the per-file coverage drill-down and Step 6 below.

### Per-file coverage drill-down (run immediately, not after doubting the summary number)
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

If a rewrite retires a gate/branch, also check whether any config field that fed only that branch is now unread (same grep pattern, scoped to the field name across `src/`, excluding its own dataclass definition). A field with zero production readers left over from a retired gate will independently fail dead-parameter regression tests (e.g. an AST-based "no dead config fields" scanner) — remove the field and its stale test fixtures in the same pass rather than treating that as a separate, later failure.

### 6. Vacuous-test scan
For any test asserting a regression fix or a "gate is reachable" property, check the assertion shape itself, not just whether it passes. Reject patterns like:
```python
assert real_condition or total == 0          # passes vacuously whenever total==0
assert some_count >= 0                        # always true, proves nothing
```
A test with an escape hatch that makes it pass when the exercised code path never ran is worse than no test — it reports false confidence. Reproduce the fixture manually (e.g. `uv run python -c "..."`) and print the intermediate values the assertion depends on to confirm the "real" branch, not the escape hatch, is what actually ran.

On **PASS**: report audit status with final coverage from output, and the per-file coverage table.
On **FAIL**: read stderr JSON diagnostic — it now lists every violation found in that phase, not just one — and apply the suggested fixes together. Before treating any pytest failure as this session's fault, `git stash` and reproduce it against the pre-session baseline — if identical, it's a pre-existing defect, deselect it, and say so explicitly rather than folding it into the fix budget.
If validation fails for **3 consecutive check cycles**, STOP and request human intervention (Strict budget).
Coverage thresholds: see testing.md §5.

## Output Format
```md
🟢 **[CHECK AUDIT PASS]** `[Blueprint Name]`
- **Spec Compliance**: Wiring ✅ | Non-dummy AST ✅
- **Static Analysis**: Mypy Strict ✅ | Regression Test ✅ | Final Cov [value]%
```
