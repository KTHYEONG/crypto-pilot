---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check

`/check` audits the implementation; it is not an architecture repair loop. Use:

```text
uv run python scripts/lean_check.py --files <modified .py files> --spec <contract.json> --skip-lint
```

## Fast preflight

Before the full command, use the cheapest relevant checks:

1. `rg` every contract symbol, caller, wiring expression, and scenario name.
2. Run ruff on touched files.
3. Run touched tests only.
4. Run `--spec-only` once after wiring.

If this finds contract drift, return it to Implement/Spec. Do not silently rename tests or modify the
contract during audit.

## Full audit order

1. Contract: literal assertions, non-dummy implementation, scenario/file mapping, and production callers.
2. Static: strict mypy.
3. Tests and changed-line coverage.
4. Manual per-file coverage, caller search, stale-field search, and vacuous-test review.

Batch coverage misses into one test-writing pass. Distinguish contract drift, logic defects, fixture issues,
and environment failures in the report. A pre-existing failure may be deselected only after reproducing it
against the baseline as required by the project rules.

## PASS report

Report the blueprint name, contract/wiring status, strict mypy, regression result, final total coverage, and
the per-file coverage table. Mention meaningful warnings or pre-existing exclusions; do not claim that
coverage alone proves the behavior is economically valid.

## Output

Keep the result compact:

`🟢 [CHECK AUDIT PASS] <Blueprint>`
`Spec/Wiring | Mypy | Regression | Final Cov | File coverage`

On PASS, do not repeat command logs or implementation details. On FAIL, report only phase, root cause,
impact, and the smallest next action.
