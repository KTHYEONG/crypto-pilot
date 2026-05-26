---
name: verify
description: Verify implemented changes with the smallest sufficient checks.
---

# verify

## Purpose
Run and report checks to ensure technical integrity and prevent regressions. Do not settle for "it should work"—prove it with objective data.

## Rules
- **Actual Summaries:** Always include the raw summary line from the tool (e.g., `5 passed, 2 failed in 0.5s`). Do not paraphrase the results.
- **Side-Effect Awareness:** If shared logic, base classes, or public APIs were changed, you MUST run tests for at least one dependent module or related test suite to check for side effects.
- **Proof of Execution:** Only report results for commands you actually executed in the current session.

## Choose Checks
Based on:
- **Dependency Analysis:** Which modules rely on the changed files?
- **Risk Level:** Higher risk requires broader test coverage.
- **Quant Context:** Financial/math logic requires precision validation.
- **Available Suites:** Use existing `pytest` marks or directory structures.

## Check Types
- **Static:** Lint (`ruff`) & Typecheck (`mypy`)
- **Dynamic:** Unit & Integration tests (`pytest`)
- **Regression:** Testing unaffected but related areas.
- **Manual:** Smoke tests via CLI or script execution.

## Output
```md
<verify>

### ✅ Verification: [PASS / FAIL / PARTIAL]
- **Summary:** `[Actual tool output summary, e.g., '15 passed in 1.2s']`
- **Commands:** `[Exact commands run, e.g., 'uv run pytest ...']`
- **Side-Effects Checked:** `[Modules/Tests checked for regressions or 'None']`
- **Passed:** `[Specific test names or categories]`
- **Failed/Skipped:** `[Specific failures with brief error message]`
- **Next:** `[Next Step]`

</verify>
```]`
- **Passed:** `[Items]`
- **Failed/Skipped:** `[Items]`
- **Risks:** [Remaining Risks]
- **Next:** `[Next Step]`

</verify>
```