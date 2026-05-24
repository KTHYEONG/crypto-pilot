---
name: verify
description: Verify implemented changes with the smallest sufficient checks.
---

# verify

## Purpose
Run and report checks needed for confidence.

## Choose Checks
Based on:
- changed files
- risk level
- implementation strategy
- affected modules
- quant context
- available tests

## Check Types
- lint
- typecheck
- unit tests
- integration tests
- build
- smoke/manual check
- regression check
- quant validation, if relevant

## Output
```md
<verify>

### ✅ Verification: [PASS / FAIL / PARTIAL]
- **Commands:** `[Cmds Run]`
- **Passed:** `[Items]`
- **Failed/Skipped:** `[Items]`
- **Risks:** [Remaining Risks]
- **Next:** `[Next Step]`

</verify>
```