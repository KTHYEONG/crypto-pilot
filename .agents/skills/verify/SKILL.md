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

## Commands Run
- ...

## Passed
- ...

## Failed
- ...

## Skipped
- item: reason

## Result
pass | fail | partial

## Remaining Risks
- ...

## Next Step
review | fix | broader verification | complete

</verify>
```