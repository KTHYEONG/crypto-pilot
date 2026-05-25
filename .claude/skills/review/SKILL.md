---
name: review
description: Review the current diff against the request, spec, verification results, and project rules.
---

# review

Do not edit files unless explicitly asked.

## Purpose
Act as an independent reviewer.

## Check
Focus strictly on these 3 core audits:
1. **Spec Acceptance:** Are all `Acceptance Criteria` defined in the `spec` completely fulfilled?
2. **Mechanical Pass:** Did the `verify` step run successfully with zero critical errors?
3. **No Regressions:** Are there any unintended changes to public APIs, unrelated refactors, or missing documentation updates (check 'last_verified' dates)?

*(Do not perform line-by-line syntax checks or redundant testing here. Focus on the logical contract).*

## Verdict
- approve
- approve with risks
- request changes

## Output
```md
### 🔍 Review Verdict: [APPROVE / REQUEST CHANGES]
- **Blocking:** [None or List]
- **Docs/Rules:** [Invariants/Verified Date status]
- **Fixes Required:** [Fixes]
- **Report:** [Final Summary]
```