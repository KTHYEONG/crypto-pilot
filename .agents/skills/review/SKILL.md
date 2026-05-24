---
name: review
description: Review the current diff against the request, spec, verification results, and project rules.
---

# review

Do not edit files unless explicitly asked.

## Purpose
Act as an independent reviewer.

## Check
- request/spec match
- over-implementation
- unrelated refactors
- meaningful tests
- missing edge cases
- unintended public API changes
- error handling
- security/permission concerns
- dependency justification
- verification sufficiency
- quant requirements, if active
- **Performance & Scalability:** 
  - Is the chosen optimization strategy appropriate for the identified bottleneck (CPU vs I/O vs Memory)?
  - Are Network/DB calls optimized (e.g., batched, cached, or pooled) to prevent excessive latency and rate-limit hits?
  - Did the implementation choose an overly complex mathematical optimization when a robust built-in method was safer?
  - Are there hidden O(N^2) loops, blocking I/O in async contexts, or memory leaks in long-running logic?
  - Does the complexity align with the declared `Scale`?

## Verdict
- approve
- approve with risks
- request changes

## Output
```md
## Review
- Verdict:
- Blocking Issues:
- Non-blocking Issues:
- Missing Tests/Checks:
- Over-implementation:
- Rule Violations:
- Required Fixes:
- Final Report Draft:
```