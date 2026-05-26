---
name: review
description: Review the current diff against the request, spec, verification results, and project rules.
---

# review

Do not edit files unless explicitly asked.

## Purpose
Act as an independent, critical reviewer to overcome self-confirmation bias. Do not just verify that the code works; ensure it is structurally sound, maintainable, and compliant with project rules.

## Check
Focus on these critical audits:

1. **Spec Acceptance & Mechanical Pass:** Are all `Acceptance Criteria` fulfilled, and did `verify` pass with objective data?
2. **Devil's Advocate (Mandatory Risk Finding):** You MUST identify at least one potential flaw, edge case, technical debt, or unhandled failure mode introduced by these changes. If you cannot find one, look harder.
3. **Adherence to Core Rules:** Does the implementation strictly adhere to the architectural and stylistic guidelines defined in `AGENTS.md` and/or `CLAUDE.md`? (Explicitly check for violations).
4. **Maintainability (6-Month Rule):** Is the code easily understandable without context? Is it free of implicit "magic" or overly complex logic that would be hard to maintain 6 months from now?
5. **Documentation Sync:** Physically verify that relevant architecture docs (`docs/`) have been updated to reflect these changes, including the 'last_verified' date.

*(Do not perform line-by-line syntax checks or redundant testing here. Focus on the logical contract and structural integrity).*

## Verdict
- approve
- approve with risks (if Devil's Advocate findings are minor)
- request changes (if rules violated or docs missing)

## Output
```md
### 🔍 Review Verdict: [APPROVE / APPROVE WITH RISKS / REQUEST CHANGES]

#### 🕵️ Critical Audits
- **Adherence to AGENTS.md/CLAUDE.md:** [Pass/Fail - Details]
- **Devil's Advocate (Potential Flaws/Debt):** [Must list at least one potential risk or technical debt]
- **Maintainability:** [Assessment of readability and complexity]
- **Documentation Sync:** [Verified Docs Paths / Missing]

#### 📋 Handoff
- **Blocking Issues:** [None or List of required fixes]
- **Report:** [Final independent summary]
```