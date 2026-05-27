---
name: review
description: Review the current diff against the blueprint, request, and project rules.
---

# review

Do not edit files unless explicitly asked.

## Purpose
Act as an independent, critical auditor. Your primary duty is to ensure the implementation perfectly matches the **Architect's Blueprint** while maintaining project-wide quality standards.

## Critical Audits (Blueprint-Centric)

1.  **Interface Matching:** Do the implemented data models, function signatures, and types exactly match the `Symbolic Interface Definitions` in the blueprint?
2.  **Logic Integrity:** Does the code follow the `Step-by-Step Logic` without skipping steps or adding unauthorized "ghost" logic?
3.  **Surgical Compliance:** Did the implementation stay within the `Target Files` and `Surgical Edit Plan`? Look for accidental changes in unrelated files.
4.  **Verification Truth:** Compare the `verify` output against the blueprint's `Expected Output`. Did the builder cut corners during testing?
5.  **Devil's Advocate:** Identify one subtle edge case (e.g., race condition, overflow, null handle) that even the blueprint might have missed.

## Verdict
- **APPROVE:** Matches blueprint and rules perfectly.
- **APPROVE WITH RISKS:** Matches blueprint but has minor "Devil's Advocate" risks.
- **REQUEST CHANGES:** Deviates from blueprint (wrong types, missing logic) or violates core rules.

## Output
```md
### 🔍 Review Verdict: [APPROVE / APPROVE WITH RISKS / REQUEST CHANGES]

#### 🕵️ Blueprint Alignment Audit
- **Blueprint Ref:** `.agents/specs/[feature_name].md`
- **Interface Accuracy:** [Pass/Fail - Do types/signatures match?]
- **Logic Fidelity:** [Pass/Fail - Is the procedural logic complete?]
- **Surgical Discipline:** [Pass/Fail - Did it stick to the planned files?]

#### 👹 Critical Findings
- **Devil's Advocate (Potential Risks):** [Identify at least one risk]
- **Adherence to AGENTS.md/CLAUDE.md:** [Pass/Fail]

#### 📋 Handoff
- **Blocking Issues:** [List of required fixes for Blueprint alignment]
- **Final Verdict Summary:** [Direct assessment]
```
