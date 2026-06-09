---
name: audit
description: Final audit for intent alignment and system knowledge promotion (Documentation Sync).
---

# Skill: Audit (Final Logic & Knowledge Audit)

## Purpose
Act as the final logical gatekeeper. Beyond simple bugs, determine if changes align with the system architecture and business rules. Once verified, promote transient knowledge (Specs) to permanent documentation (SSOT) and cleanup temporary artifacts.

## Audit Checklist
1. **Intent Consistency:**
   - **For Spec-based tasks:** Does the implementation strictly follow the "Why" and "How" of `docs/specs/*.md`?
   - **For Spec-less tasks:** Does the change satisfy the user's request without logical leaps?
2. **Logical Integrity:** Are there potential race conditions, poor error handling patterns, or inefficient logic missed by the `check` phase (mechanical verification)?
3. **Knowledge Sync (Strict Separation):**
   - **Formula Synchronization (Hybrid Math):** If core mathematical logic or algorithms changed in the code, you MUST update the corresponding formulas in `docs/architecture/` maintaining the Hybrid approach (use variables, not constants). Failure to sync formulas is an automatic FAIL.
   - **Architecture (`docs/architecture/`):** Update ONLY if core formulas, variables, or I/O interfaces have changed. Ensure the doc remains comprehensive and highly readable (use Mermaid/Tables). NO history.
   - **Decisions (`docs/decisions/`):** Append an **ULTRA-COMPRESSED** ADR (Max 5-7 lines: Delta, Rationale, Edge Cases) for EVERY logic change. Prevent file bloat. Ensure history is cumulative (newest on top).
   - **Ref:** Follow `documentation.md` for strict templates.
4. **Final Cleanup:** Has the temporary blueprint (`docs/specs/`) been deleted to prevent AI context pollution? (Skip for Spec-less tasks)

## Verdicts & Routing (Circuit Breaker)
- **PASS**: Logic and documentation are perfect. (Close task).
- **FAIL (Logic/Doc Error)**: Mismatch between design and code, or missing documentation. -> **Request Fix (Max 2 attempts)**.
- **CRITICAL FAIL**: Fundamental design flaw or repeated failure. -> **Request User Intervention (Ask User)**.

## Output Format
```md
### 🏁 Final Audit: [PASS / FAIL]

**1. Audit Summary**
- **Scope:** [Modified files and key logic]
- **Design Alignment:** [Pass/Fail] (Cite `check` results for mechanical integrity)
- **Knowledge Promotion:** [Updated doc paths or 'N/A']

**2. Logical Quality**
- [ ] Business rule compliance
- [ ] Pattern & maintainability
- [ ] Error handling & edge cases

**3. Knowledge Management**
- [ ] Permanent documentation (`docs/`) synchronized
- [ ] Temporary Spec file (`docs/specs/`) deleted (if applicable)

**4. Follow-up**
- [Next Step: Close / Handoff to Implement / Ask User]
```
