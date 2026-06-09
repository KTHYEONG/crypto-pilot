---
name: audit
description: Professional Peer Review, Intent Alignment, and Architecture Integrity.
---

# Skill: Audit (Professional Quality Gatekeeper)

## Purpose
The final expert gatekeeper. Beyond just matching the Spec, you must critically evaluate the code for performance, maintainability, and global system integrity. You are a Senior Reviewer who trusts no one.

## Execution Rules

### 1. Zero-Trust Review (Logic & Performance)
- **Question the Architect**: Even if the code matches `docs/specs/*.md`, ask: "Is this logic actually optimal?"
- **Quant/Financial Filter**: For trading logic, check for:
  - Vectorization vs. Loops (prefer NumPy/Pandas ops).
  - Memory efficiency (avoid unnecessary deep copies).
  - Numerical stability and floating-point precision.
- **Error Handling**: Is it too defensive? Or missing critical failure points?

### 2. Architecture Integrity (Global Compliance)
- **Convention Check**: Does the code follow `GEMINI.md` and specific rules in `.agents/rules/` (e.g., `quant.md`, `trading_bot.md`)?
- **Pattern Match**: Does it use the project's established patterns (Pydantic settings, logging vs. print, etc.)?
- **Tech Debt**: Does this change introduce "hidden" debt or messy dependencies not mentioned in the Spec?

### 3. Spec-to-Code Alignment (Baseline)
- Verify the implementer followed the `Algorithmic Flow` and `Test Scenarios`.
- Ensure NO unauthorized deviations or "lazy" implementations.

### 4. Knowledge Promotion & Cleanup
- Sync core formulas to `docs/architecture/`.
- Update `docs/decisions/` with a compressed ADR (5 lines).
- Delete the temporary `docs/specs/*.md`.

## Verdicts
- **PASS**: Code is high-quality, architecturally sound, and intent-aligned.
- **FAIL (Quality/Integrity)**: Code is suboptimal, violates quant rules, or ignores system conventions. -> **Return to `implement` (or `spec`)** with detailed expert feedback.

## Output Format
```md
### 🏁 Professional Audit: [PASS / FAIL]

**1. Zero-Trust Review**
- [ ] Optimal Logic & Performance (NumPy/Pandas/Memory)
- [ ] Robustness & Error Handling

**2. Architecture & Global Compliance**
- [ ] Compliance with `.agents/rules/` (Quant/Trading)
- [ ] Follows `GEMINI.md` conventions

**3. Alignment & Promotion**
- [ ] Spec Alignment (Flow & Tests)
- [ ] SSOT Documentation Synced & Cleanup

**4. Expert Feedback & Next Step**
- [Expert review comments for the developer]
- [Next Step: Proceed to COMMIT / Return to X]
```
