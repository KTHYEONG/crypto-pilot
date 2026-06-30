---
name: audit
description: Intent Alignment & Core Logic Verification.
---

# Skill: Audit (Lightweight Intent & Contract Gatekeeper)

## Purpose
Verify that the implementation strictly adheres to the public interfaces (signatures) and business formulas defined in the Spec blueprint BEFORE running regression tests. Prevent token waste by enforcing static comparison and restricting natural language explanations.

## Execution Rules

### 1. Positioning & Workflow Transition
- **Trigger**: Run immediately after `implement` completes.
- **Pass Transition**: If audit passes, proceed directly to `check` (Regression & Coverage Audit).
- **Fail Transition**: Return to `implement` (or `spec`) with a highly compressed Gap Analysis.

### 2. Static Interface & Logic Inspection (Strict Fact-Checking)
- **Zero Explanation Rule**: Do not write paragraphs explaining why the code was written this way. Compare code against Spec purely by facts.
- **Interface Contract**: Compare public API names, parameters, and type hints in the code directly against the Spec. They must match 100%. Use Serena MCP `get_symbols_overview` for rapid signature checks.
- **Formulas & Intent**: Verify mathematical operations or Pandas/NumPy logic matches the Spec logic exactly.

### 3. Circuit Breaker (Anti-Loop)
- **3-Strike Rule**: If the code fails to align with the Spec intent for **3 consecutive audit attempts**, STOP execution immediately.
- **Action**: Do not try to rewrite code. Explicitly request **Human Intervention** to clarify the Spec or override.

### 4. Skip Execution/Mechanical Verification
- Do not run tests (`check` phase) or update documentation (`sync` phase) in this step. Keep checks strictly static.

## Verdicts
- **PASS**: Signatures and business logic formulas match the Spec perfectly. -> **Transition to `check`**
- **FAIL**: Discrepancies in signatures, missing formulas, or wrong types. -> **Return to `implement`** (or `spec`) with the compact Gap Analysis.

## Output Format (Strict Limit: Max 10 lines overall)
```md
### 🏁 Audit Result: [PASS / FAIL]

#### 🔍 Gap Analysis (Required ONLY if FAIL)
*Strictly max 3 bullet points, max 1 sentence per point. No raw code blocks.*
- **Signature Mismatch:** Spec `[expected_sig]` vs Code `[actual_sig]` in `[file:line]`
- **Logic Gap:** Spec Formula `[expected_formula]` missing/incorrect in `[file:line]`
- **Next Action:** [Return to implement | Return to spec | Human Intervention Required]
```
