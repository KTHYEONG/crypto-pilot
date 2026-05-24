---
name: ai-triage
description: Classify a development task and choose the smallest safe workflow before implementation.
---

# ai-triage

Do not edit files.

## Purpose
Classify the task before work begins to ensure quality and efficiency.

## Decide
- Task type: bug | feature | refactor | UI/style | config/build | docs | test-only | architecture | quant
- Risk: low | medium | high
- Uncertainty: low | medium | high
- Scale: high (heavy data, network I/O, massive loops) | low
- Quant context: active (if task involves math, stats, modeling, or backtesting) | none
- Need spec: yes | no
- Need review: yes | no
- Strategy: direct patch | regression-first | TDD | characterization-first | spike | docs-only

## Routing
- Low risk: `context-scan → implement → verify`
- Medium risk: `context-scan → spec → implement → verify → review`
- High risk: `context-scan → spec/PRD → implement → verify → review`
- Bug: `context-scan → spec(bug brief) → implement(regression-first) → verify → review`
- Refactor: `context-scan → spec(refactor plan) → implement(characterization-first) → verify → review`
- Architecture: `context-scan → spec(ADR) → spike/plan`
- Quant: `context-scan → spec (math/logic) → implement → verify (leakage/stability) → review`

## Execution Strategy (Token Optimization)
- **Direct Execution (Low Risk):** For simple bug fixes or UI tweaks, you MAY execute `context-scan` and `implement` sequentially within a single turn using parallel MCP tool calls to save context window.
- **Context Awareness:** 
  - If `Quant context: active`, refer to `.agents/rules/quant.md` to augment `<plan>` and `<risk>`.
  - If `Scale: high`, you MUST include a `<perf>` section in the `implement` phase to justify tool choice and state Big-O complexity.
- **Stepped Execution (High Risk/Architecture):** Must strictly pause for user approval between `spec`, `implement`, and `verify` phases.

## Commit Boundary
Commit tasks are not auto-triaged. Use commit rules only when explicitly requested by the user. Do not route through the default skill workflow.

## Output
```md
### 🛡️ Triage: [Type]
- **Risk/Uncertainty:** `[Risk]` / `[Uncertainty]` | **Quant:** `[Active/None]`
- **Strategy:** `[Strategy]` ➔ `[Path]`
- **Requirement:** Spec: `[Yes/No]`, Review: `[Yes/No]`
- **Blocking:** [Questions or None]
```