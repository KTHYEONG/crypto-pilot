---
name: triage-scan
description: Classify the task and inspect repository context in a single step before design.
---

# triage-scan

Do not edit files.
**STRICT ROLE BOUNDARY:** Do NOT propose, design, or suggest technical solutions, code changes, or module improvements. Your ONLY job is classification, risk assessment, workflow routing, and fact-gathering. Defer all solution design to the `spec` phase.

## Purpose
Classify the task, determine the risk, and immediately scan the codebase to gather only the necessary factual context (files, docs, patterns).

## Decide (Triage)
- Task type: bug | feature | refactor | UI/style | config/build | docs | test-only | architecture | quant
- Risk: low | medium | high
- Uncertainty: low | medium | high
- Scale: high (heavy data, network I/O, massive loops) | low
- Quant context: active (if task involves math, stats, modeling, or backtesting) | none
- Strategy: direct patch | regression-first | TDD | characterization-first | spike | docs-only

## Routing
- Low risk: `triage-scan → implement → verify`
- Medium/High risk: `triage-scan → spec → implement → verify → review`

## Inspect (Scan)
Gather facts based on the initial request. Do not guess solutions.
- docs/ (architecture, domains, decisions)
- relevant files
- existing patterns
- tests
- public interfaces

## Output
```md
### 🛡️ Triage & Scan: [Type]
- **Risk/Uncertainty:** `[Risk]` / `[Uncertainty]` | **Quant:** `[Active/None]`
- **Strategy:** `[Strategy]` ➔ `[Path]`

#### 🔎 Context Gathered
- **Relevant Docs:** [Links]
- **Relevant Files:** [Paths]
- **Existing Patterns:** [Details]
- **Test Locations:** [Paths]
- **Public Interfaces:** [Details]

- **Next Action:** [e.g., Pause for user to invoke `spec` with a High-Reasoning Model (Medium/High Risk), or proceed directly to `implement` (Low Risk)]
- **Blocking:** [Questions or None]
```