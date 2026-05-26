---
name: triage-scan
description: Classify the task and inspect repository context in a single step before design.
---

# triage-scan

**STRICT OBSERVATION MODE ONLY.** Do not edit files. Do not design solutions.

## 🚫 Forbidden Actions (HARD CONSTRAINTS)
- **NO SOLUTIONS:** Proposing, designing, or suggesting technical solutions, code changes, or logic improvements is **STRICTLY PROHIBITED**.
- **NO SNIPPETS:** Do not provide code examples or "how-to" snippets.
- **NO REFACTORING ADVICE:** Do not suggest cleaner ways to write existing code. 
- **NO SPECULATION:** Do not guess how a bug might be fixed. Defer all "How" and "Why" (solution-wise) to the `spec` phase.

## 🎯 Purpose
Your ONLY job is **Discovery & Classification**. You are a scout mapping the terrain, not an engineer building a bridge. Gather facts to empower the next phase (`spec` or `implement`).

## ⚖️ Decide (Triage)
Classify based on the request:
- **Task type:** bug | feature | refactor | UI/style | config/build | docs | test-only | architecture | quant
- **Risk/Uncertainty:** low | medium | high
- **Scale:** high (data/IO intensive) | low
- **Quant context:** active | none
- **Strategy:** direct patch | regression-first | TDD | characterization-first | spike | docs-only

## 🔎 Inspect (Scan - Factual Gathering)
Gather raw data. Do not interpret or improve.
- **Relevant Docs:** Search `docs/` for architecture, domains, and decisions.
- **Relevant Files:** Identify paths that *will* be touched or referenced.
- **Existing Patterns:** Document "what exists now" (e.g., "Uses X library for Y").
- **Test Locations:** Locate where new tests should go or where existing tests reside.
- **Public Interfaces:** Note signatures/APIs that might be affected.

## 📋 Self-Correction Checklist (Before Output)
1. Did I suggest a code change? (If yes, DELETE IT)
2. Did I explain *how* to fix the problem? (If yes, DELETE IT)
3. Is my output 100% focused on "What" and "Where"? (If no, REWRITE IT)

## 📤 Output Format
```md
### 🛡️ Triage & Scan: [Type]
- **Risk/Uncertainty:** `[Risk]` / `[Uncertainty]` | **Quant:** `[Active/None]`
- **Strategy:** `[Strategy]` ➔ `[Path]`

#### 🔎 Context Gathered (Facts Only)
- **Relevant Docs:** [Links]
- **Relevant Files:** [Paths]
- **Existing Patterns:** [Observation of current state]
- **Test Locations:** [Paths]
- **Public Interfaces:** [Current Signatures]

- **Next Action:** [Pause for user to invoke `spec` or proceed to `implement`]
- **Blocking:** [Missing information or None]
```