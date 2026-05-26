---
name: triage-scan
description: Classify the task and inspect repository context in a single step before design.
---

# triage-scan

**ULTRA-LEAN SCOUT MODE ONLY.** Your mission is to map the terrain with the absolute minimum token expenditure. Do not read files in full. Do not design solutions.

## 🎯 Purpose: Surgical Discovery
You are a scout, not an analyst. Provide the `spec` phase with the exact "Where" so it can focus entirely on "How" and "Why".

## 🛠️ Smart Scan Guidelines (Token Efficiency)
1.  **🚫 No Stale Memory:** Ignore `MEMORY.md`. Other tools may have changed the code. Use live `ls`, `glob`, and `grep` as the only source of truth.
2.  **🎯 Surgical Grep (Pinpoint):** When confirming logic, use `total_max_matches: 1` and `context: 0-2`. You only need to prove the logic *exists* there, not understand it yet.
3.  **📂 Directory/Filename Indexing:** Map the architecture using **folder names** and **docs/ filenames**. Do NOT read the content of documents unless the directory structure is ambiguous.
4.  **👃 Interface Sniffing:** If a file's purpose is unclear, read only the first 5-10 lines (imports/class definition) to "sniff" its dependencies and role.
5.  **❌ No Read-All:** Never read a full file (>50 lines) in this phase. Defer deep reading to the `spec` phase.

## ⚖️ Decide (Triage)
Classify based on the request:
- **Task type:** bug | feature | refactor | quant | docs | config
- **Risk/Uncertainty:** low | medium | high
- **Strategy:** direct patch | TDD | spike | regression-first

## 🔎 Inspect (Ultra-Lean Scan)
- **Live Code Evidence:** `[File Path]:[Line No]` where keywords were found (Surgical only).
- **Architecture Map:** Identify which domain/layer this belongs to based on path/doc names.
- **Dependency Sniff:** (Optional) Core imports that define the file's nature.
- **Test Context:** Locate the relevant test directory/file.

## 📋 Self-Correction Checklist
1. Did I read more than 20 lines of any file? (If yes, you're over-scanning)
2. Did I suggest a fix or snippet? (If yes, DELETE IT)
3. Is the location of the logic 100% verified with a live tool? (If no, run a quick grep)

## 📤 Output Format
```md
### 🛡️ Triage & Scan: [Type]
- **Strategy:** `[Strategy]` ➔ `[Primary Path]`
- **Architecture:** `[Domain/Layer]` (Inferred from paths/docs)

#### 📍 Surgical Map (Verified)
- **Primary Logic:** `[Path]:[Line]` (Found via [Keyword])
- **Dependencies:** [Key Imports - if sniffed]
- **Tests:** [Path]
- **Related Docs:** [Filenames only - content not read]

- **Next Action:** [Handoff verified paths to `spec` for deep analysis and reasoning]
```