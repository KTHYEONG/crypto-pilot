---
name: triage-scan
description: Classify tasks and map repository context efficiently before design.
---

# Skill: Triage-Scan

## Purpose
Quickly identify task type, strategy, and locate relevant code/tests with minimal token usage. Do not design or read full files.

## Scan Guidelines (Efficiency)
1. **Live Truth:** Use `ls`, `glob`, and `grep` only. Use live code as the source of truth.
2. **Pinpoint Grep:** Use `total_max_matches: 1` and minimal context to confirm existence, not deep logic.
3. **Index Mapping:** Map architecture via folder/file names. Read only the first 5-10 lines if the purpose is unclear.
4. **No Full Reads:** Never read >20 lines of any file. Defer deep analysis to the `spec` phase.

## Triage Classifications
- **Type:** bug | feature | refactor | quant | docs | config
- **Risk:** low | medium | high
- **Strategy:** direct patch | TDD | spike | regression-first

## Output Format
```md
### 🛡️ Triage-Scan: [Type]

**1. Triage**
- **Strategy:** `[Strategy]` (Risk: `[Risk]`)
- **Domain/Layer:** `[Layer Name]`

**2. Verified Map**
- **Logic:** `[Path]:[Line]` (via [Keyword])
- **Tests:** `[Path]`
- **Deps:** `[Key Imports/Classes]`
- **Related:** `[Relevant Filenames]`

**3. Next Step**
- [Hand off verified paths to `spec` or `implement`]
```