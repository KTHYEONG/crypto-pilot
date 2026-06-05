---
name: scan
description: Scout and map relevant file paths (code, tests, docs) BEFORE any design work.
---

# Skill: Scan (File Discovery & Mapping)

## Purpose
Act as a lightweight scout. Your ONLY goal is to locate the exact files, tests, and documentation related to the user's request. **DO NOT design solutions, analyze logic, or formulate strategies.** 

## Scout Guidelines (Strict Token Efficiency)
1. **Search Only:** Rely heavily on `grep_search`, `glob`, and `list_directory`. 
2. **Zero Logic Analysis:** Do not read entire files. If you find the target class/function via grep, record its path and line number, then immediately stop reading.
3. **Trace Ecosystem:** Always find the Holy Trinity for the target:
   - **Core Code:** Where is the logic defined?
   - **Tests:** Where is the `test_*.py` file for it?
   - **Docs:** Which `docs/domains/*.md` or `docs/architecture/*.md` governs this?
4. **No Guesses:** If you cannot find a related test or doc, explicitly state `None found`. Do not hallucinate paths.

## Output Format
```md
### 🔍 File Exploration Completed

**1. Core Target Files**
- `[File Path]` (Line: X) - *[Short Description: e.g., ML Gate logic definition]*

**2. Related Ecosystem**
- **Dependencies/References:** `[Related module path]`
- **Test Files:** `[test_*.py path]` (or 'None found')
- **Related Documentation:** `[docs/*.md path]` (or 'None found')

**3. Next Step**
- ➡️ Proceed to `spec` skill for detailed design based on the discovered paths.
```
