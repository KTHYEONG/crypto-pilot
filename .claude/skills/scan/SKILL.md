---
name: scan
description: Scout and map relevant file paths (code, tests, docs) BEFORE any design work.
---

# Skill: Scan (File Discovery & Mapping)

## Purpose
Act as a high-speed scout to generate a "Context Manifest". Your only goal is to provide the `spec` agent with the exact URI and functional purpose of relevant files, minimizing redundant exploration in later phases.

## Scout Guidelines (Strict AI Efficiency)
1. **Search & Discovery Strategy**:
   - **Primary (High Precision)**: Prioritize Serena MCP tools (`find_symbol`, `find_file`, `get_symbols_overview`) to map internal structures without reading raw bytes.
   - **Secondary (Broad Coverage)**: Use `grep_search` or `glob` to find indirect references or usage patterns.
2. **Zero-Read Policy:** Do NOT read file bodies. Identify the target, confirm its existence and signature via MCP, and move on.
3. **The Manifest (Holy Trinity Mapping)**:
   - For every feature request, you MUST locate:
     - **Implementation (Core):** The primary logic file.
     - **Validation (Tests):** The corresponding `test_*.py` file.
     - **Context (Docs):** The governing `docs/architecture/` or `docs/domains/` file.
4. **No Hallucinations:** If a test or doc is missing, explicitly mark it as `MISSING`.

## Output Format (Scan Manifest)
```md
### [SCAN_MANIFEST]

**1. Primary Targets**
- `[Path]` (Symbol: [X]) : [One-sentence functional purpose]

**2. Related Ecosystem**
- **Tests:** `[Path]` (Status: Found/Missing)
- **Documentation:** `[Path]` (Status: Found/Missing)
- **Dependencies:** `[Path]` (Key dependency for this task)
```
