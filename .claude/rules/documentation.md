---
trigger:
  - on_file_path_regex: "docs/.*\\.md"
priority: 7
---

# Documentation Separation & Maintenance Policy

## 1. Docstrings
- **Standard:** Follow Google Style Docstrings for all public classes, methods, and functions.

## 2. Architecture Documentation (`docs/architecture/`)
- **Purpose:** "AI-First Structured Constraints". Contains system boundary, LaTeX mathematical formalisms, strict I/O tables, and Mermaid topology.
- **Line Limit:** Keep each document strictly under 300 lines.
- **Surgical Update Only:** Never append raw text to architecture files. Edit existing tables, schemas, or Mermaid nodes inline.
- **Prohibitions:** Omit procedural logic, code optimization details, logging policies, conversational prose, temporal examples, change history, and `[ADR_...]` tags.
- **Contract Priority:** In case of mismatch, in-code Type/Protocol definitions strictly supersede external markdown files.

## 3. Decisions Log Architecture (`docs/decisions/`)
- **Active Window (`decisions.md`):** Cumulative log. Max 5 lines per task appended to the top. Max 15 active entries.
- **Archive (`decisions_archive.md`):** Managed automatically via `python scripts/archive_decisions.py --max-entries 15`. No manual edits.
- **Sync Workflow:** The `sync` skill appends implementation decisions to `decisions.md` and triggers the archiving script automatically.
