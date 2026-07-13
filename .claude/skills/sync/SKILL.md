---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Enforce the Single Source of Truth (SSOT). Finalize the task by promoting ephemeral implementation knowledge into official documentation, updating the global index, and purging temporary artifacts.

## Execution Rules

### 1. Truth Promotion & Knowledge Anchoring
- **Conceptual Distinction & Surgical Boundaries (Crucial)**:
  - **Architecture (docs/architecture/)**: **"The What/Current Static State"** (SSOT).
    - **Surgical Update Only & Token Saving**: Never append raw text or specs directly to the end of architecture files. You must surgically edit existing tables, schemas, or Mermaid nodes to match the file's current layout. **Do NOT load the entire document; use view_file with specific line ranges to read and edit only the targeted sections.**
    - **Strict Content Restriction**: Only include public API signatures (I/O contracts), core state transitions, domain formulas, and data structures.
    - **No Implementation/History**: Do not include implementation guides, step-by-step logic, temporal examples, change history, or `[ADR_...]` tags in architecture files (these belong only in `decisions.md`).
  - **Decisions (docs/decisions/)**: **"The Why/How/History"** (ADR).
    - Isolate and record technical options, implementation context, work progress, and compromises at specific points in time.
- **In-Code ADR Referencing**:
  - Insert a brief absolute tag `[ADR_YYYYMMDD_TaskID]` directly into the Docstrings of modified source classes/functions. This links the code directly to its architectural history without polluting the logic.
- **Dependency Indexing (docs/index.json)**:
  - **Automated Dependency Indexing**: Update `docs/index.json` using `uv run python scripts/update_index.py --source <source_file> [--test <test_file>] [--doc <doc_file>]` rather than manual edits.
- **Two-File Decisions Log & Sliding Window (Automated)**:
  - **Context Alignment Obligation**: Before running the script, you **MUST** read the top 3 entries of `docs/decisions/decisions.md` (using a narrow line range) to align on the recent architectural decisions and check for consistency/conflicts.
  - **Do NOT edit decisions.md manually.** Execute the automated script to insert the ADR and handle pruning/archiving in one command:
    `uv run python scripts/add_adr.py --task "<TASK_ID>" --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>"`


### 2. Zero-Residue Cleanup (CRITICAL)
- **Purge Specs**: Proactively delete all `.md` files in `docs/specs/` related to the current task ID or feature. **Keep the `docs/specs/` directory itself; do not remove the folder.**
- **Wipe Artifacts**: Remove any temporary test data, logs, or intermediate files (`.tmp`, `.bak`) generated during `implement` or `check`.
- **Verify Clean State**: Ensure no new untracked files (except legitimate docs) remain in the workspace.

### 3. Single Responsibility
- Do not re-test or re-verify. Focus exclusively on Document/Index Synchronization and Workspace Hygiene.

## Output Format (Sync Manifest)
```markdown
### 🏁 [SYNC:OK] [ADR_ID] | Promoted: [updated architecture files] | Indexed: [updated source files] | Pruned: [Yes (Append to decisions_archive.md) / No] | Cleaned: spec
```
