---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Enforce the Single Source of Truth (SSOT). Finalize the task by promoting ephemeral implementation knowledge into official documentation, updating the global index, and purging temporary artifacts.

## Execution Rules

### 1. Truth Promotion & Knowledge Anchoring
- **Conceptual Distinction (Crucial)**:
  - **Architecture (docs/architecture/)**: **"The What/Current State"**. Represents static verified system structure. Use **Present Tense** only.
  - **Decisions (docs/decisions/)**: **"The Why/How/History"**. Two-file log architecture. Use **Past Tense** only.
- **In-Code ADR Referencing**:
  - Insert a brief absolute tag `[ADR_YYYYMMDD_TaskID]` directly into the Docstrings of modified source classes/functions. This links the code directly to its architectural history without polluting the logic.
- **Dependency Indexing (docs/index.json)**:
  - Update `docs/index.json` to map any modified or newly created source files directly to their corresponding architecture documents and tests. Avoid recording individual ADR mappings here.
- **Two-File Decisions Log & Sliding Window**:
  - **decisions.md (Active Window)**: Append new ADR entries to the **top** of `docs/decisions/decisions.md`. Maintain a maximum of **15 active entries**.
  - **Max 5 Lines Rule**: Keep every entry highly condensed: `[Date] [Task ID] [Title] - Context/Why (2 lines) - Resolution/What (2 lines) - Impact (1 line)`. Maximum 5 lines overall.
  - **decisions_archive.md (Permanent Archive)**: When `decisions.md` exceeds 15 entries, relocate the oldest excess entries to the **top** of `docs/decisions/decisions_archive.md`. Do not create multiple date-based archive files.

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
