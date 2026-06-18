---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Enforce the Single Source of Truth (SSOT). Finalize the task by promoting ephemeral implementation knowledge into official documentation and purging all temporary artifacts to maintain a clean workspace.

## Execution Rules

### 1. Truth Promotion (STRICT SEPARATION)
- **Conceptual Distinction (Crucial)**:
  - **Architecture (docs/architecture/)**: **"The What/Current State"**. Represents the static, verified system structure. Use **Present Tense** only.
  - **EH (docs/decisions/)**: **"The Why/How/History"**. Logs deltas, and rationale. Use **Past Tense** only.
- **Architecture Consolidation (NO LOGS/NO HISTORY)**:
  - **State-based Updates**: Transform implementation knowledge into system states. **Prohibit** using dates (`YYYY-MM-DD`), past-tense verbs (`fixed`, `added`, `improved`), or phrases like "Updated to...".
  - **Integration over Appending**: Do NOT simply append a new section for a task. Locate the relevant logical section (formulas, tables, diagrams) and **merge** the changes into the existing structure.
  - **Primary Files Only**: Strictly use primary domain files. Do NOT create new architecture files for sub-features.
- **ADR Consolidation (EH Update)**: 
  - **Append to Domain EH**: Always append new ADR entries to the existing domain-specific EH file.
  - **Content**: Log Delta and Rationale concisely for future AI context. Focus on "Why it changed".

### 2. Zero-Residue Cleanup (CRITICAL)
- **Purge Specs**: Proactively delete all `.md` files in `docs/specs/` related to the current task ID or feature. **Keep the `docs/specs/` directory itself; do not remove the folder.**
- **Wipe Artifacts**: Remove any temporary test data, logs, or intermediate files (`.tmp`, `.bak`) generated during `implement` or `check`.
- **Verify Clean State**: Ensure no new untracked files (except legitimate docs) remain in the workspace.

### 3. Single Responsibility
- Do not re-test or re-audit. Focus exclusively on Document Synchronization and Workspace Hygiene.

## Output Format (Sync Manifest)
```md
### [SYNC_MANIFEST]

**1. Documentation State**
- **Promoted:** `[List of updated docs/architecture/*.md]`
- **Consolidated ADR:** `[Path to the REUSED docs/decisions/*.md]`

**2. Cleanup State**
- **Deleted Specs:** `[File names]`
- **Removed Artifacts:** `[List of purged files]`

**3. Final Status**
- ✅ Workspace is synchronized and clean.
```
