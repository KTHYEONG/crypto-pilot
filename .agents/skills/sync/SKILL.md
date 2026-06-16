---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Enforce the Single Source of Truth (SSOT). Finalize the task by promoting ephemeral implementation knowledge into official documentation and purging all temporary artifacts to maintain a clean workspace.

## Execution Rules

### 1. Truth Promotion
- **Conceptual Distinction**:
  - **EH (docs/decisions/)**: "The Why" - Logs decisions, deltas, and rationale to provide context for future AI agents.
  - **Architecture (docs/architecture/)**: "The What" - Represents the current, verified state of the system structure.
- **Architecture Consolidation (NO BLOAT)**:
  - **Primary Files Only**: Strictly use primary domain files (e.g., `layer1.md`, `layer2.md`, `universe.md`).
  - **Update Policy**: Do NOT create new architecture files for sub-features. Locate the primary domain file and update the relevant section or append a new sub-section.
  - **On-Touch Cleanup**: Only perform minor structural cleanup on the *file being modified* if it exceeds 500 lines or becomes unreadable. Avoid global refactoring.
- **ADR Consolidation (EH Update)**: 
  - **Append to Domain EH**: Always append new ADR entries to the existing domain-specific EH file (e.g., `layer1-eh.md`). Prohibit new EH file creation.
  - **Content**: Log Delta and Rationale concisely for future AI context.

### 2. Zero-Residue Cleanup (CRITICAL)
- **Purge Specs**: Proactively delete all `.md` files in `docs/specs/` related to the current task ID or feature.
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
- ✅ Workspace is synchronized and clean. Proceed to `commit`.
```
s/architecture/*.md]`
- **Logged ADR:** `[Path to docs/decisions/*.md]`

**2. Cleanup State**
- **Deleted Specs:** `[File names]`
- **Removed Artifacts:** `[List of purged files]`

**3. Final Status**
- ✅ Workspace is synchronized and clean. Proceed to `commit`.
```
