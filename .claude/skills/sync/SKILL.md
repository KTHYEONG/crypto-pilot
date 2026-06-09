---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Reflect implemented and verified knowledge into official system documentation and clean up temporary files created during the task to maintain the Single Source of Truth (SSOT).

## Execution Rules

### 1. Knowledge Promotion (Architecture Sync)
- **Core Formulas**: Update core formulas or algorithms in the relevant documents within `docs/architecture/`.
- **Domain Logic**: Reflect changed domain rules or data flows.

### 2. ADR (Architectural Decision Record) Logging
- **Decision Capture**: Record significant design choices made during implementation in `docs/decisions/` as a compressed ADR (max 5 lines).
- **Context**: Focus on "why" this approach was chosen to provide context for future maintainers.

### 3. Workspace Cleanup
- **Spec Deletion**: Delete completed temporary specification documents (`docs/specs/*.md`).
- **Artifact Removal**: Clean up temporary data or log files generated during the task.

## Output Format
```md
### 🔄 System Knowledge Sync: [COMPLETE]

**1. Documentation Update**
- [ ] Updated `docs/architecture/*.md` (List files)
- [ ] Logged ADR in `docs/decisions/*.md`

**2. Cleanup**
- [ ] Deleted temporary spec: `docs/specs/*.md`
- [ ] Removed workspace artifacts

**3. Next Step**
- [Next Step: Proceed to COMMIT]
```
