---
name: sync
description: Documentation Synchronization, ADR Logging, and Cleanup.
---

# Skill: Sync (Knowledge Synchronizer)

## Purpose
Reflect implemented and verified knowledge into official system documentation and clean up temporary files created during the task to maintain the Single Source of Truth (SSOT).

## Execution Rules

### 1. Knowledge Promotion (Architecture Sync)
- **API and Schema Mapping**: Utilize Serena MCP (`get_symbols_overview`) to inspect the final shape of implemented classes, methods, and types. This ensures precise mapping of inputs/outputs in system documents.
- **Core Formulas**: Update core formulas or algorithms in the relevant documents within `docs/architecture/`.
- **Domain Logic**: Reflect changed domain rules or data flows. Use direct file reads only to pull out implementation blocks if strictly needed for code snippets.

### 2. ADR (Architectural Decision Record) Logging
- **Decision Capture**: Record significant design choices made during implementation in `docs/decisions/` as a compressed ADR (max 5 lines).
- **Context**: Focus on "why" this approach was chosen to provide context for future maintainers.

### 3. Workspace Cleanup
- **Spec Deletion**: Delete completed temporary specification documents (`docs/specs/*.md`).
- **Artifact Removal**: Clean up temporary data or log files generated during the task.

### 4. Single Responsibility (DO NOT OVERSTEP)
- You are ONLY the Documenter/Cleaner. Do not write or review code, and do not run tests. Focus entirely on updating the Architecture and ADRs.

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
- [Next Step: ✅ `commit` is ready.]
```
