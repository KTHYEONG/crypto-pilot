---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync
## Automated
Run: `uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>" --source src/x.py [--test tests/...] [--doc docs/architecture/...] [--remove-specs spec_file_or_prefix ...] [--keep-all-specs]`

Script handles: ADR append to decisions.md, archive pruning, index.json update, scratch/temp file cleanup, and spec file cleanup.
> [!IMPORTANT]
> **Spec File Cleanup Execution Rule**:
> - `docs/specs/` files MUST be **automatically removed** during normal sync.
> - **DO NOT pass `--keep-all-specs` or `--keep-specs`** unless the user explicitly requested to preserve spec files in their prompt.
> - Default command format: `uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>" --source src/x.py` (Omitting keep flags to ensure clean specs).

## Manual
- **Architecture docs** (`docs/architecture/`): surgically edit existing sections only. Format rules in AGENTS.md §12.
- **In-code ADR tag**: Insert `[ADR_YYYYMMDD_TaskID]` into modified class/fn docstrings.
- **Verify clean state & Temp cleanup**: Run `git status`. Ensure clean up of temporary files under `scratch/`. Do NOT delete files in `docs/specs/` manually unless explicitly instructed by user. No untracked temp files should remain.
