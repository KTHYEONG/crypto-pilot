---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync
## Automated
Run: `uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>" --source src/x.py [--test tests/...] [--doc docs/architecture/...] [--remove-specs spec_file.md ...] [--clean-specs]`

Script handles: ADR append to decisions.md, archive pruning, index.json update, scratch/temp file cleanup.
> [!NOTE]
> **Spec File Cleanup Safety**:
> - `docs/specs/` files are **preserved by default**.
> - To delete specific completed/obsolete spec files, pass `--remove-specs spec_a.md spec_b.md`.
> - To clean unpreserved specs, pass `--clean-specs` (with optional `--keep-specs ...`).

## Manual
- **Architecture docs** (`docs/architecture/`): surgically edit existing sections only. Format rules in AGENTS.md §12.
- **In-code ADR tag**: Insert `[ADR_YYYYMMDD_TaskID]` into modified class/fn docstrings.
- **Verify clean state & Temp cleanup**: Run `git status`. Ensure clean up of temporary files under `scratch/`. Do NOT delete files in `docs/specs/` manually unless explicitly instructed by user. No untracked temp files should remain.
