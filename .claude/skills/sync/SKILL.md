---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync
## Automated
Run: `uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>" --source src/x.py [--test tests/...] [--doc docs/architecture/...] [--keep-specs spec_file.md ...] [--keep-all-specs]`
Script handles: ADR append to decisions.md, archive pruning, index.json update, spec file cleanup (with option to preserve specified or all spec files via `--keep-specs` or `--keep-all-specs`).

## Manual
- **Architecture docs** (`docs/architecture/`): surgically edit existing sections only. Format rules in AGENTS.md §12.
- **In-code ADR tag**: Insert `[ADR_YYYYMMDD_TaskID]` into modified class/fn docstrings.
- **Verify clean state & Spec/Scratch cleanup**: Run `git status`. Clean up temporary files under `scratch/` or unneeded `docs/specs/` files (except those explicitly designated for retention). No untracked temp files should remain.
