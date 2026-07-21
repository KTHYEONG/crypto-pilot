---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync
## Automated
Run: `uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context/Why>" --what "<Resolution/What>" --impact "<Impact>" --source src/x.py [--test tests/...] [--doc docs/architecture/...]`
Script handles: ADR append to decisions.md, archive pruning, index.json update, spec file cleanup.

## Manual
- **Architecture docs** (`docs/architecture/`): surgically edit existing sections only. Format rules in AGENTS.md §12.
- **In-code ADR tag**: Insert `[ADR_YYYYMMDD_TaskID]` into modified class/fn docstrings.
- **Verify clean state & Spec/Scratch cleanup**: Run `git status`. If any `docs/specs/*.md`, `*_contract.json`, or files under the `scratch/` directory remain, delete them directly using `rm` command. No untracked files (except legitimate docs) should remain.
