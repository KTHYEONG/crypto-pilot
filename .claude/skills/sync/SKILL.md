---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync

Automated task synchronization, ADR logging, and artifact cleanup protocol.

## Execution Rules

Execute task sync via script (keep `--why`, `--what`, `--impact` strictly to 1 concise sentence each):
```bash
uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context>" --what "<Resolution>" --impact "<Impact>" --source src/x.py
```

- **Spec Cleanup Execution Rule**: `docs/specs/` files MUST be automatically removed by default. Do NOT pass `--keep-all-specs` unless explicitly requested by the user in prompt.

## Manual Steps (Only if applicable)
- Surgically update `docs/architecture/` if architectural contracts changed.
- Insert `[ADR_YYYYMMDD_TaskID]` tag in modified class/fn docstrings.

## Output Format

Do NOT repeat logs or document text in response. Return ONLY this 2-line status summary:

🔄 `[SYNC COMPLETE] Task: <TASK_ID> - <Title>`
`ADR: <Logged> | Specs Cleared: <Count/All> | Scratch: <Cleared> | Arch Docs: <Updated/None>`

