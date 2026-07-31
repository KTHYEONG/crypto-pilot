---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync

Automated task synchronization, ADR logging, Smart JSON index/Anti-pattern registration, and artifact cleanup protocol.

## Execution Rules

Execute task sync via script (keep `--why`, `--what`, `--impact` strictly to 1 concise sentence each):
```bash
uv run python scripts/sync_task.py --task TASK_ID --title "<Title>" --why "<Context>" --what "<Resolution>" --impact "<Impact>" --source src/x.py --domain <signal/risk/execution> [--failed-hypothesis "<Hypothesis>" --failure-reason "<Reason>"]
```

- **Spec Cleanup Execution Rule**: `docs/specs/` files MUST be automatically removed by default. Do NOT pass `--keep-all-specs` unless explicitly requested by the user in prompt.
- **Scratch Cleanup Execution Rule**: All temporary files under `scratch/` directory MUST be automatically removed during task sync.
- **Smart Registry Auto-Update**: Script automatically updates `docs/decisions/task_index.json`, `docs/code_map.json`, and `docs/decisions/anti_patterns.json`.

## Manual Steps (Only if applicable)
- Surgically update `docs/architecture/` if architectural contracts changed.
- Insert `[ADR_YYYYMMDD_TaskID]` tag in modified class/fn docstrings.

## Output Format

Do NOT repeat logs or document text in response. Return ONLY this compact card:

### 📌 [SYNC] <Task / ADR Title>

- **Status**: COMPLETE
- **ADR**: <ADR_ID | None>
- **Details**: Indexes updated | Specs & Scratch cleared


