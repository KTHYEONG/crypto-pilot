---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync

Automated task synchronization, ADR logging, Smart JSON index registration, and artifact cleanup protocol.

## Execution Rules

Execute task sync via script (keep `--why`, `--what`, `--impact` strictly to 1 concise sentence each):
```bash
uv run python tools/agent_skills/sync_task.py --task TASK_ID --title "<Title>" --why "<Context>" --what "<Resolution>" --impact "<Impact>" --source src/x.py --domain <signal/risk/execution>
```

- **Spec Cleanup Execution Rule**: `docs/specs/` files MUST be automatically removed by default. Do NOT pass `--keep-all-specs` unless explicitly requested by the user in prompt.
- **Scratch Cleanup Execution Rule**: All temporary files under `scratch/` directory MUST be automatically removed during task sync.
- **Smart Registry Auto-Update**: Script automatically updates `docs/decisions/task_index.json` and `docs/code_map.json`.

## Manual Steps (Only if applicable)
- Surgically update `docs/architecture/` if architectural contracts changed.
- Insert `[ADR_YYYYMMDD_TaskID]` tag in modified class/fn docstrings.

## Output Format

Do NOT repeat logs or document text in response. Return ONLY this compact card:

### 📌 [SYNC] <Task / ADR Title>

- **Status**: COMPLETE
- **ADR**: <ADR_ID | None>
- **Details**: Indexes updated | Specs & Scratch cleared


