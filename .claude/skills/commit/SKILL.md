---
name: commit
description: Analyze modifications and execute git commits directly.
---

# Commit

Automated git execution protocol. Partitions modifications into clean logical units and executes commits with strict Why/What context.

## Rigid Operational Directives

- **Direct Execution**: Immediately run `git add` and `git commit` commands to persist changes. Do NOT output markdown drafts for user approval.
- **Security First**: If secrets/API keys are detected in diff, STOP immediately and report: `🚨 SECURITY ALERT: Secrets detected in diff.`
- **Auto-Splitting Criteria**:
  - **Single Commit**: Triggered if changes are <= 10 files, <= 300 lines, and in the same logical layer.
  - **Multi-Commit**: Triggered if changes > 10 files, > 300 lines, or cross layer boundaries (e.g. config vs src). Execute consecutive split commits.

## Deterministic 3-Step Execution Pipeline

- [ ] **Step 1 (Pre-check)**: Run `git status --short` to inspect untracked/modified files and check for secret leaks.
- [ ] **Step 2 (Staging)**: Stage files by logical unit (`git add <files>`).
- [ ] **Step 3 (Commit)**: Execute `git commit -m "<Subject>" -m "- **Why:** <Reason>" -m "- **What:** <Details>"`.

## Message Formatting Standard (Strict Why/What Context)

1. **Subject**: Standard conventional commit type (`feat`, `fix`, `refactor`, `chore`, `docs`) + concise Korean summary (<= 50 chars, no trailing period).
2. **Double-Bulleted Body**:
   - `- **Why:** <Include Task_ID/ADR reference or concrete metric/bug cause ending with "~함." (e.g. TASK-0730 ADR 설계 규격에 맞춰 변동성 지표 오차를 해결함.)>`
   - `- **What:** <Explain exact logic/structure changes in Korean ending with "~함." (e.g. MAD-Z 동적 스케일링 알고리즘 추가함.)>`
3. **Language**: Subject/Body MUST be in Korean with noun-form termination (`~함.`). No AI attribution.


## Output Format

Return ONLY this 2-line summary card:

📌 `[COMMIT COMPLETE] Total: <commit_count> commit(s)`
`• [<short_hash>] <subject> (<file_count> files)`

