---
name: commit
description: Analyze changes and generate logical, atomic commit messages or plans.
---

# Skill: Commit (Git Commit Analyst)

## 🎯 Purpose
Analyze `<diff>`, `git status`, and untracked files to organize changes into an efficient and logical commit structure. This skill helps maintain a clean, readable, and professional git history.

## 🛠 Operational Mandate
- **Atomic Commits:** Each commit in the plan must represent a single logical unit.
- **Action-Oriented:** Provide a concrete **Commit Plan** for diverse or large changes instead of just recommendations.
- **Security First:** If secrets are detected, ABORT immediately and output: `🚨 SECURITY ALERT: Secrets detected in diff.`
- **Language:** Subject and Body MUST be in Korean.

## 🧠 Internal Logic (Grouping & Structuring)
1. **Logical Grouping:**
   - **Build Integrity:** DO NOT split commits if it breaks the build or tests. Interface changes and their implementations MUST stay together.
   - **Dependency Coupling:** Keep logic changes and their direct dependency updates (e.g., `uv.lock`, `pyproject.toml`) in the same commit.
   - **By Type/Layer:** Group by feat/fix/refactor or layer, only if they are truly independent.
2. **Trigger for Multi-Commit Plan:**
   - Multiple independent change types or layers exist.
   - More than 10 files or 300 lines of change.
3. **Drafting Standards:**
   - **Subject:** Max 50 chars, imperative, no trailing period. (Korean)
   - **Body:** Focus on 'WHY' and 'HOW'. (Korean)
   - **Hunk Splitting:** If a single file contains multiple distinct changes, explicitly add `(Use git add -p)` next to the file name in the plan.

## 📋 Output Schema

### CASE A: Single Focused Change
🤖 **Suggested Commit Message**

```text
<type>(<scope>)[!]: <subject>

- <변경 이유>
- <구현 내용>

[BREAKING CHANGE: <description>]
[ISSUE: <id>]
```

### CASE B: Diverse or Large Changes (Commit Plan)
🤖 **Suggested Commit Plan**

**Commit 1: <type>(<scope>)**
- **Target Files:** `<file1>`, `<file2>`
- **Message:**
```text
<type>(<scope>): <subject>

- <변경 이유/내용 요약>

[BREAKING CHANGE: <description>]
[ISSUE: <id>]
```

**Commit 2: <type>(<scope>)**
- **Target Files:** `<file3>`
- **Message:**
```text
<type>(<scope>): <subject>

- <변경 이유/내용 요약>

[BREAKING CHANGE: <description>]
[ISSUE: <id>]
```

---
[Alternative Option - Provide ONLY if a different single-commit perspective is plausible]
```text
<type>(<scope>): <subject>
```
