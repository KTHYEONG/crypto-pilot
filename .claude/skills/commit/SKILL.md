---
name: commit
description: Analyze modifications and execute git commits directly.
---

# Skill: Commit (Automated Git Executer)

## 🎯 Purpose
Analyze unstaged modifications, automatically partition them into clean logical units, and execute git commits directly using a standardized, highly readable message format optimized for low-reasoning models.

## 🛠 Operational Mandate
- **Direct Execution:** DO NOT output markdown drafts waiting for user approval. Immediately run `git add` and `git commit` commands to persist changes.
- **Auto-Splitting Criteria:**
  - **Single Commit:** Triggered if modifications are within 10 files, 300 lines of change, and belong to the same logical layer.
  - **Multi-Commit:** Triggered if modifications exceed 10 files, 300 lines, or cross layer boundaries (e.g. config vs src). Automatically split file groups and execute consecutive commits.
- **Security First:** If secrets are detected, ABORT immediately and output: `🚨 SECURITY ALERT: Secrets detected in diff.`
- **Language & Standards:** Subject/Body must be in Korean. Maximum 50 characters for Subject. No AI attribution.

## 🧠 Message Formatting Standards (Strict Why/What Separation)
All generated commit messages MUST strictly adhere to the following layout to prevent vague, single-word, or unreadable logs:
1. **Subject:** Use standard conventional commit types (e.g., `feat`, `fix`, `refactor`, `chore`, `docs`) followed by a concise, 50-character Korean summary. No trailing period.
2. **Double-Bulleted Body (Mandatory):** The body MUST contain exactly two bullet points formatted with bold headers:
   - `- **Why:** <Explain the root cause or quantitative hypothesis in a complete Korean sentence ending with "~함.">`
   - `- **What:** <Explain the exact logic or structural changes applied in a complete Korean sentence ending with "~함.">`
3. **No Vague Words:** Sentence must end with "~함.". Vague terms like "버그 수정", "코드 개선" are strictly prohibited.

## 🧠 Internal Logic (Grouping & Structuring)
1. **Logical Grouping:**
   - **Build Integrity:** DO NOT split commits if it breaks the build or tests. Interface changes and their implementations MUST stay together.
   - **Dependency Coupling:** Keep logic changes and their direct dependency updates (e.g., `uv.lock`, `pyproject.toml`) in the same commit.
   - **By Type/Layer:** Group by feat/fix/refactor or layer, only if they are truly independent.

## 📋 Output Format (Minimalist Manifest)
```markdown
### 🏁 [COMMIT:OK]
- **Commit 1:** [hash] | [subject] (Target: [files])
- **Commit 2:** [hash] | [subject] (Target: [files])
```
