---
name: implement
description: Implement a scoped change using the selected strategy. TDD is optional.
---

# implement

Edit only task-relevant files.

## Purpose
Apply the smallest correct change.

## Strategies
- `direct patch`: low-risk local changes
- `regression-first`: bug fixes
- `TDD`: high-risk logic, auth, permissions, billing, data integrity, quant logic
- `characterization-first`: refactoring
- `spike`: unclear feasibility

## Rules
- Do not broaden scope.
- Do not perform unrelated refactors.
- Do not change public APIs unless required.
- Always update relevant architecture and domain documents in docs/ alongside code changes.
- **Promotion Lifecycle:** You MUST promote approved specs (ADR, PRD, etc.) into the '5. Detailed Specifications' section of the target architecture/domain document and update the 'last_verified' Frontmatter date.
- Preserve behavior unless change is requested.
- Hand off to `verify` after implementation.

## Output
```md
<perf>
(Only if Scale is 'high')
- **Bottleneck:** CPU / IO / Memory | **Complexity:** T(O), S(O)
- **Strategy:** [Tool/Method]
- **Trade-offs:** [Sacrifices]
</perf>

### 🏗️ Implementation: [Strategy]
- **Changes:** `[Files]`
- **Docs:** `[Updated/Verified]` | **Handoff:** `verify`
- **Notes:** [Details]
```* [Sacrifices]
</perf>

<risk>
- ...
</risk>

### 🏗️ Implementation: [Strategy]
- **Changes:** `[Files]`
- **Docs:** `[Updated/Verified]` | **Handoff:** `verify`
- **Notes:** [Details]
```