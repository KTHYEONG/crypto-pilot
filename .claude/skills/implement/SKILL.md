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
- Preserve behavior unless change is requested.
- Hand off to `verify` after implementation.

## Output
```md
<plan>
- ...
</plan>

<perf>
(Only if Scale is 'high')
- Primary Bottleneck: CPU (Math) / Network I/O / Disk I/O / Memory?
- Optimization Strategy & Tool Justification: (e.g., Vectorization, Async/Caching, Batching, Connection Pooling)
- Complexity: Time (O), Space (O)
- Trade-offs: What is sacrificed? (e.g., higher memory usage for caching, readability for Numba)
</perf>

<risk>
- ...
</risk>

## Implementation
- Strategy:
- Changed Files:
- Notes:
- Verification Handoff:
```