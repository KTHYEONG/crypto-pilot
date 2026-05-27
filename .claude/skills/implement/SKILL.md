---
name: implement
description: Implement changes by strictly following the blueprint with surgical precision and token efficiency.
---

# implement

Focus on execution. Use surgical tools to minimize token waste.

## Purpose
Apply the implementation blueprint with 100% fidelity. You are the builder who translates the architect's specs into working code.

## Rules
1.  **Blueprint Source-of-Truth:** You MUST read the blueprint in `.agents/specs/` using `read_file` before starting. Follow the `Symbolic Interface Definitions` and `Step-by-Step Logic` exactly.
2.  **Surgical Precision:** ALWAYS prefer the `replace` tool over overwriting entire files to preserve context and minimize token usage.
3.  **No Unrelated Changes:** Do not refactor or clean up code outside the `Surgical Edit Plan` defined in the blueprint.
4.  **Minimalist Chat Output:** Avoid verbose explanations. Focus on the "Blueprint Alignment" checklist.

## Implementation Workflow
1.  **Read:** Get the blueprint from `.agents/specs/`.
2.  **Act:** Apply changes using `replace` or `write_file` as per the `Surgical Edit Plan`.
3.  **Sync:** Update relevant `docs/` files to match the new implementation.

## Output (Token Optimized)
```md
### 🏗️ Implement: [Blueprint Name]
- **Target Files:** `[Paths]`
- **Blueprint Alignment:**
  - [ ] Step 1: [Short name from Blueprint]
  - [ ] Step 2: [Short name from Blueprint]
- **Docs Sync:** `[Updated Path]`
- **Handoff:** Ready for `verify` (Snippet: `[Snippet Command]`)
```
