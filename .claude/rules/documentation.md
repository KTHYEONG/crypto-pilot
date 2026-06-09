# Documentation & AI Workflow Guide

## 0. Purpose of Documentation (Strict Separation from Spec)
- **This is PERMANENT KNOWLEDGE:** Documents here (`docs/architecture/`, `docs/domains/`) are the Single Source of Truth (SSOT) for the system's logic and architecture.
- **NOT a Work Order:** Do NOT write implementation steps, code diffs, or surgical plans here. (Those belong in ephemeral `docs/specs/*.md` files).
- **Format Rule:** To save tokens and optimize AI reading, absolutely **AVOID long paragraphs**. Use bullet points, Mermaid diagrams, tables, and strict interfaces.

## 1. Core Principles & Trust Order
- **Doc/Code Synchronization:** When modifying code, related documentation MUST be updated in the same transaction. Discrepancies are treated as stale risks and require re-validation.
- **Trust Order:** 1. Active Architecture -> 2. Active Domains -> 3. ADRs (Decisions) -> 4. Code -> 5. Tests.
- **Strict Separation:** 
  - `docs/architecture/` = "What/Logic/Formulas". Static, concise, SSOT.
  - `docs/decisions/` = "Why/How/History". Cumulative, decision-centric, ADR.

## 2. Metadata Standard (Frontmatter)
Every core document must include the following YAML Frontmatter:
```yaml
title: [Title]
domain: [domain]
type: architecture | domain-spec | adr | guide
status: active | deprecated | proposal
priority: critical | high | medium | low
ai_read_policy: always | when_related | optional
related_paths: [Array of related code paths]
change_triggers: [Array of globs for review triggers]
dependencies: { documents: [Array of related docs] }
last_verified: [YYYY-MM-DD]
```

## 3. Standard Templates (AI-Optimized Structure)

### 3.1 Architecture Documents (`docs/architecture/*.md`)
- **Goal:** High-density, comprehensive readability. Both humans and AI must instantly understand the complete module structure.
- **Constraint:** NO implementation steps. NO history or "how we fixed it" prose.
- **`1. Purpose`**: 1-line statement of the module's exact role in the system.
- **`2. Core Logic & Math`**: Complete mathematical formulas ($y = f(x)$) or state machine logic.
- **`3. Architecture Flow`**: MUST use `mermaid` diagrams for visual structural mapping.
- **`4. Core Variables & I/O`**: Use Markdown Tables strictly for parameters, inputs, and outputs.

### 3.2 Decision Records (`docs/decisions/*.md`)
- **Goal:** Ultra-compressed logic history to prevent file bloat over multiple iterations.
- **Constraint:** Maximum 5-7 lines per entry. Focus purely on the *delta* (what changed) and *rationale* (why). Cumulative (newest at the TOP).
- **Format:**
  ```markdown
  ## [YYYY-MM-DD] [Topic]
  - **Delta:** [1 line: What exact logic/formula was changed or added]
  - **Rationale:** [1 line: Why it was needed (e.g., bug fix, edge case, new feature)]
  - **Edge Cases/Trade-offs:** [1-2 bullets: Critical safeguards added (e.g., zero-division defense)]
  ```

## 4. AI Document Lifecycle & Integration
- **[scan] Selective Loading:** Index only documents matching `change_triggers`. Read only necessary headings (e.g., `Business Rules`).
- **[audit] Knowledge Promotion (Post-Spec):** 
  1. If formulas or I/O changed: Update `docs/architecture/` (Strictly only formulas/interfaces).
  2. For all logic changes: Append a compressed ADR to `docs/decisions/`.
  3. Delete the temporary spec file (`docs/specs/`).
- **Anti-Sprawl & SSOT:** Do not duplicate business logic across multiple documentation files. Define common rules in architecture docs and reference them in domain docs.
- **Archiving:** Mark obsolete logic as `status: deprecated` and specify `replaced_by` immediately.

