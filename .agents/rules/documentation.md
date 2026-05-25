# Documentation & AI Workflow Guide

## 1. Core Principles & Trust Order
- **Doc/Code Synchronization:** When modifying code, related documentation MUST be updated in the same transaction. Discrepancies are treated as stale risks and require re-validation.
- **Trust Order:** 1. Active Architecture -> 2. Active Domains -> 3. ADRs -> 4. Code -> 5. Tests -> 6. Deprecated.
- **Structure:** `docs/architecture/` (Systems/Engines), `docs/domains/` (Business Logic/Strategies), `docs/decisions/` (ADR).

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

## 3. Standard Templates (Required Sections)
- **Domain/Architecture:** `1. Overview` -> `2. Core Components` -> `3. Data Flow` -> `4. Business Rules` -> `5. Detailed Specifications` (Specs, Schemas, Constants, etc.) -> `6. Examples` -> `7. Testing Expectations`
- **ADR:** `Status` -> `Context` -> `Decision` -> `Alternatives` -> `Consequences` (Positive/Negative)
- **Deprecated:** Must specify `status: deprecated` and `replaced_by: [path]` in the Frontmatter.

## 4. AI Skill Pipeline & Token Optimization
Operational guidelines for the AI pipeline (`triage-scan -> spec -> implement -> verify -> review`).

- **[triage-scan] Selective Loading:**
  - Index only documents matching `change_triggers`. For large documents, load only required headings (e.g., Business Rules) to optimize the context window.
- **[implement] Mechanical Co-Modification:**
  - **Prohibitions:** Adding undocumented behavior or implementing based on assumed business logic.
  - **Mandates:** When modifying Data flow, Public API, or Invariants, you MUST update the document body and the `last_verified` date.
- **[spec -> documentation] Promotion Lifecycle:**
  - Temporary designs (ADR, PRD, etc.) created via the `spec` skill must be merged (Promoted) into official documents (e.g., `docs/domains/`) upon completion, and temporary files must be deleted to prevent fragmentation.
- **[Maintenance] Anti-Sprawl & SSOT:**
  - **Anti-Sprawl:** Integrate new features into existing domain documents whenever possible. Consider splitting files only if they exceed 500 lines.
  - **SSOT (Single Source of Truth):** Do not duplicate the same business logic across multiple documents. Define common rules in architecture docs and reference them in domain docs.
  - **Archiving:** Mark obsolete logic as `status: deprecated` and specify `replaced_by` immediately to isolate AI from incorrect context.
- **[review] Hard Block Rules:**
  - **Request changes immediately if:** 1) `last_verified` date is not updated after code changes. 2) Documented `Invariants` are violated. 3) New exceptions are missing from `Edge Cases`. 4) New code references `deprecated` logic.
