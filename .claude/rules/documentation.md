# Documentation & AI Workflow Guide

## 0. Purpose of Documentation (Strict Separation from Spec)
- **This is PERMANENT KNOWLEDGE:** Documents here (`docs/architecture/`, `docs/domains/`) are the Single Source of Truth (SSOT) for the system's logic and architecture.
- **NOT a Work Order:** Do NOT write implementation steps, code diffs, or surgical plans here. (Those belong in ephemeral `docs/specs/*.md` files).
- **Format Rule:** To save tokens and optimize AI reading, absolutely **AVOID long paragraphs**. Use bullet points, Mermaid diagrams, tables, and strict interfaces.

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
change_triggers: [Array of globs for audit triggers]
dependencies: { documents: [Array of related docs] }
last_verified: [YYYY-MM-DD]
```

## 3. Standard Templates (AI-Optimized Structure)
When writing or updating documents, follow this strict, token-efficient structure:
- **`1. Overview`**: 1-2 lines maximum summary.
- **`2. Core Components`**: Use lists mapping `[Component Name] -> [Responsibility] -> [File Path]`.
- **`3. Data Flow`**: Use `mermaid` graphs or simple text arrows (`A -> B -> C`).
- **`4. Business Rules & Invariants`**: Bullet points of hard rules (e.g., "Must never exceed X amount").
- **`5. Data Schemas`**: Only core fields, JSON/TypeScript interfaces preferred.
- **`6. Testing Expectations`**: Which conditions must be covered in tests.

## 4. AI Document Lifecycle & Integration
- **[scan] Selective Loading:** Index only documents matching `change_triggers`. Read only necessary headings (e.g., `Business Rules`).
- **Knowledge Update (Post-Spec):** If a temporary blueprint (`docs/specs/`) introduces new business rules or architectural changes, you MUST update the relevant official document here FIRST, and then the temporary spec file must be deleted.
- **Anti-Sprawl & SSOT:** Do not duplicate business logic across multiple documentation files. Define common rules in architecture docs and reference them in domain docs.
- **Archiving:** Mark obsolete logic as `status: deprecated` and specify `replaced_by` immediately.
