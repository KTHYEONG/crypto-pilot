---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec

Frontier reasoning model protocol (Specification Engineering & Empirical Validation). Owns architectural decisions, trade-offs, and machine-readable contracts.

## Core Directives & Constraints (KISS Principle)

- **Goal**: Define the single optimal architecture and frozen contract so implementation requires zero redesign.
- **Constraints**:
  - Preserve causal timestamps, cost/funding accounting, numerical stability, and fail-closed behavior.
  - Do not relax thresholds or public contracts merely to make a test pass.
  - Keep main spec under 500 lines. Move large domain details to linked reference files.
  - **Mandatory Concrete Plan**: Every spec artifact MUST define an actionable implementation plan (target code/file changes, new contract, or concrete next step). Even if current hypotheses fail, do NOT end as a mere status report—specify the exact next architecture/hypothesis to attempt and its required codebase modifications.

## Ambiguity & Alignment Interrogation (Pre-Spec Gate)

Before formulating architecture or hypotheses:
- **Blind Spot & Trade-off Check**: Assess if requirements contain ambiguous business/quant trade-offs, unspecified boundary conditions, or open design choices.
- **Concise Interrogation (Grill-Me)**: If ambiguity exists, present 1–3 high-impact questions directly to the user before proceeding. Always list the recommended choice as option 1 `(Recommended)`.
- **Pass Through**: If requirements are already fully clear and unambiguous, skip questioning and proceed immediately to Memory Check.

## Memory & Anti-Pattern Check (Mandatory Pre-Spec Phase)

Before formulating architecture or hypotheses:
1. **Anti-Pattern Check**: Read `docs/decisions/anti_patterns.json`. Filter by task domain (`signal`, `risk`, `execution`). Instantly DISCARD any hypothesis, parameter range, or logic that previously failed.
2. **Context Lookup**: Inspect `docs/decisions/task_index.json` to review recent ADR resolutions for the relevant domain without reading monolithic logs.

## Empirical Sandbox Protocol (Mandatory Before Spec Freeze)


Do NOT rely solely on theoretical hypothesis or unverified context assumptions.
1. **Scratch Experimentation**: For any non-trivial algorithm, signal, risk model, or performance-critical logic, write a temporary Python script in `<appDataDir>/brain/<conversation-id>/scratch/test_<topic>.py`.
2. **Empirical Execution**: Execute the script via `uv run python <scratch_script>` to collect actual data metrics, execution logs, or runtime behavior.
3. **Evidence Requirement**: Record the verified console output metric in Section 2 (Evidence & Alternatives) of the spec artifact. Unverified hypotheses MUST NOT be frozen.

## Symbol Audit & Contract Standard

- **Symbol Audit & Wiring Inspection**: Use `rg` and `view_file` to classify affected symbols (`reuse`, `extend`, `new`, `retire`) and inspect exact file paths, signatures, line numbers, and call sites before freezing the wiring specification.
- **Contract Deliverables**:
  - `docs/specs/<feature>.md`
  - `docs/specs/<feature>_contract.json`

### `contract.json` Executable Schema
Every contract item MUST declare:
- `symbol`, `file`, `signature`, `error_policy`, `side_effects`, `semantic_rules`
- `python_assertion`: A single executable Python expression (e.g. `assert calc_fee(100.0) == 0.05`) for immediate deterministic testing.
- `fixture_reference` & `expected_property`
- `scenario_id`, `scope`, `test_name`, `target_test_file`
- `wiring` (`target`, `anchor`, `callee`, `invocation_expression`)

## Blueprint Artifact Sections

1. Goal & Selected Architecture
2. Evidence & Alternatives (Contains Empirical Sandbox execution logs)
3. Rules, Limits, Resilience, & Resource Budget
4. Integration/Wiring Plan
5. Contract Changes & Executable Assertions
6. TDD Scenario Matrix & Minimal Fixtures
7. Implementation Manifest

## Output Format

Do NOT repeat full blueprint, code, or logs in the response. Offload details into artifact files and return ONLY this compact card:

### 📌 [SPEC] <Feature / Task Title>

- **Objective**: <1-line core goal>
- **Diagnosis**: <1-line phenomenon & root cause summary>
  > <Optional 1-line key formula or condition>
- **Action**: <1-line planned changes summary>
📄 **Artifacts**: [<spec.md>](file:///docs/specs/<feature>.md) | [<contract.json>](file:///docs/specs/<feature>_contract.json)
