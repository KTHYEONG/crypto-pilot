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

- **Symbol Audit**: Use `rg` and `view_file` to classify affected symbols: `reuse`, `extend`, `new`, or `retire`.
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

## Output

Do NOT repeat the full blueprint in the response. Return ONLY this 4-line summary card:

📐 `[SPEC CREATED] <Feature/Blueprint Name> (Tier <1/2/3>)`
• `Goal`: `<1-line core objective>`
• `Empirical Evidence`: `<1-line verified log/metric from scratch execution>`
• `Artifacts`: [spec.md](file:///docs/specs/<feature>.md) | [contract.json](file:///docs/specs/<feature>_contract.json)
• `Next`: Proceed to `/implement`


