---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec

Use for architectural, trading, allocation, risk, data-contract, or multi-file changes. The Spec model
owns the decision; the Implement model should be able to execute the result without redesigning it.

## Workflow

1. Read `docs/decisions/decisions.md`, applicable `.agents/rules/`, and the relevant source/tests.
2. Choose a tier: Tier 1 skips this skill; Tier 2 produces a short spec and contract; Tier 3 adds an
   interview, experiments, and full integration design.
3. For Tier 3, ask only questions whose answers materially change the design (at most three), then wait
   when the work cannot proceed safely without them.
4. Inspect existing symbols with `rg`. Decide `reuse`, `extend`, `new`, or `retire` for each affected
   symbol before choosing an API.
5. Where the change affects signal, portfolio, risk, performance, or runtime data, compare at least two
   plausible approaches on the available data. Record the metric that selected the approach.
6. Before freezing the design, state the objective/acceptance measure, key assumptions, failure modes,
   and evidence that would falsify the selected approach.
7. Write `docs/specs/<feature>.md` and `docs/specs/<feature>_contract.json`.

## Design standard

- State one selected design; keep rejected alternatives brief.
- Preserve causal timestamps, cost/funding accounting, numerical stability, and fail-closed behavior.
- Do not change thresholds or public contracts merely to make a result pass.
- Keep the main spec under 500 lines. Move domain detail to a directly linked reference only when it is
  reused or too large for the main blueprint.

## Contract standard

Every contract item declares:

- exact symbol, file, signature, error policy, side effects, and semantic rules;
- literal executable assertions when an input/output can be computed without a fixture;
- a fixture reference and concrete expected property when execution requires real objects or data;
- scenario id, scope, exact test name, and exact target test file;
- wiring target, anchor, callee, and invocation expression for production integration.

Do not make prose fixture labels look like callable parameters. Freeze scenario names and file paths in the
contract; an implementation rename is a handoff conflict.

## Blueprint sections

Use only the sections needed by the tier:

1. Goal and selected architecture
2. Evidence and alternatives (required when experimentation is triggered)
3. Rules, limits, resilience, and resource budget
4. Integration/wiring plan
5. Contract changes
6. TDD scenario matrix and minimal fixtures
7. Implementation manifest

The manifest lists source ownership, test ownership, wiring edges, reuse/retire decisions, and explicit
non-goals. It is the Implement model's checklist, not another explanation of the design.

The TDD matrix should cover, as applicable, a happy path, boundary/error behavior, and a real caller
integration; add a performance or causality case when the change affects either.

## Handoff

After creating the artifacts, report only a short Korean summary: goal, flow, selected decision, and
`Proceed` as the next action. Do not repeat the full blueprint in the response.

## Output

Keep the first response within eight Korean lines:

`🎯 핵심 목표` · `🔄 로직 흐름` · `⚖️ 설계 핵심 결정`
