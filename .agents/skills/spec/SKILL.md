---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios for TDD.
---

# Skill: Spec (Unified Architecture & TDD Blueprint)

## Purpose
Merge core architecture conceptualization (Why/What) and detailed TDD interfaces (How/API Contract) into a single blueprint document (`docs/specs/[feature].md`).
Leverage **high-reasoning models (Thinker)** for creative architectural design, while outputting absolute, deterministic specifications for **low-cost models (Doer)** to implement mechanically with zero token waste.

## Execution Rules

### 1. Pre-process & Dynamic Interview (/grill-me)
- **Decisions Context**: Load `docs/decisions/decisions.md` (align history).
- **Rule Constraints**: Read and strictly adhere to `performance.md` and `quant.md` (timezone isolation, look-ahead bias).
- **Determine Workflow Tier**:
  - **Tier 1 (Light)**: Minor refactor, simple fix. *Directly skip Spec and proceed to implement.*
  - **Tier 2 (Standard)**: Medium complexity. Generate lightweight Markdown spec + `[feature]_contract.json`.
  - **Tier 3 (Architectural)**: Complex module, trading logic. Requires full Markdown spec + `[feature]_contract.json`.
- **Dynamic Interview (/grill-me)**:
  - For **Tier 3** tasks, before creating the spec document, analyze the user's prompt for design ambiguities.
  - Output exactly **3 targeted questions** to clarify architecture, DB schema, API interfaces, or edge-case handling.
  - **STOP immediately** and wait for user answers. Do NOT write the spec until the user answers.

### 2. High-Reasoning Architectural Thought (High Autonomy & Multi-Hypothesis)
- **Multi-Hypothesis Data-Driven Experimentation**:
  - **Trigger**: Automatically activate when working on signal logic, portfolio allocation, risk sizing, performance optimization, or whenever DB schemas / API payloads have runtime uncertainty.
  - **Hypothesis Setup**: Formulate at least 2~3 alternative hypotheses (e.g., `[HYPOTHESIS-A]`, `[HYPOTHESIS-B]`).
  - **Execution & Data Harvesting**: Write focused verification scripts in `scratch/`. Reuse existing `src/` modules via fast `rg` lookups when obvious, or write self-contained helper functions directly in `scratch/` when search overhead is high, ensuring zero token waste and unconstrained hypothesis testing.
  - **Quantitative Evaluation**: Evaluate hypotheses using concrete metrics (e.g., Sharpe Ratio, Max Drawdown, Execution Latency, Win Rate, or Memory Usage).
  - **Evidence-Based Selection**: Select the winning hypothesis based strictly on empirical evidence and document the benchmark comparison in the spec.
- **Alternatives & Trade-offs**: Contrast multiple design options using empirical benchmark data. Justify why the winning hypothesis was selected.
- **Constraints & Boundaries**: Identify edge cases, performance bottlenecks, and algorithmic limitations, tagging each with a unique label (`[LIMIT-01]`, etc.).
- **Quant & System Resilience**: Explicitly plan for network timeouts, timezone normalization, and look-ahead bias prevention.

### 3. Low-Reasoning Implementation Specifications (Deterministic Constraints)
To ensure low-reasoning models can build and integrate the code mechanically:
- **Exact Contract changes**: Define class/function signatures with Python 3.11+ type hints, including 1-line `error_policy` (Raise vs Fallback) and explicit `side_effects` (state mutations, logging).
- **Mandatory Assertions**: Define exact input/output mathematical or structural mappings in `assertions` to prevent dummy/stub implementations. **`/check` actually executes these** whenever `input`/`output` are JSON primitives (numbers/bools/strings-without-spaces/arrays thereof) whose keys match the target function's real parameters — it imports the module and calls the function for real, not just checking the array is non-empty. Prefer a literal computable value ("output": 100.0) over a description ("output": "breakeven cost in bps") whenever the value is actually computable by hand; a description is silently skipped (still useful as an implementor hint, but it earns no automatic verification). Array-valued inputs may be written as real nested JSON arrays (`[[1.0, 2.0]]`) — they are auto-coerced to `NDArray[np.float64]` (or `np.bool_` if every leaf is a Python bool) before the call.
- **Wiring & Connection Plan**: Define the exact file path, class, method, local context anchor line, and mandatory `invocation_symbol` (the exact 1-line invocation statement to guarantee pipeline integration).
- **Minimal Mock / Fixture Hints (Token Efficient)**:
  - Do **NOT** generate full 100+ line copy-paste test files in the markdown spec (saves 40%+ tokens).
  - Provide only a **3~5 line fixture/mock snippet** if complex setup or API mocking is required.
- **Content-over-Shape Assertions for the Assert Column**: Every assert cell in the Scenario Matrix MUST name at least one concrete numerical or structural property the implementor can pin a value assertion on. Behavioral descriptions alone are prohibited as the sole assert. Rationale: the spec (high-reasoning model) bears the small token cost of writing concrete values so the implement (low-reasoning model) can assert without re-deriving the correct answer. A spec whose assert column cannot be mechanically translated into `assert actual == expected` by a low-cost model is under-specified.
- **TDD Scenario Matrix**:
  - Map each LIMIT and PERF tag directly to a concrete test scenario:
    - **Scenario 1 (Happy Path)**: Unit input/output.
    - **Scenario 2 (Edge Cases)**: `[LIMIT-xx]` boundary conditions.
    - **Scenario 3 (Error Handling)**: Expected Exceptions and Error logs.
    - **Scenario 4 (Integration Verification)**: Asserting the correct trigger inside the parent module without over-mocking. Must instantiate Real Objects (no top-level MagicMock masking) to verify caller-callee pipeline integration.
  - Every scenario entry in `_contract.json` MUST carry a `target_test_file` (exact path, resolved via the co-modification convention: `tests/<category>/<module_dir>/test_<module>.py`). This lets `/check`'s fix_hint point directly at the file to edit instead of forcing a re-read of the whole spec.

### 4. Machine-Readable Compliance Contract (`docs/specs/[feature]_contract.json`)
*(Mandatory for Tier 2 and Tier 3)*
Generate a semantic JSON contract alongside the spec markdown to provide strict, deterministic constraints for Doer implementation and L2 verification:
```json
{
  "contracts": [
    {
      "kind": "class|function",
      "name": "ExactName",
      "file_hint": "src/domain/x.py",
      "signatures": "def ExactName(param: type) -> return_type",
      "error_policy": "Raise ValueError on invalid param; Return None on timeout",
      "side_effects": ["Updates self.state cache"],
      "semantic_rules": [
        "Rule 1: business logic constraint (e.g. rounded to 8 decimals)",
        "Rule 2: timezone isolation or look-ahead check rules"
      ],
      "assertions": [
        {"input": {"param": "valid_value"}, "output": "expected_result"},
        {"input": {"param": "invalid_value"}, "exception": "ValueError"}
      ]
    }
  ],
  "scenarios": [
    {"id": 1, "scope": "unit", "name": "test_exact_name_happy_path", "target_test_file": "tests/unit/domain/x/test_x.py"},
    {"id": 2, "scope": "unit", "name": "test_exact_name_edge_case", "target_test_file": "tests/unit/domain/x/test_x.py"},
    {"id": 3, "scope": "unit", "name": "test_exact_name_error", "target_test_file": "tests/unit/domain/x/test_x.py"},
    {"id": 4, "scope": "integration", "name": "test_parent_module_wiring", "target_test_file": "tests/unit/domain/x/test_caller_module.py"}
  ],
  "wiring": [
    {
      "target_file": "src/path/to/caller_module.py",
      "anchor_symbol": "ExactAnchorSymbol",
      "import_symbol": "ExactName",
      "invocation_symbol": "self.x = ExactName(); self.x.run()",
      "invocation_regex": "self\\.[a_z0-9_]+\\s*=\\s*ExactName\\(.*\\)"
    }
  ]
}
```

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Alternatives & Trade-offs**: Brief comparison.
- **Mermaid Diagram**: Text-based sequence showing system integration context.

# 🧪 Multi-Hypothesis Empirical Benchmarks (If Triggered)
- **Target Metrics**: (e.g., Sharpe, MDD, Execution Time)
- **Dataset / Experiment Script**: `scratch/verify_[feature].py`

| Hypothesis | Approach / Algorithm | Metric 1 | Metric 2 | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `[HYPOTHESIS-A]` | Baseline approach | Value | Value | Baseline |
| `[HYPOTHESIS-B]` | Alternative 1 | Value | Value | Selected (Best Sharpe/Latency) |
| `[HYPOTHESIS-C]` | Alternative 2 | Value | Value | Rejected |

- **Selection Rationale**: Empirical justification for choosing the winning hypothesis.

# ⚡ Performance & Resource Budget
- Complexity, limits, concurrency.

# ⚙️ Logical Rules, State Machine & Resilience
- Logical rules, state transition tables, tagged constraints (`[LIMIT-01]`, etc.), and resilience/recovery flow.

# 🔌 Integration & Connection Plan
- Target Location, state impact, error behavior.

# ✍️ Contract Changes
- Exact imports, class definitions, function signatures, and return types.

# 🧪 TDD Test Scenario Matrix & Mocks
- Scenario 1-4 descriptions.
- **Minimal Mock / Fixture Snippets**: (3~5 lines only if complex setup required)
```

## First Response Protocol after Spec Creation (Token-Efficient)
Do NOT repeat the technical contents. Present a highly-condensed summary in **Korean** within 8 lines:
1. **🎯 핵심 목표**: [Business goal]
2. **🔄 로직 흐름**: [A -> B -> C flow]
3. **⚖️ 설계 핵심 결정**: [Reasoning for design]
4. **👉 후속 대기**: [Wait for user review. Type `Proceed` to start Implement pipeline]
