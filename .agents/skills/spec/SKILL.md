---
name: spec
description: Produce a machine-readable, zero-invention implementation blueprint (Markdown Spec) from probed architecture rationale.
---

# Spec Protocol

Produce an unambiguous implementation blueprint (`docs/specs/<feature>_spec.md`) optimized for mechanical downstream execution.

## Allocation

The design rationale and empirical proof are resolved in `/probe`.
Your output in `/spec` focuses strictly on:
1. **Target Interfaces & Invariants**: Exact signatures, finalized production docstrings (explaining Why and constraints), and bulleted core invariants. Do not include temporary pseudo-code or recipes in docstrings.
2. **Wiring Points**: Caller anchor and invocation snippet.
3. **Invariant Verification Scenarios**: Concrete scenarios (Given / When / Invariant) validating contracts without freezing internal test syntax prematurely.

## Directives

1. **Context Alignment**:
   - Infer feature context, invariants, and decisions directly from the preceding probe step.
   - Inspect target files and immediate callers for exact imports and AST anchors without scanning unrelated paths.

2. **Blueprint Structure (`docs/specs/<feature>_spec.md`)**:
   Pure Markdown specification with three sections:

   ### A. Target Blueprint (`## Target: <relative_path>`)
   - Function/class signature with finalized production docstring (domain context, Args, Returns, Raises).
   - Core Invariants: bulleted preconditions, calculation sequence, edge case handling, and postconditions.

   ### B. Wiring Blueprint (`## Wiring: <caller_file>`)
   - Anchor point (`- Anchor: <symbol_or_line>`) and invocation snippet.

   ### C. Invariant Scenarios (`## Invariant Scenarios: <target_test_file>`)
   - Bulleted test scenarios: Given / When / Invariant (contract assertion, boundary, or fail-closed expectation).

3. **Validation**:
   Validate blueprint paths and caller anchors:
   ```bash
   uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_spec.md --pre-impl
   ```

## Chat Output Format

Keep chat output ultra-compact:

### 📐 [SPEC] <기능명>
> 📄 **청사진**: [`docs/specs/<feature>_spec.md`](file:///docs/specs/<feature>_spec.md)  
> 🚦 **하네스 검증**: `lean_check --pre-impl` **PASS**

- 🎯 **작업 요약**: <구현할 핵심 기능 1줄 요약>
- 📦 **작업 규모**: <N>개 파일 대상 · <N>개 테스트 시나리오 명세

---
👉 **다음 단계**: `/implement docs/specs/<feature>_spec.md`
