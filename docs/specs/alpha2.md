---
title: Alpha gate taxonomy and cost-wall responsibility repair
domain: futures-alpha
type: refactor
status: proposed
priority: critical
ai_read_policy: always
created: 2026-05-31
references:
  - docs/specs/alpha0.md
  - docs/specs/alpha1.md
  - docs/results/re-alpha.md
target_phase: alpha2
---

# Alpha Gate Taxonomy and Cost-Wall Responsibility Repair

## 0. Current Finding

Latest `re-alpha.md` state:

- `ALPHA_PASS=FALSE`.
- Validation-only rank policy is operating honestly.
- Every fold returned `validation_net_lcb_bps <= 0`, so `apply_rank_selection_policy()` emitted no trade.
- The alpha mode then failed with `alpha_panel empty`.
- The visible gate reasons included:
  - `alpha_p95_below_cost_wall`
  - `tradable_long_nz_below_threshold`
  - `tradable_short_nz_below_threshold`

This is no longer the original alpha0 mechanical bug class. The current blocker is a mixed responsibility problem:

1. `rank_sized` alpha emits portfolio selection weights, not gross EV bps.
2. `alpha_gate_diagnostics()` still treats the emitted matrix as return magnitude and compares `active_alpha_p95_bps` with `friction_bps + hurdle_bps`.
3. The 24bps wall is economically valid, but it must be applied to a realized or expected portfolio edge, not to a rank weight.
4. A validation-failed no-trade policy should produce a diagnostic alpha evaluation report, not an empty panel that prevents evaluation.

## 1. Cold Evaluation Criteria Audit

### 1.1 What must remain strict

Keep these criteria strict. They are useful and defensible:

- Time isolation: calibration must use validation folds only, never OOS/test realized returns.
- Rank skill: `rank_ic_lcb >= breakeven_ic_eff`.
- Statistical significance: Newey-West t-stat threshold must remain explicit.
- Robustness: DSR / deflated Sharpe must remain a separate robustness criterion.
- Post-selection economics: selected basket net LCB after execution cost must be positive before claiming economic alpha.
- Bear/regime diagnostics must remain visible.

### 1.2 What is currently wrong

The current `24bps` cost wall is being checked in two different semantic contexts:

- Correct context: basket/economic evaluation, where realized gross spread minus execution cost is measured.
- Incorrect context: alpha admission, when `alpha_long_2d` / `alpha_short_2d` are rank weights from `apply_rank_selection_policy()`.

Therefore:

- `basket_net_bps_lcb_24bps > 0` is a valid economic criterion.
- `alpha_active_p95_bps >= 24bps` is valid only for EV-return-unit alpha.
- `alpha_active_p95_bps >= 24bps` is not valid for rank-weight alpha.

The next phase must not relax the economic requirement. It must move the requirement to the correct layer.

## 2. Target Files

- `src/domain/futures/strategy/diagnostics.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/domain/futures/strategy/rank_selection.py`
- `src/domain/futures/forecast/compose.py`
- `src/execution/opt_main_futures.py`
- `src/domain/futures/strategy/config.py`
- `tests/unit/domain/futures/strategy/test_ml_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/domain/futures/forecast/test_compose.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`
- `docs/results/re-alpha.md`

## 3. Contracts

### 3.1 Gate taxonomy

Add explicit gate layers in `src/domain/futures/strategy/diagnostics.py`:

```python
from typing import Literal, TypedDict

AlphaOutputUnit = Literal["return_fraction", "rank_weight"]
AlphaGateLayer = Literal[
    "mechanical_integrity",
    "rank_skill",
    "policy_economics",
    "execution_realism",
    "statistical_robustness",
]


class AlphaGateReason(TypedDict):
    reason: str
    layer: AlphaGateLayer
    metric: str
    observed: float
    threshold: float
    unit: str
```

Do not replace the current `alpha_gate_fail_reasons: list[str]` yet. Add structured reasons in parallel:

```python
"alpha_gate_reason_details": list[AlphaGateReason]
```

### 3.2 Alpha gate diagnostics

Replace the existing signature with a backward-compatible extension:

```python
def alpha_gate_diagnostics(
    *,
    alpha_p95_bps: float,
    friction_bps: float,
    hurdle_bps: float,
    long_nz: float,
    short_nz: float,
    xs_long_preservation_ratio: float,
    xs_short_preservation_ratio: float,
    min_long_nz: float,
    min_short_nz: float,
    min_xs_preservation: float,
    cost_wall_tolerance_bps: float = 0.0,
    active_alpha_p95_bps: float | None = None,
    tradable_long_nz: float = 0.0,
    tradable_short_nz: float = 0.0,
    min_tradable_long_nz: float = 0.0,
    min_tradable_short_nz: float = 0.0,
    alpha_output_unit: AlphaOutputUnit = "return_fraction",
    require_alpha_cost_wall: bool = True,
) -> dict[str, object]:
    """Evaluate alpha admission diagnostics without mixing rank weights and return bps."""
```

Rules:

- If `alpha_output_unit == "return_fraction"` and `require_alpha_cost_wall is True`, keep the current cost-wall check.
- If `alpha_output_unit == "rank_weight"`, skip `alpha_p95_below_cost_wall`.
- If `alpha_output_unit == "rank_weight"`, also skip `tradable_*_nz_below_threshold`, because `alpha >= cost_floor` is unit-invalid.
- Always report:
  - `alpha_gate_floor_bps`
  - `alpha_gate_metric_bps`
  - `alpha_gate_metric_source`
  - `alpha_output_unit`
  - `alpha_cost_wall_required`
  - `alpha_gate_reason_details`

### 3.3 Rank policy failure report

Add to `src/domain/futures/strategy/rank_selection.py`:

```python
def policy_is_no_trade(policy: RankSelectionPolicy) -> bool:
    """Return True when validation failed and policy intentionally emits no trade."""
```

Implementation:

```python
return policy.validation_net_lcb_bps <= 0.0 or policy.n_obs <= 0
```

### 3.4 Alpha evaluation report extension

Extend `AlphaEvaluationReport`:

```python
policy_validation_net_lcb_bps: float = float("nan")
policy_validation_gross_bps: float = float("nan")
policy_validation_ir_t: float = float("nan")
policy_validation_monotonicity: float = float("nan")
policy_no_trade: bool = False
evaluation_layer_failures: dict[str, list[str]] = field(default_factory=dict)
```

Rules:

- `evaluate_alpha()` remains the post-facto evaluator.
- It must not decide whether rank-weight alpha passes a raw bps p95 wall.
- It may fail `policy_economics` if validation policy metadata says no-trade.

### 3.5 Config

Add to `StrategyMLConfig`:

```python
alpha_output_unit: Literal["return_fraction", "rank_weight"] = "rank_weight"
require_alpha_cost_wall: bool = False
require_rank_policy_positive_lcb_for_emit: bool = True
```

Validation:

- `alpha_output_unit` must be one of the literals.
- `require_alpha_cost_wall=True` is allowed only when `alpha_output_unit == "return_fraction"`.

## 4. Step-by-Step Logic

### Step A. Stop unit-invalid alpha gate failures

File: `src/domain/futures/strategy/diagnostics.py`

1. Add `AlphaOutputUnit`, `AlphaGateLayer`, `AlphaGateReason`.
2. Extend `alpha_gate_diagnostics()`.
3. Keep string fail reasons for compatibility.
4. Add structured reason details.
5. Do not append:
   - `alpha_p95_below_cost_wall`
   - `tradable_long_nz_below_threshold`
   - `tradable_short_nz_below_threshold`
   when `alpha_output_unit == "rank_weight"`.

### Step B. Pass the output unit from builder

File: `src/domain/futures/strategy/ml_builder.py`

1. Determine output unit:
   - `rank_sized` + `rank_policy_enabled=True` => `"rank_weight"`
   - EV emission path => `"return_fraction"`
2. Call `alpha_gate_diagnostics(..., alpha_output_unit=..., require_alpha_cost_wall=...)`.
3. Do not raise `strategy ml alpha gate failed` solely because a validation policy produced no-trade.
4. If all fold policies are no-trade, still build a panel with rank score columns and zero alpha weights.
5. Attach:
   - `panel.attrs["rank_policy_no_trade"]`
   - `panel.attrs["rank_policy_failure_reason"] = "validation_net_lcb_non_positive"`

### Step C. Make alpha mode evaluate no-trade panels

File: `src/execution/opt_main_futures.py`

1. In `--mode alpha`, do not replace a gate-failed run with empty `ml_out` if the failure is diagnostic only.
2. If the panel has zero alpha but rank score columns exist:
   - still compute dense rank IC using `rank_score_long/short`.
   - report `policy_no_trade=True`.
   - skip basket execution pass claims.
3. `ALPHA_PASS` must remain false when policy is no-trade.
4. The final reason must be `policy_economics.validation_net_lcb_non_positive`, not `alpha_panel empty`.

### Step D. Keep the 24bps wall, but in economic gates

Files:

- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/domain/futures/strategy/diagnostics.py`
- `src/execution/opt_main_futures.py`

Rules:

1. `top_bottom_spread_bps(..., cost_bps=24.0)` remains a valid diagnostic.
2. `basket_net_bps_lcb_24bps <= 0` remains a hard economic failure.
3. The log label must say `policy_economics`, not alpha magnitude.
4. The 24bps source should come from `round_trip_cost_bps()` plus explicit hurdle if intended. Do not leave unexplained magic `24.0` in new code.

### Step E. Update compose semantics

File: `src/domain/futures/forecast/compose.py`

Current `rank_cs_neutral` policy path sets:

```python
mu_long = np.where(policy_long > 0.0, policy_long, -np.inf)
mu_short = np.where(policy_short > 0.0, policy_short, -np.inf)
```

This is correct only if downstream treats `mu_*` as rank weights. Add explicit comments and diagnostics:

- `rank_cs_neutral` returns selection/weight surfaces, not EV-return surfaces.
- Cost-adjusted expected return must be evaluated by realized basket spread or a separate EV model.

Do not subtract `cost_frac` from rank weights.

## 5. Surgical Plan

### `src/domain/futures/strategy/diagnostics.py`

[ACTION: REPLACE]

- Extend `alpha_gate_diagnostics()` as specified in section 3.2.
- Add structured reason details.
- Skip unit-invalid cost-wall checks for `rank_weight`.

### `src/domain/futures/strategy/config.py`

[ACTION: ADD]

- Add config fields from section 3.5.
- Add validation checks.

### `src/domain/futures/strategy/rank_selection.py`

[ACTION: ADD]

- Add `policy_is_no_trade(policy: RankSelectionPolicy) -> bool`.

### `src/domain/futures/strategy/ml_builder.py`

[ACTION: REPLACE]

- Pass `alpha_output_unit` and `require_alpha_cost_wall` to `alpha_gate_diagnostics()`.
- Attach no-trade policy metadata.
- Avoid turning validation no-trade into `alpha_panel empty` in alpha diagnostic mode.

### `src/domain/futures/strategy/alpha_evaluation.py`

[ACTION: REPLACE]

- Extend `AlphaEvaluationReport`.
- Populate policy metadata from caller-provided attrs or optional params if added.
- Keep cost-aware basket criteria separate from alpha magnitude criteria.

### `src/execution/opt_main_futures.py`

[ACTION: REPLACE]

- Report dense rank IC even when emitted alpha is all zero.
- Summarize failures by layer:
  - `rank_skill`
  - `policy_economics`
  - `execution_realism`
  - `statistical_robustness`
  - `mechanical_integrity`
- Replace `alpha_panel empty` for policy no-trade with a structured failure summary.

### Tests

[ACTION: ADD/REPLACE]

- `test_alpha_gate_diagnostics_skips_cost_wall_for_rank_weight`
- `test_alpha_gate_diagnostics_keeps_cost_wall_for_return_fraction`
- `test_rank_policy_no_trade_panel_preserves_rank_scores`
- `test_alpha_mode_reports_policy_no_trade_not_empty_panel`
- `test_compose_rank_cs_neutral_does_not_subtract_cost_from_rank_weights`

## 6. Verification

### L1

```bash
uv run ruff check --fix \
  src/domain/futures/strategy/diagnostics.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/strategy/alpha_evaluation.py \
  src/domain/futures/forecast/compose.py \
  src/execution/opt_main_futures.py \
  tests/unit/domain/futures/strategy/test_ml_diagnostics.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/forecast/test_compose.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py
```

```bash
uv run mypy \
  src/domain/futures/strategy/diagnostics.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/strategy/alpha_evaluation.py \
  src/domain/futures/forecast/compose.py \
  src/execution/opt_main_futures.py
```

### L2

```bash
uv run pytest \
  tests/unit/domain/futures/strategy/test_ml_diagnostics.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/forecast/test_compose.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py \
  --tb=short
```

### Smoke

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python \
  src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h \
  --reference-date 2026-05-01
```

Expected minimum:

```text
[RANK-POLICY] ... val_lcb=...
[SCORE-IC] dense_ranker ...
[ALPHA-GATE] alpha_output_unit=rank_weight alpha_cost_wall_required=False
[ALPHA-POLICY] policy_no_trade=True reason=validation_net_lcb_non_positive
ALPHA_PASS: FALSE
```

The smoke must not end with:

```text
alpha_panel empty
```

## 7. Acceptance Checklist

```text
[ ] 24bps cost wall is retained in basket/economic evaluation.
[ ] 24bps cost wall is not applied to rank-weight alpha magnitude.
[ ] No-trade validation policy produces a non-empty diagnostic panel when rank scores exist.
[ ] ALPHA_PASS remains false for no-trade policy.
[ ] Failure reason identifies policy economics, not mechanical panel emptiness.
[ ] Logs separate rank skill, policy economics, execution realism, and robustness.
[ ] docs/results/re-alpha.md records the new alpha2 smoke result.
```

## 8. Next Phase After Alpha2

Only after alpha2 makes the evaluation readable:

1. If dense rank IC is positive but policy LCB stays negative, improve model/labels/features.
2. If rank IC is weak or unstable, stop portfolio work and return to alpha feature/label research.
3. If rank skill is good but post-selection economics fail, optimize rank policy, holding horizon, and turnover/cost model.
4. If economics pass in validation but fail OOS, add stronger purging/embargo and regime stability checks.
