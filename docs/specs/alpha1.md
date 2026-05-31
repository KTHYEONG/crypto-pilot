---
title: Alpha rank-to-portfolio calibration and gate diagnostics repair
domain: futures-alpha
type: spec-lite
status: ready
priority: critical
ai_read_policy: when_related
last_verified: 2026-05-31
related_paths:
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/forecast/compose.py
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/config.py
dependencies:
  documents: [docs/specs/alpha0.md, docs/results/re-alpha.md]
change_triggers:
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/forecast/compose.py
  - src/execution/opt_main_futures.py
---

# Alpha rank-to-portfolio calibration and gate diagnostics repair (Phase 1)

## 0. Current State

`alpha0` (Phase 3) fixed the mechanical evaluation defects:

```text
[OOS-DIAG] rank_cols=96 finite_rows=1417 oos_idx=1417 common_idx=1417
[RANK-CONTRACT] c3_signed_nz=0.586 c3_signed_std=0.003136
[RANK-QUALITY L1] ic=0.0095 t=1.45 breadth=16.7
[RANK-IC C3] ic=0.0174 t=1.04 breadth=8.79
[L3-BASKET] ew_bps=-7.26 net_bps=-31.26 n=1405
[C3-EXEC] NET_IC=0.0024 T-STAT=0.14 BRDTH=12.00 BE_IC=0.0152
```

Failure is now a real alpha construction problem, not an OOS index or signed-rank contract bug.

Observed blockers:

- `signal_skill_passes=FAIL`: mean `resid_ic=0.0174` exceeds `be_eff=0.0136`, but LCB/t-stat fail.
- `signal_preserved_after_selection=FAIL`: post-selection `NET_IC / dense_RANK_IC = 0.14`.
- `basket_net_positive=FAIL`: equal-weight basket is negative after 24bps cost.
- `MONOTONICITY`: `top-bot=-4.1bps`, `mono_rho=-0.70`, while full C3 rank IC is positive.
- `SWEEP`: only 18h passes; the default 12h portfolio conversion fails.

Interpretation:

1. Dense rank has weak but non-zero information.
2. Current static selection (`top q long, bottom q short`, q=0.35, 12h) destroys most of that information.
3. Tail behavior is unstable: current test tail has inverted/negative monotonicity even though full-panel IC is positive.
4. Passing the gate by relaxing thresholds would invalidate `alpha0` gate honesty. The correct fix is validation-only calibration of the rank-to-portfolio policy.

## 1. Evaluation Criteria Audit

Do not relax these gates:

- `rank_ic_lcb >= breakeven_ic_eff`
- `ic_t_stat_nw >= 3.0`
- `basket_net_bps_lcb_24bps > 0`
- `deflated_sharpe >= 0.95`
- `clip_preservation_ratio >= 0.7`

Issues to repair in diagnostics only:

1. Scoreboard displays `t >= 2.0`, while `evaluate_alpha()` gates at `t >= 3.0`.
   - Fix log threshold to 3.0 or source it from a single constant.
2. `signal_below_effective_breakeven` is triggered by `rank_ic_lcb < breakeven_ic_eff`, not by mean `resid_ic`.
   - Keep the gate, but log `rank_ic_lcb` next to `resid_ic`.
   - Rename display text to `signal_lcb_below_effective_breakeven` if possible.
3. `deflated_sharpe_too_low` is categorized as `regime_stability`.
   - Move it to `statistical_robustness`.
4. `RANK-QUALITY L1` log still says `rank_score_long-short`.
   - After `alpha3`, the signal is `derive_signed_rank_signal(...)`; update text.
5. `ALPHA_PASS` should remain strict.
   - `ALPHA_PASS=false` after alpha1 is acceptable only if diagnostics clearly identify whether the blocker is validation-calibrated rank skill, post-selection economics, or robustness.

## 2. Target Files

- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/forecast/compose.py`
- `src/execution/opt_main_futures.py`
- `src/domain/futures/strategy/config.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/forecast/test_compose.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`
- `docs/results/re-alpha.md`

## 3. Contracts

### 3.1 RankSelectionPolicy

Add a small policy model in `src/domain/futures/strategy/alpha_evaluation.py` or a new local module `src/domain/futures/strategy/rank_selection.py`.

Preferred new module:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RankSelectionPolicy:
    """Validation-calibrated mapping from signed rank score to long/short baskets."""

    polarity: Literal[1, -1]
    quantile: float
    min_abs_z: float
    weighting: Literal["equal", "zscore", "tanh"]
    weight_k: float
    holding_bars: int
    validation_net_lcb_bps: float
    validation_gross_bps: float
    validation_ir_t: float
    validation_monotonicity: float
    n_obs: int
```

Rules:

- `polarity=1`: higher signed score is long, lower signed score is short.
- `polarity=-1`: lower signed score is long, higher signed score is short.
- `quantile` must satisfy `0 < q < 0.5`.
- `min_abs_z` filters weak cross-sectional scores after z-scoring.
- `holding_bars` must be chosen from configured candidates only.
- Policy calibration must use validation/in-sample fold data only. Never use OOS/test realized returns to choose policy.

### 3.2 Policy calibration

Add:

```python
def calibrate_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    realized_fwd_ret_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    quantiles: tuple[float, ...],
    min_abs_z_grid: tuple[float, ...],
    holding_bars: int,
    cost_bps: float,
    min_obs: int = 120,
    weight_k: float = 3.0,
) -> RankSelectionPolicy:
    """Select polarity/quantile/floor using validation-only post-cost spread LCB."""
```

Calibration logic:

1. Convert `signed_score_2d` to cross-sectional z-score per bar.
2. For every `polarity in (1, -1)`, `q in quantiles`, `min_abs_z in min_abs_z_grid`:
   - apply `score = polarity * z_score`.
   - long basket: top q where `score >= min_abs_z`.
   - short basket: bottom q where `score <= -min_abs_z`.
   - compute bar spread: `mean(realized[long]) - mean(realized[short])`.
   - compute `gross_bps`, `net_bps = gross_bps - cost_bps`, `ir_t`, `hit`, `n_obs`.
   - compute one-sided LCB: `(gross_bps - se_bps) - cost_bps`.
   - compute monotonicity across 5 score buckets.
3. Choose the candidate maximizing:

```text
objective = validation_net_lcb_bps + 0.25 * validation_gross_bps
```

4. Hard reject candidates when:
   - `n_obs < min_obs`
   - `validation_net_lcb_bps <= 0`
   - `validation_monotonicity <= 0`
5. If no candidate passes:
   - return a conservative default policy with `validation_net_lcb_bps <= 0`, and let gates fail honestly.

Anti-bias requirement:

- Use only validation fold labels/returns.
- No reference to test/OOS realized returns, `common_idx` smoke metrics, or `docs/results/re-alpha.md` values.

### 3.3 Policy application

Add:

```python
def apply_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Emit alpha_long/alpha_short from signed score using a fixed policy."""
```

Required output:

- `alpha_long_2d`, `alpha_short_2d` shape equals `signed_score_2d`.
- Non-selected cells are exactly `0.0`.
- Selected cells:
  - `equal`: `1.0`
  - `zscore`: positive absolute z-score
  - `tanh`: `tanh(weight_k * abs(z))`
- No realized returns are used.

### 3.4 Policy metadata

`ml_builder.py` must attach:

```python
panel.attrs["rank_selection_policy"] = {
    "polarity": policy.polarity,
    "quantile": policy.quantile,
    "min_abs_z": policy.min_abs_z,
    "weighting": policy.weighting,
    "weight_k": policy.weight_k,
    "holding_bars": policy.holding_bars,
    "validation_net_lcb_bps": policy.validation_net_lcb_bps,
    "validation_gross_bps": policy.validation_gross_bps,
    "validation_ir_t": policy.validation_ir_t,
    "validation_monotonicity": policy.validation_monotonicity,
    "n_obs": policy.n_obs,
}
```

For multi-fold WF:

- Store `panel.attrs["rank_selection_policy_by_fold"]` as a list of fold policies.
- Also store an aggregate policy summary using median/majority vote for diagnostics.

## 4. Step-by-Step Logic

### Step A. Fix diagnostics without changing gates

File: `src/execution/opt_main_futures.py`

- Replace scoreboard t-stat display threshold from `2.0` to `3.0`.
- Add `rank_ic_lcb` to the scoreboard line.
- Split blocker categories:

```python
blocker_categories = {
    "rank_skill": [],
    "post_selection": [],
    "cost_turnover": [],
    "statistical_robustness": [],
    "regime_stability": [],
}
```

Mapping:

```python
"signal_below_effective_breakeven" -> "rank_skill"
"signal_t_stat_too_low" -> "statistical_robustness"
"deflated_sharpe_too_low" -> "statistical_robustness"
"signal_lost_after_selection" -> "post_selection"
"portfolio_ic_below_raw_breakeven" -> "post_selection"
"basket_net_lcb_non_positive" -> "cost_turnover"
"basket_net_not_profitable" -> "cost_turnover"
```

### Step B. Create rank selection policy SSOT

File: `src/domain/futures/strategy/rank_selection.py`

- Add `RankSelectionPolicy`.
- Add `_cs_zscore_2d()` or reuse existing `_cs_zscore` if importing from `forecast.compose` does not create circular imports.
- Add `calibrate_rank_selection_policy()`.
- Add `apply_rank_selection_policy()`.
- Add `policy_to_dict()` and `policy_from_dict()` if needed by `compose.py`.

### Step C. Use policy in ML alpha emission

File: `src/domain/futures/strategy/ml_builder.py`

Current `_emit_rank_sized_alpha()` uses static:

```python
select_q = alpha_emit_select_q
wl = rank_score_long
ws = -rank_score_short
```

Replace only the rank-sized path:

1. After fold predictions, derive signed rank score using `derive_signed_rank_signal()`.
2. For each non-virtual fold, calibrate policy on validation predictions and validation realized target.
3. Apply the fold policy to that fold's test rows.
4. For virtual/live refit, use the aggregate policy from non-virtual folds. Do not calibrate on live/OOS rows.
5. Preserve existing `rank_score_long` and `rank_score_short` metadata.

If validation target arrays are not currently retained in `_FoldPredictResult`, extend it with:

```python
valid_long_index_map: tuple[tuple[int, int], ...] | np.ndarray
valid_short_index_map: tuple[tuple[int, int], ...] | np.ndarray
score_valid_long: np.ndarray
score_valid_short: np.ndarray
```

Use existing labels to build validation realized residual returns. Do not recompute from future test data.

### Step D. Use same policy in compose/evaluation

Files:

- `src/domain/futures/forecast/compose.py`
- `src/execution/opt_main_futures.py`

`compose_mu()` rank_cs_neutral branch must use `apply_rank_selection_policy()` when `AlphaForecast` carries policy metadata or when params provide policy values:

```python
policy = RankSelectionPolicy(...)
alpha_l_sel, alpha_s_sel = apply_rank_selection_policy(
    signed_score_2d=derive_signed_rank_signal(rank_l2d, rank_s2d),
    eligible_2d=np.isfinite(rank_l2d) | np.isfinite(rank_s2d),
    policy=policy,
)
```

`opt_main_futures.py` scoreboard selection must use the same policy as the alpha panel when available. This prevents evaluation from using static q=0.35 while builder/live uses calibrated q/polarity.

Fallback:

- If no policy metadata exists, preserve current static q behavior for backward compatibility.

### Step E. Optional horizon calibration

Use this only after Step B-D are stable.

- Add config:

```python
rank_policy_holding_candidates: tuple[int, ...] = (6, 12, 18)
rank_policy_min_validation_obs: int = 120
rank_policy_min_abs_z_grid: tuple[float, ...] = (0.0, 0.25, 0.5)
```

- If the model label horizon remains 12, do not silently trade 18 as primary unless validation selected `holding_bars=18`.
- If 18h is selected, the smoke report must explicitly show:

```text
[RANK-POLICY] holding_bars=18 source=validation
```

This is allowed because the choice is validation-only, not OOS-smoke driven.

## 5. Surgical Plan

### `src/domain/futures/strategy/rank_selection.py`

[ACTION: ADD]

Implement:

- `RankSelectionPolicy`
- `calibrate_rank_selection_policy`
- `apply_rank_selection_policy`
- small private helpers for z-score, spread, monotonicity

### `src/domain/futures/strategy/config.py`

[ACTION: ADD]

Add fields:

```python
rank_policy_enabled: bool = True
rank_policy_quantiles: tuple[float, ...] = (0.20, 0.25, 0.35)
rank_policy_min_abs_z_grid: tuple[float, ...] = (0.0, 0.25, 0.50)
rank_policy_weighting: Literal["equal", "zscore", "tanh"] = "tanh"
rank_policy_holding_candidates: tuple[int, ...] = (12, 18)
rank_policy_min_validation_obs: int = 120
```

Validation:

- all quantiles must satisfy `0 < q < 0.5`
- all holding candidates must be positive
- `rank_policy_min_validation_obs >= 30`

### `src/domain/futures/strategy/ml_builder.py`

[ACTION: REPLACE]

Replace static rank emission with policy-calibrated emission when `rank_policy_enabled=True`.

[ACTION: ADD]

Log:

```text
[RANK-POLICY] fold=<id> polarity=<1|-1> q=<q> floor=<min_abs_z> hold=<bars> val_lcb=<bps> val_ir=<t> mono=<rho>
```

### `src/domain/futures/forecast/compose.py`

[ACTION: REPLACE]

Use `derive_signed_rank_signal()` + `apply_rank_selection_policy()` for `rank_cs_neutral` when policy params are available.

### `src/execution/opt_main_futures.py`

[ACTION: REPLACE]

- Use panel policy metadata in alpha smoke scoreboard.
- Fix t-stat display threshold.
- Add `rank_ic_lcb` log.
- Fix blocker categories.
- Update log text from `rank_score_long-short` to `signed rank score`.

### Tests

[ACTION: ADD]

Add unit tests:

- policy calibration rejects test-lookahead by accepting only provided arrays.
- same validation panel with inverted tails selects `polarity=-1`.
- positive monotonic validation panel selects `polarity=1`.
- policy application emits non-overlapping long/short masks.
- no policy fallback preserves current static behavior.
- blocker categorization maps DSR to `statistical_robustness`.

## 6. Verification

### L1

```bash
uv run ruff check --fix \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/forecast/compose.py \
  src/execution/opt_main_futures.py \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/domain/futures/forecast/test_compose.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py
```

```bash
uv run mypy \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/forecast/compose.py \
  src/execution/opt_main_futures.py
```

### L2

```bash
uv run pytest \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/domain/futures/forecast/test_compose.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py \
  --tb=short
```

### Smoke

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python \
  src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h \
  --reference-date 2026-05-01 2>&1 | \
  grep -E "RANK-POLICY|RANK-IC C3|MONOTONICITY|L3-BASKET|C3-EXEC|ALPHA_PASS"
```

Expected minimum:

```text
[RANK-POLICY] ... source=validation
[RANK-IC C3] breadth > 0
[MONOTONICITY] mono_rho >= 0.0
[L3-BASKET] n > 200
[C3-EXEC] NET_IC improves over alpha3 baseline 0.0024
```

Target pass-oriented thresholds:

```text
rank_ic_lcb >= breakeven_ic_eff
resid_t_stat_nw >= 3.0
clip_preservation_ratio >= 0.7
basket_net_bps > 0.0
basket_ir_t >= 2.0
portfolio_ic_above_breakeven = OK
```

If these do not pass, implementation is still acceptable only when the logs prove that policy was calibrated using validation data and the remaining blocker is genuine statistical weakness, not static selection mismatch.

## 7. Acceptance Checklist

```text
[ ] Gate thresholds are not relaxed.
[ ] Scoreboard threshold/log labels match actual gate logic.
[ ] DSR failure is categorized as statistical_robustness, not regime_stability.
[ ] Rank-to-portfolio policy is calibrated on validation folds only.
[ ] Builder, compose, and smoke evaluation share the same selection policy.
[ ] Current alpha3 failure mode `mono_rho=-0.70`, `basket_net=-31bps`, `presv=0.14` is improved or explicitly proven irreducible by validation metrics.
[ ] docs/results/re-alpha.md records alpha1 policy and post-smoke metrics.
```
