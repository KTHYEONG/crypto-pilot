# 🎯 Objective
L0 Alpha Foundry에 비용·IC·cross-sectional 검증을 강화한 V2 gate와 adaptive timeframe search contract를 추가해 L1로 전달되는 signal 후보의 경제성을 높인다.

# 📦 Context & Dependencies

## Target Imports

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    AlphaArchetype,
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    SymbolScope,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel
```

## Existing Anchors

| File | Anchor | Use |
|---|---|---|
| `src/domain/futures/alpha_foundry/contracts.py` | `CheapGateEvidence`, `AlphaFoundryEvidenceRow`, `AlphaFoundryRuntimeConfig` | add V2 dataclasses and config fields |
| `src/domain/futures/alpha_foundry/cheap_gate.py` | `evaluate_panel_cheap_gate()` | add V2 evaluator helpers without breaking existing API |
| `src/domain/futures/alpha_foundry/recipes.py` | `RECIPE_DEFINITIONS`, `FAMILY_ARCHETYPE`, `FAMILY_SIDE_RULE`, `FAMILY_EXIT_POLICY`, `FAMILY_MAX_TURNOVER` | add recipe catalog entries for new families |
| `src/domain/futures/signals/rules.py` | `ALL_SIGNAL_FAMILIES`, `build_rule_signal_panels()` | add signal families and panels |
| `src/domain/futures/strategy/rule_signals.py` | mirror of `signals/rules.py` | keep registry and signal logic synchronized |
| `src/domain/futures/optimization/metrics.py` | `_bars_per_year_for_tf()` | adaptive timeframe annualization |

## New Type Aliases

```python
AlphaEntryMode: TypeAlias = Literal["sparse", "continuous", "cross_sectional_rank"]
AlphaSearchStatus: TypeAlias = Literal["pending", "screened", "gated", "l1_queued", "retired"]
AlphaTimeframe: TypeAlias = Literal["30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d"]
```

# ✍️ Contract Changes

## `contracts.py`

### `AlphaSignalBlueprint`

```python
@dataclass(slots=True, frozen=True)
class AlphaSignalBlueprint:
    family: str
    variant: str
    archetype: AlphaArchetype
    timeframe: str
    required_fields: tuple[str, ...]
    causal_lag_bars: int
    lookback_bars: tuple[int, ...]
    holding_bars: int
    max_turnover_per_year: float
    entry_mode: AlphaEntryMode
    side_rule_id: str
    exit_policy_id: str

    def __post_init__(self) -> None: ...
```

Validation:
- `family`, `variant`, `timeframe`, `side_rule_id`, `exit_policy_id` must be non-empty.
- `causal_lag_bars >= 1`.
- `holding_bars >= 1`.
- every `lookback_bars` item must be `>= 1`.
- `max_turnover_per_year >= 0.0`.
- `continuous` mode is allowed only when `max_turnover_per_year <= 365.0`.

### `L0SearchCell`

```python
@dataclass(slots=True, frozen=True)
class L0SearchCell:
    blueprint_id: str
    family: str
    variant: str
    timeframe: str
    tf_minutes: int
    symbol_scope: SymbolScope
    cost_floor_bps: float
    expected_event_rate: float
    family_prior_score: float
    status: AlphaSearchStatus = "pending"

    def __post_init__(self) -> None: ...
```

Validation:
- `blueprint_id`, `family`, `variant`, `timeframe` must be non-empty.
- `tf_minutes > 0`.
- `cost_floor_bps >= 0.0`.
- `expected_event_rate >= 0.0`.
- `family_prior_score` must be finite.

### `AlphaGateEvidenceV2`

```python
@dataclass(slots=True, frozen=True)
class AlphaGateEvidenceV2:
    recipe_id: str
    timeframe: str
    symbol_scope: SymbolScope
    n_events: int
    effective_n: float
    mean_gross_bps: float
    mean_cost_bps: float
    mean_net_bps: float
    gross_lcb_bps: float
    net_lcb_bps: float
    nw_tstat: float
    rank_ic: float
    rank_ic_tstat: float
    cost_drag_ratio: float
    turnover_per_year: float
    event_hit_rate: float
    payoff_skew: float
    regime_edge_bps: Mapping[str, float]
    xs_spread_lcb_bps: float | None
    liquidity_cost_stress_bps: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    gate_passed: bool
    reject_reasons: tuple[str, ...]
    soft_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None: ...
```

Validation:
- `n_events >= 0`.
- `effective_n >= 0.0`.
- all numeric fields except `xs_spread_lcb_bps` must be finite.
- `cost_drag_ratio >= 0.0`.
- `0.0 <= event_hit_rate <= 1.0`.
- `xs_spread_lcb_bps` may be `None`; if not `None`, it must be finite.

### `CheapGateConfig` additions

```python
min_candidate_rank_ic_tstat: float = 2.0
min_xs_symbols_per_bar: int = 5
max_abs_btc_beta: float = 0.80
high_turnover_per_year: float = 180.0
liquidity_cost_stress_mult: float = 1.0
enable_v2_gate_metrics: bool = False
```

Validation:
- `min_candidate_rank_ic_tstat >= 0.0`.
- `min_xs_symbols_per_bar >= 2`.
- `max_abs_btc_beta >= 0.0`.
- `high_turnover_per_year >= 0.0`.
- `liquidity_cost_stress_mult >= 0.0`.

## `cheap_gate.py`

### Metric Helpers

```python
def compute_cost_drag_ratio_v2(*, mean_cost_bps: float, mean_gross_bps: float, eps: float = 1e-10) -> float: ...

def compute_rank_ic_with_tstat(
    *,
    fwd_ret_bps: NDArray[np.float64],
    score: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float]: ...

def compute_payoff_stats(values_bps: NDArray[np.float64]) -> tuple[float, float]: ...

def compute_xs_spread_lcb_bps(
    *,
    net_bps: NDArray[np.float64],
    score: NDArray[np.float64],
    event_mask: NDArray[np.bool_],
    min_symbols_per_bar: int,
    quantile: float = 0.20,
) -> float | None: ...

def compute_liquidity_cost_stress_bps(
    *,
    aligned: AlignedMarketData,
    event_mask: NDArray[np.bool_],
    stress_mult: float,
) -> float: ...
```

Rules:
- `compute_cost_drag_ratio_v2` uses event means, not totals.
- `compute_rank_ic_with_tstat` returns `(0.0, 0.0)` for fewer than 3 observations or constant inputs.
- `compute_payoff_stats` returns `(hit_rate, payoff_skew)` where `hit_rate = mean(values_bps > 0)` and `payoff_skew = mean(pos) / abs(mean(neg))`; no negative/zero denominator leak.
- `compute_xs_spread_lcb_bps` groups by bar, ranks scores cross-sectionally, computes top-minus-bottom net spread, and returns 5th percentile bootstrap/block LCB compatible value.
- `compute_liquidity_cost_stress_bps` uses `aligned.execution_cost_bps_2d` when available; otherwise returns `0.0`.

### V2 Evaluator

```python
def evaluate_panel_gate_v2(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    config: CheapGateConfig,
    bars_per_year: float,
) -> AlphaGateEvidenceV2: ...
```

Reject rules:
- `insufficient_events`: `n_events < min_events`.
- `insufficient_effective_n`: `effective_n < min_effective_n`.
- `non_positive_gross`: `mean_gross_bps <= 0.0`.
- `non_positive_lcb`: `net_lcb_bps <= min_lcb_net_bps`.
- `weak_tstat`: `abs(nw_tstat) < min_nw_tstat`.
- `excess_cost_drag`: `cost_drag_ratio > max_cost_drag_ratio`.
- `excess_turnover`: `turnover_per_year > min(max_turnover_per_year, recipe.max_turnover_per_year)`.
- `gross_lcb_below_cost`: `turnover_per_year >= high_turnover_per_year` and `gross_lcb_bps <= mean_cost_bps + liquidity_cost_stress_bps`.
- `xs_spread_fail`: `recipe.archetype == "cross_sectional"` and `xs_spread_lcb_bps is None or xs_spread_lcb_bps <= 0.0`.

Soft flags:
- `weak_rank_ic`: `abs(rank_ic) < 1 / sqrt(max(n_events - 3, 1))`.
- `weak_rank_ic_tstat`: `abs(rank_ic_tstat) < min_candidate_rank_ic_tstat`.
- `bootstrap_disagree`: bootstrap sign disagrees with net LCB sign.

### Compatibility Adapter

```python
def downgrade_gate_v2_to_cheap_evidence(evidence: AlphaGateEvidenceV2) -> CheapGateEvidence: ...
```

Purpose:
- Preserve existing downstream `run_alpha_foundry_l0_pipeline()` behavior while allowing new tests to assert V2 metrics.

## `search_space.py` new module

```python
DEFAULT_ALPHA_TIMEFRAME_GRID: tuple[str, ...] = ("30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d")

def timeframe_to_minutes(tf: str) -> int: ...

def resolve_alpha_timeframe_grid(
    *,
    enable_fast_timeframes: bool,
    include_daily: bool = True,
) -> tuple[str, ...]: ...

def make_alpha_blueprint_id(
    *,
    family: str,
    variant: str,
    timeframe: str,
    params: Mapping[str, float | int | str],
) -> str: ...

def build_l0_search_cells(
    *,
    blueprints: Sequence[AlphaSignalBlueprint],
    family_prior_scores: Mapping[str, float],
    cost_floor_bps_by_tf: Mapping[str, float],
) -> tuple[L0SearchCell, ...]: ...

def mark_retired_search_cells(
    *,
    cells: Sequence[L0SearchCell],
    failed_keys: set[tuple[str, str, str]],
) -> tuple[L0SearchCell, ...]: ...
```

Rules:
- `resolve_alpha_timeframe_grid(enable_fast_timeframes=False)` returns `("3h", "4h", "6h", "8h", "12h", "1d")`.
- `resolve_alpha_timeframe_grid(enable_fast_timeframes=True)` includes `30m`, `1h`, `2h`.
- `timeframe_to_minutes` supports only `m`, `h`, `d`; invalid suffix raises `ValueError`.
- `mark_retired_search_cells` uses key `(family, timeframe, variant)`.

## Recipe and Signal Family Additions

Add these family names to both `ALL_SIGNAL_FAMILIES` declarations and recipe maps:

```python
NEW_ALPHA_FAMILIES: tuple[str, ...] = (
    "sparse_breakout_retest_v2",
    "trend_pullback_quality_v2",
    "residual_momentum_xs",
    "funding_contra_carry_sparse",
    "oi_price_divergence_unwind",
    "taker_flow_exhaustion",
    "liquidity_vacuum_breakout",
    "volatility_contraction_expansion",
    "btc_regime_relative_strength",
    "mean_reversion_after_liquidation_proxy",
)
```

Minimum implementation for this spec:
- Register all 10 families in recipe metadata maps.
- Implement panels for the first 3 families only: `sparse_breakout_retest_v2`, `trend_pullback_quality_v2`, `residual_momentum_xs`.
- Leave the other 7 registered with recipe definitions only if no rule implementation exists yet; they must not emit panels until implemented.

# 🧪 TDD Test Scenario Matrix (CRITICAL)

## Scenario 1: Happy Path - Test Setup

| Test | Location | Assertion |
|---|---|---|
| `test_alpha_signal_blueprint_validates_successfully` | `tests/unit/domain/futures/alpha_foundry/test_alpha_search_contracts.py` | valid blueprint is frozen dataclass with exact fields |
| `test_l0_search_cell_builds_from_blueprint_grid` | same | `build_l0_search_cells()` creates deterministic ids and status `pending` |
| `test_timeframe_grid_includes_fast_tfs_when_enabled` | same | fast grid includes `30m`, `1h`, `2h`; slow grid excludes them |
| `test_gate_v2_passes_sparse_positive_cost_adjusted_panel` | `tests/unit/domain/futures/alpha_foundry/test_gate_v2.py` | positive sparse trend panel has `gate_passed=True`, positive `mean_net_bps`, finite `rank_ic_tstat` |
| `test_gate_v2_downgrade_preserves_existing_fields` | same | adapter maps V2 to `CheapGateEvidence` without unit mismatch |
| `test_residual_momentum_xs_panel_has_cross_sectional_archetype` | `tests/unit/domain/futures/alpha_foundry/test_alpha_family_registration.py` | family is registered and emits `cross_sectional` recipe metadata |

## Scenario 2: Edge Cases - Validation/Bounds

| Core Constraint | Test | Expected |
|---|---|---|
| Look-ahead prevention | `test_gate_v2_uses_causal_lag_for_forward_return` | changing future-only close before `t+lag` does not change event return |
| HTF bypass prohibition | `test_every_selected_tf_requires_v2_evidence` | selected panel without matching V2 evidence is excluded from L1 handoff |
| Unit consistency | `test_cost_drag_v2_uses_event_means_not_totals` | duplicating event count with same means does not change cost drag |
| Cost realism | `test_high_turnover_requires_gross_lcb_above_cost_stress` | high-turnover panel gets `gross_lcb_below_cost` |
| Multiple testing | `test_search_cells_can_be_retired_by_family_tf_variant` | repeated failed key changes status to `retired` only for matching cells |
| Sparse entry | `test_gate_v2_counts_only_flat_to_active_entries` | continuous same-side bars do not inflate `n_events` |
| Cross-sectional alpha | `test_xs_spread_requires_min_symbols_per_bar` | fewer than configured symbols returns `xs_spread_lcb_bps=None` and reject |
| Data availability | `test_blueprint_required_fields_fail_closed` | missing `oi` or `lsr` in required field check raises or rejects |
| 24/7 futures | `test_gate_v2_includes_funding_over_holding_period` | funding cost changes `mean_cost_bps` and `mean_net_bps` |
| Compute budget | `test_build_search_cells_does_not_materialize_feature_arrays` | search cell creation uses metadata only and creates no `[T,N]` arrays |
| Adaptive timeframe | `test_timeframe_to_minutes_supports_new_grid` | `30m`, `3h`, `1d` map to `30`, `180`, `1440` |
| Cross-sectional beta cap | `test_xs_blueprint_records_beta_cap_constraint` | `residual_momentum_xs` metadata includes `max_abs_btc_beta` |

## Scenario 3: Error Handling - Exceptions

| Test | Trigger | Expected |
|---|---|---|
| `test_alpha_signal_blueprint_rejects_empty_family` | `family=""` | `ValueError("family must not be empty")` |
| `test_alpha_signal_blueprint_rejects_invalid_lag` | `causal_lag_bars=0` | `ValueError("causal_lag_bars must be >= 1")` |
| `test_alpha_signal_blueprint_rejects_invalid_lookback` | `lookback_bars=(0,)` | `ValueError("lookback_bars must be >= 1")` |
| `test_l0_search_cell_rejects_invalid_tf_minutes` | `tf_minutes=0` | `ValueError("tf_minutes must be positive")` |
| `test_l0_search_cell_rejects_nonfinite_prior` | `family_prior_score=np.nan` | `ValueError("family_prior_score must be finite")` |
| `test_gate_v2_raises_on_invalid_shape` | panel arrays not `[T,N]` | existing `ValueError("shape")` |
| `test_gate_v2_rejects_invalid_bars_per_year` | `bars_per_year=0.0` | `ValueError("bars_per_year must be positive")` |
| `test_timeframe_to_minutes_rejects_bad_suffix` | `"4x"` | `ValueError("unsupported timeframe")` |
| `test_timeframe_to_minutes_rejects_zero_value` | `"0h"` | `ValueError("timeframe value must be positive")` |
| `test_gate_v2_handles_constant_score_ic` | constant score | `rank_ic=0.0`, `rank_ic_tstat=0.0`, no exception |

## Mock Boilerplate Snippet

```python
from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.contracts import AlphaRecipe
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData


def make_gate_v2_aligned(*, n_symbols: int = 6, funding: float = 0.0001) -> AlignedMarketData:
    datetimes = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-03-01T00:00:00"),
        np.timedelta64(4, "h"),
        dtype="datetime64[ns]",
    )
    t = int(datetimes.shape[0])
    symbols = tuple(f"SYM{i}USDT" for i in range(n_symbols))
    base = 100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))
    close = np.column_stack([base * (1.0 + i * 0.01) for i in range(n_symbols)])
    mask = np.ones_like(close, dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full_like(close, 1_000.0),
        funding_2d=np.full_like(close, funding),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
        oi_2d=np.full_like(close, 10_000.0),
        lsr_2d=np.full_like(close, 1.2),
        taker_buy_2d=np.full_like(close, 500.0),
        trades_2d=np.full_like(close, 100.0),
        execution_cost_bps_2d=np.full_like(close, 2.5),
    )


def make_sparse_panel(aligned: AlignedMarketData, *, recipe_id: str) -> CandidateSignalPanel:
    t, n = aligned.close_2d.shape
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    score = side.astype(np.float64)
    return CandidateSignalPanel(
        family="sparse_breakout_retest_v2",
        variant="bor_v2_20",
        params={"channel": 20, "retest": 3},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": recipe_id, "source": "catalog_exact"},
        archetype="trend",
    )


SAMPLE_V2_RECIPE = AlphaRecipe(
    recipe_id="sparse_breakout_retest_v2:bor_v2_20:4h",
    family="sparse_breakout_retest_v2",
    variant="bor_v2_20",
    timeframe="4h",
    archetype="trend",
    indicator_params={"channel": 20, "retest": 3},
    side_rule_id="breakout_retest_sparse",
    exit_policy_id="atr_trail_2",
    required_fields=("close", "high", "low", "volume"),
    causal_lag_bars=1,
    max_turnover_per_year=120.0,
)
```

# 🛠️ Algorithmic Plan

## Step 1: Contracts

Target:
- `src/domain/futures/alpha_foundry/contracts.py`

Flow:
1. Add `AlphaEntryMode`, `AlphaSearchStatus`, `AlphaTimeframe`.
2. Add `AlphaSignalBlueprint`, `L0SearchCell`, `AlphaGateEvidenceV2`.
3. Extend `CheapGateConfig` with V2 config fields and validation.
4. Keep existing `CheapGateEvidence` and `AlphaFoundryEvidenceRow` fields backward compatible.

## Step 2: Search Space

Target:
- new `src/domain/futures/alpha_foundry/search_space.py`

Flow:
1. Implement `timeframe_to_minutes()`.
2. Implement grid resolver with fast timeframe switch.
3. Implement deterministic `blueprint_id` hash.
4. Implement search cell creation from blueprints and prior scores.
5. Implement retirement status update by `(family, timeframe, variant)`.

## Step 3: V2 Gate Metrics

Target:
- `src/domain/futures/alpha_foundry/cheap_gate.py`

Flow:
1. Reuse existing `_validate_shape()`, `_sparse_entry_mask()`, `_compute_turnover_per_year()`, `_compute_block_means()`, `_bootstrap_block_ci()`.
2. Compute forward gross return with `close[t + holding_bars] / close[t + causal_lag_bars]`.
3. Compute event mean cost as stress round-trip plus funding over holding period plus optional liquidity stress.
4. Compute `gross_lcb_bps` and `net_lcb_bps` from block means.
5. Compute rank IC and Fisher-z t-stat.
6. Compute payoff stats and cross-sectional spread LCB.
7. Apply reject rules and soft flags exactly as listed above.
8. Add adapter to legacy `CheapGateEvidence`.

## Step 4: Family Registration

Target:
- `src/domain/futures/alpha_foundry/recipes.py`
- `src/domain/futures/signals/rules.py`
- `src/domain/futures/strategy/rule_signals.py`

Flow:
1. Add all 10 family ids to recipe maps.
2. Add `sparse_breakout_retest_v2`, `trend_pullback_quality_v2`, `residual_momentum_xs` to rule panel generation.
3. Keep rules causal: all rolling highs/lows/z-scores must use shifted or trailing-only windows.
4. Update `_resolve_panel_archetype()` trend/cross-sectional fallback if needed.

## Step 5: Tests and Checks

Target tests:
- `tests/unit/domain/futures/alpha_foundry/test_alpha_search_contracts.py`
- `tests/unit/domain/futures/alpha_foundry/test_gate_v2.py`
- `tests/unit/domain/futures/alpha_foundry/test_alpha_family_registration.py`

Commands:

```bash
uv run pytest tests/unit/domain/futures/alpha_foundry/test_alpha_search_contracts.py --tb=short
uv run pytest tests/unit/domain/futures/alpha_foundry/test_gate_v2.py --tb=short
uv run pytest tests/unit/domain/futures/alpha_foundry/test_alpha_family_registration.py --tb=short
uv run pytest tests/unit/domain/futures/alpha_foundry --tb=short
```
