# 🎯 Objective
BTC/ETH holdout long 고착을 줄이되 fit/cal 정상 수익을 보존하기 위해, 기존 adverse-only directional veto를 상태 기반 contextual directional veto로 확장한다.

# 📦 Context & Dependencies

## Target Files

- `src/domain/futures/strategy/tiered_workflow/awf_sim.py`
- `src/domain/futures/strategy/tiered_workflow/dataclasses.py`
- `src/domain/futures/strategy/tiered_workflow/pipeline.py`
- `src/domain/futures/strategy/tiered_logging.py`
- `tests/unit/domain/futures/strategy/tiered_workflow/test_directional_veto.py`
- `tests/unit/domain/futures/strategy/tiered_workflow/test_directional_veto_replay.py`
- `tests/unit/domain/futures/strategy/test_tiered_workflow.py`

## Exact Imports

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2Result,
    Layer3Result,
)
from src.domain.futures.strategy.walk_forward import WFFold
```

## Existing Contracts Reused

- `DirectionalVetoSnapshot`
- `DirectionalVetoSummary`
- `Layer2FoldAttribution.directional_veto_snapshots`
- `Layer2AllocationConfig.from_mapping(...)`
- `run_l2_awf(...)`
- `run_l3_holdout(...)`
- `run_directional_veto_economic_replay(...)`
- `_directional_veto_replay_adoption_verdict(...)`
- `_write_directional_veto_replay_csv(...)`

## Existing Anchors

- `awf_sim.py`: comment anchor `# [L2 Directional Veto]: adverse regime long veto`
- `pipeline.py`: `_directional_veto_replay_adoption_verdict`
- `pipeline.py`: `run_directional_veto_economic_replay`
- `tiered_logging.py`: `_format_directional_veto_line`

# ✍️ Contract Changes

## 1. `src/domain/futures/strategy/tiered_workflow/dataclasses.py`

### `Layer2AllocationConfig` Extend

Keep existing directional-veto fields and add:

```python
@dataclass(slots=True, frozen=True)
class Layer2AllocationConfig:
    ...
    l2_regime_directional_veto_mode: Literal["adverse_only", "contextual"] = "adverse_only"
    l2_regime_directional_veto_persistence_bars: int = 3
    l2_regime_directional_veto_loss_lookback_bars: int = 18
    l2_regime_directional_veto_loss_trigger_bps: float = 150.0
    l2_regime_directional_veto_cap_mu_bps: float = 0.0
    l2_regime_directional_veto_release_raw_mu_nonpos: bool = True
    l2_regime_directional_veto_release_regime_bull_bars: int = 2
    l2_regime_directional_veto_cooldown_bars: int = 3
    l2_regime_directional_veto_max_fit_net_value_loss: float = 0.0
    l2_regime_directional_veto_min_l3_total_return_delta: float = 0.02
    l2_regime_directional_veto_max_l2_cagr_delta_loss: float = 0.005
```

### Validation Rules in `from_mapping(...)`

Add exact validation:

```python
l2_regime_directional_veto_mode in {"adverse_only", "contextual"}
l2_regime_directional_veto_action in {"drop_long", "zero_mu", "cap_mu"}
l2_regime_directional_veto_persistence_bars >= 1
l2_regime_directional_veto_loss_lookback_bars >= 1
l2_regime_directional_veto_release_regime_bull_bars >= 1
l2_regime_directional_veto_cooldown_bars >= 0
l2_regime_directional_veto_loss_trigger_bps >= 0.0
l2_regime_directional_veto_cap_mu_bps >= 0.0
l2_regime_directional_veto_max_fit_net_value_loss >= 0.0
l2_regime_directional_veto_min_l3_total_return_delta >= 0.0
l2_regime_directional_veto_max_l2_cagr_delta_loss >= 0.0
```

### No New Result Types

Keep `Layer2Result.directional_veto_summary` and `Layer3Result.directional_veto_summary` as:

```python
directional_veto_summary: tuple[DirectionalVetoSummary, ...] = ()
```

No type rename.

## 2. `src/domain/futures/strategy/tiered_workflow/awf_sim.py`

### Extend `DirectionalVetoSnapshot`

Preserve current fields and add:

```python
@dataclass(slots=True, frozen=True)
class DirectionalVetoSnapshot:
    fold_idx: int
    t: int
    symbol: str
    regime_code: int
    raw_mu_before: float
    raw_mu_after: float
    counterfactual_weight: float
    weight_after: float
    fired: bool
    was_missing: bool
    bar_price_return_after: float
    counterfactual_long_return: float
    state_before: Literal["idle", "watch", "armed", "veto", "cooldown"] = "idle"
    state_after: Literal["idle", "watch", "armed", "veto", "cooldown"] = "idle"
    rolling_symbol_return: float = 0.0
    release_reason: str = ""
    actual_symbol_return: float = 0.0
```

### Extend `DirectionalVetoSummary`

Preserve current fields and add:

```python
@dataclass(slots=True, frozen=True)
class DirectionalVetoSummary:
    symbol: str
    n_obs: int
    n_missing: int
    n_adverse: int
    n_fired: int
    fire_rate: float
    adverse_fire_rate: float
    false_positive_rate: float
    opportunity_cost: float
    avoided_loss: float
    net_veto_value: float
    n_watch: int = 0
    mean_trigger_loss: float = 0.0
    mean_episode_bars: float = 0.0
```

### New Internal Dataclass

```python
@dataclass(slots=True)
class ContextualDirectionalVetoState:
    symbol: str
    state: Literal["idle", "watch", "armed", "veto", "cooldown"] = "idle"
    adverse_long_streak: int = 0
    bull_release_streak: int = 0
    cooldown_left: int = 0
    entry_t: int | None = None
    last_action: Literal["none", "cap_mu", "zero_mu", "drop_long"] = "none"
```

### New Helpers

```python
def _compute_contextual_directional_veto_signal(
    *,
    symbol: str,
    raw_mu: float,
    regime_code: int,
    rolling_symbol_return: float,
    state: ContextualDirectionalVetoState,
    config: Layer2AllocationConfig,
) -> tuple[
    ContextualDirectionalVetoState,
    bool,
    float,
    str,
    Literal["idle", "watch", "armed", "veto", "cooldown"],
    Literal["idle", "watch", "armed", "veto", "cooldown"],
]:
    ...


def _compute_symbol_rolling_return(
    *,
    close_2d: NDArray[np.float64],
    t: int,
    symbol_idx: int,
    lookback_bars: int,
) -> float:
    ...
```

### Update Existing Helper

```python
def summarize_directional_veto(
    fold_attributions: tuple[Layer2FoldAttribution, ...],
    *,
    symbols: tuple[str, ...],
) -> tuple[DirectionalVetoSummary, ...]:
    ...
```

New summary behavior:

- `n_watch = count(snapshot.state_after in {"watch", "armed", "veto"})`
- `mean_trigger_loss = mean(snapshot.rolling_symbol_return where snapshot.fired)`
- `mean_episode_bars = average contiguous veto-state run length`

### `_run_awf_simulation(...)` Behavioral Changes

No public signature change.

Replace current binary adverse-only veto block with:

1. Initialize `dict[str, ContextualDirectionalVetoState]` once per fold for configured symbols.
2. After `_combine_sleeve_signals_to_symbol(...)`, compute per-symbol rolling return using close prices up to `t`.
3. If `mode == "adverse_only"`, preserve current logic.
4. If `mode == "contextual"`, route through `_compute_contextual_directional_veto_signal(...)`.
5. Apply action:
   - `cap_mu`: `valid_signals[sym] = replace(sig, raw_mu=min(sig.raw_mu, cap_mu_bps * 1e-4))`
   - `zero_mu`: `replace(sig, raw_mu=0.0)`
   - `drop_long`: `del valid_signals[sym]`
6. Continue to update `weight_after`, `counterfactual_long_return`, and new `actual_symbol_return`.

## 3. `src/domain/futures/strategy/tiered_workflow/pipeline.py`

### Extend `DirectionalVetoReplayVariant`

```python
@dataclass(slots=True, frozen=True)
class DirectionalVetoReplayVariant:
    name: str
    directional_veto_enabled: bool
    directional_veto_mode: Literal["adverse_only", "contextual"]
    directional_veto_action: Literal["drop_long", "zero_mu", "cap_mu"]
    directional_veto_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    directional_veto_adverse_codes: tuple[int, ...] = (1, 2)
```

### Keep `DirectionalVetoReplayResult` and Extend

```python
@dataclass(slots=True, frozen=True)
class DirectionalVetoReplayResult:
    variant: str
    baseline_parity: bool
    l2_cagr: float
    l2_mdd: float
    l2_turnover: float
    l2_average_gross_exposure: float
    l2_gate_passed: bool
    l2_blocker_reason: str
    l2_directional_veto_summary: tuple[DirectionalVetoSummary, ...]
    l3_cagr: float
    l3_mdd: float
    l3_sharpe: float
    l3_total_return: float
    l3_gate_passed: bool
    l3_blocker_reason: str
    l3_realized_price_long_by_symbol: tuple[tuple[str, float], ...]
    l3_directional_veto_summary: tuple[DirectionalVetoSummary, ...]
    adoption_passed: bool
    blocker_reason: str
```

No new fields required if detail CSV is added separately.

### Update `_directional_veto_replay_variants()`

Return exact set:

```python
def _directional_veto_replay_variants() -> tuple[DirectionalVetoReplayVariant, ...]:
    return (
        DirectionalVetoReplayVariant(
            name="baseline",
            directional_veto_enabled=False,
            directional_veto_mode="adverse_only",
            directional_veto_action="drop_long",
        ),
        DirectionalVetoReplayVariant(
            name="veto_adverse_only",
            directional_veto_enabled=True,
            directional_veto_mode="adverse_only",
            directional_veto_action="drop_long",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_cap_mu",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="cap_mu",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_zero_mu",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="zero_mu",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_crisis_only",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="cap_mu",
            directional_veto_adverse_codes=(2,),
        ),
    )
```

### Update `_directional_veto_replay_adoption_verdict(...)`

Keep signature and add one parameter:

```python
def _directional_veto_replay_adoption_verdict(
    *,
    baseline: DirectionalVetoReplayResult,
    candidate: DirectionalVetoReplayResult,
    max_fit_false_positive_rate: float,
    min_gross_ratio: float,
    max_turnover_delta: float,
    max_fit_net_value_loss: float,
    min_l3_total_return_delta: float,
    max_l2_cagr_delta_loss: float,
) -> tuple[bool, str]:
    ...
```

Exact rule changes:

```python
if candidate.l2_cagr < baseline.l2_cagr - max_l2_cagr_delta_loss:
    return False, "fit_cagr_degradation"

if any(
    s.net_veto_value < -max_fit_net_value_loss
    for s in candidate.l2_directional_veto_summary
    if s.n_fired > 0
):
    return False, "fit_net_value_negative"

_bl_long_loss = sum(
    max(-v, 0.0) for sym, v in baseline.l3_realized_price_long_by_symbol
    if sym in ("BTCUSDT", "ETHUSDT")
)
_ca_long_loss = sum(
    max(-v, 0.0) for sym, v in candidate.l3_realized_price_long_by_symbol
    if sym in ("BTCUSDT", "ETHUSDT")
)
if _ca_long_loss >= _bl_long_loss:
    return False, "major_long_loss_not_improved"

if candidate.l3_total_return < baseline.l3_total_return + min_l3_total_return_delta:
    return False, "below_min_total_return_delta"
```

### Add Detail CSV Writer

```python
def _write_directional_veto_replay_detail_csv(
    results: tuple[DirectionalVetoReplayResult, ...],
    *,
    path: Path,
) -> None:
    ...
```

Required columns:

```python
variant,layer,symbol,n_obs,n_watch,n_fired,fire_rate,false_positive_rate,
opportunity_cost,avoided_loss,net_veto_value,mean_trigger_loss,mean_episode_bars
```

### `run_directional_veto_economic_replay(...)`

Behavioral changes:

1. `baseline_parity` must be computed from the replayed baseline row once.
2. Copy that same boolean into every returned variant row.
3. `variant_cfg` must set:
   - `l2_regime_directional_veto_mode`
   - `l2_regime_directional_veto_action`
   - `l2_regime_directional_veto_adverse_codes`
4. After summary list is built, write both summary CSV and detail CSV from caller env block.

## 4. `src/domain/futures/strategy/tiered_logging.py`

### `_format_directional_veto_line(...)`

Keep signature:

```python
def _format_directional_veto_line(summary: Any) -> str:
    ...
```

Append new fields when present:

```python
"watch=... trig_loss=... ep_bars=..."
```

No change to `[L2-DIRECTIONAL-VETO]` / `[L3-DIRECTIONAL-VETO]` call sites.

# 🧪 TDD Test Scenario Matrix (CRITICAL)

## Scenario 1. Happy Path - Test Setup

### S1. contextual state transitions into veto only after persistence and loss trigger

- Target: `awf_sim.py::_compute_contextual_directional_veto_signal`
- Given:
  - `mode="contextual"`
  - `raw_mu > 0`
  - adverse regime for 3 consecutive bars
  - rolling return breaches `-loss_trigger_bps`
- Expect:
  - state path `idle -> watch -> armed -> veto`
  - `fired=True` only on trigger bar
  - action-ready `raw_mu_after` for `cap_mu`

### S2. release path exits veto and enters cooldown, then idle

- Given:
  - current state `veto`
  - `raw_mu <= 0` or bull regime streak satisfied
- Expect:
  - `release_reason` captured
  - state `cooldown`
  - after `cooldown_bars`, state `idle`

### S3. `summarize_directional_veto(...)` computes new metrics

- Given snapshots with:
  - mixed watch/armed/veto states
  - at least 2 veto episodes
- Expect:
  - `n_watch`
  - `mean_trigger_loss`
  - `mean_episode_bars`
  - old metrics still preserved

### S4. replay variants include new contextual candidates

- Target: `_directional_veto_replay_variants()`
- Expect exact names:
  - `baseline`
  - `veto_adverse_only`
  - `contextual_cap_mu`
  - `contextual_zero_mu`
  - `contextual_crisis_only`

### S5. adoption gate passes on bounded fit damage plus improved holdout

- Given:
  - `candidate.l2_cagr` within allowed degradation
  - `candidate.l3_total_return` improves by configured minimum
  - `candidate.l3_mdd` lower
  - negative major-long loss reduced
- Expect:
  - `(True, "")`

## Scenario 2. Edge Cases - Validation/Bounds

### E1. look-ahead guard on rolling return window

- Constraint mapped: `Look-ahead`
- Given:
  - return spike occurs only on `t -> t+1`
- Expect:
  - trigger at `t` does not see that spike
  - rolling return uses `[t-lookback, t)` only

### E2. fold boundary reset

- Constraint mapped: `Fold boundary`
- Given:
  - end of fold A left state in `veto`
  - fold B starts immediately
- Expect:
  - new fold state is `idle`
  - no carryover streak or cooldown

### E3. missing signal does not fire and decays state

- Constraint mapped: `Missing symbol`
- Expect:
  - snapshot `was_missing=True`
  - `fired=False`
  - state returns or remains `idle`

### E4. regime source remains compressed external code

- Constraint mapped: `Regime source`
- Given:
  - `regime_code_t` supplied from replay caller
- Expect:
  - helper consumes integer code only
  - no price-based regime recomputation path

### E5. BNB remains control symbol

- Constraint mapped: `BNB control`
- Expect:
  - default treatment symbols exclude `BNBUSDT`
  - BNB can still appear in `major_symbol_diag`
  - no BNB veto snapshots by default

### E6. no forced short on negative `raw_mu`

- Constraint mapped: `No forced short`
- Given:
  - `raw_mu < 0`
- Expect:
  - action never flips to short
  - signal unchanged by contextual veto

### E7. `cap_mu` preserves gross better than `drop_long`

- Constraint mapped: `Turnover`
- Given:
  - identical trigger context
- Expect:
  - `cap_mu` keeps symbol in candidate set
  - `weight_after` remains finite or near-zero
  - `drop_long` removes symbol

### E8. explicit denominator guards

- Constraint mapped: `Numerical guard`
- Given:
  - empty snapshot tuples
  - `n_fired=0`
  - `n_obs=0`
- Expect:
  - all ratios finite
  - no division warnings

### E9. CSV traceability writes summary and detail rows

- Constraint mapped: `CSV traceability`
- Expect:
  - summary CSV still writes variant-level metrics
  - detail CSV writes per-layer per-symbol veto summaries

### E10. runtime shape remains small and Python-side only

- Constraint mapped: `Runtime`
- Test style:
  - unit test verifies helper accepts scalar inputs and state objects
  - no numba decoration or ndarray-of-objects dependency introduced

## Scenario 3. Error Handling - Exceptions

### X1. invalid veto mode

- Input:
  - `l2_regime_directional_veto_mode="invalid"`
- Expect:
  - `ValueError`

### X2. invalid action outside allowed set

- Input:
  - `l2_regime_directional_veto_action="flip_short"`
- Expect:
  - `ValueError`

### X3. non-positive persistence/lookback/release bars

- Inputs:
  - `persistence_bars=0`
  - `loss_lookback_bars=0`
  - `release_regime_bull_bars=0`
- Expect:
  - `ValueError`

### X4. negative thresholds

- Inputs:
  - `loss_trigger_bps < 0`
  - `cap_mu_bps < 0`
  - `max_fit_net_value_loss < 0`
  - `min_l3_total_return_delta < 0`
  - `max_l2_cagr_delta_loss < 0`
- Expect:
  - `ValueError`

### X5. adoption gate catches fit damage

- Expect blockers:
  - `fit_cagr_degradation`
  - `fit_net_value_negative`
  - `fit_false_positive`

### X6. adoption gate uses negative long-loss metric

- Given:
  - baseline long contribution `("BTCUSDT", -0.04)`
  - candidate long contribution `("BTCUSDT", -0.02)`
- Expect:
  - improved loss metric
  - no false blocker from positive-side sum logic

## Mock Boilerplate Snippet

```python
from types import SimpleNamespace

import numpy as np

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    ContextualDirectionalVetoState,
    DirectionalVetoSnapshot,
    DirectionalVetoSummary,
    Layer2FoldAttribution,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
)


def make_cfg(**overrides: object) -> Layer2AllocationConfig:
    base = {
        "l2_regime_directional_veto_enabled": True,
        "l2_regime_directional_veto_mode": "contextual",
        "l2_regime_directional_veto_symbols": ("BTCUSDT", "ETHUSDT"),
        "l2_regime_directional_veto_adverse_codes": (1, 2),
        "l2_regime_directional_veto_action": "cap_mu",
        "l2_regime_directional_veto_persistence_bars": 3,
        "l2_regime_directional_veto_loss_lookback_bars": 18,
        "l2_regime_directional_veto_loss_trigger_bps": 150.0,
        "l2_regime_directional_veto_cap_mu_bps": 0.0,
        "l2_regime_directional_veto_release_raw_mu_nonpos": True,
        "l2_regime_directional_veto_release_regime_bull_bars": 2,
        "l2_regime_directional_veto_cooldown_bars": 3,
        "l2_regime_directional_veto_max_fit_false_positive_rate": 0.50,
        "l2_regime_directional_veto_max_fit_net_value_loss": 0.0,
        "l2_regime_directional_veto_min_gross_ratio": 0.90,
        "l2_regime_directional_veto_max_turnover_delta": 0.05,
        "l2_regime_directional_veto_min_l3_total_return_delta": 0.02,
        "l2_regime_directional_veto_max_l2_cagr_delta_loss": 0.005,
    }
    base.update(overrides)
    return Layer2AllocationConfig.from_mapping(base)


def make_attr(snaps: tuple[DirectionalVetoSnapshot, ...]) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=1,
        n_rebal=1,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        directional_veto_snapshots=snaps,
    )


def make_replay_result(**overrides: object) -> SimpleNamespace:
    row = SimpleNamespace(
        variant="contextual_cap_mu",
        baseline_parity=True,
        l2_cagr=0.24,
        l2_mdd=0.22,
        l2_turnover=0.12,
        l2_average_gross_exposure=0.50,
        l2_gate_passed=False,
        l2_blocker_reason="cagr",
        l2_directional_veto_summary=(
            DirectionalVetoSummary(
                symbol="BTCUSDT",
                n_obs=10,
                n_missing=0,
                n_adverse=6,
                n_fired=2,
                fire_rate=0.2,
                adverse_fire_rate=2 / 6,
                false_positive_rate=0.0,
                opportunity_cost=0.0,
                avoided_loss=0.03,
                net_veto_value=0.03,
                n_watch=4,
                mean_trigger_loss=-0.025,
                mean_episode_bars=2.0,
            ),
        ),
        l3_cagr=-0.05,
        l3_mdd=0.18,
        l3_sharpe=-0.30,
        l3_total_return=-0.04,
        l3_gate_passed=False,
        l3_blocker_reason="negative_return",
        l3_realized_price_long_by_symbol=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
        l3_directional_veto_summary=(),
        adoption_passed=False,
        blocker_reason="",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row
```

# 🛠️ Algorithmic Plan

## Target Location 1

- File: `src/domain/futures/strategy/tiered_workflow/dataclasses.py`
- Anchor:
  - `class Layer2AllocationConfig`
  - `_validate_directional_veto_action`
  - `from_mapping(...)`

### Steps

1. Extend validator set for `mode` and `action="cap_mu"`.
2. Add new contextual config fields.
3. Enforce lower bounds and non-negative thresholds.

## Target Location 2

- File: `src/domain/futures/strategy/tiered_workflow/awf_sim.py`
- Anchor:
  - `# [L2 Directional Veto]: adverse regime long veto`
  - `summarize_directional_veto(...)`

### Steps

1. Extend snapshot/summary dataclasses with contextual fields.
2. Add `ContextualDirectionalVetoState`.
3. Add scalar helper for state transition.
4. Add causal rolling-return helper using `aligned.close_2d[:t]`.
5. Replace current veto block with:
   - `mode == "adverse_only"` -> current path
   - `mode == "contextual"` -> state machine path
6. Keep action insertion point before `rank_and_select`.
7. Accumulate `actual_symbol_return` and `counterfactual_long_return` inside bar loop.
8. Extend summary aggregation with watch counts and episode stats.

## Target Location 3

- File: `src/domain/futures/strategy/tiered_workflow/pipeline.py`
- Anchor:
  - `_directional_veto_replay_variants`
  - `_directional_veto_replay_adoption_verdict`
  - `_write_directional_veto_replay_csv`

### Steps

1. Expand replay variants to 5-arm.
2. Pass contextual config knobs into each variant config.
3. Fix major-long-loss computation to use negative realized contribution.
4. Add fit-CAGR degradation and min-holdout-improvement gates.
5. Write companion detail CSV for per-symbol veto summaries.
6. Propagate replayed baseline parity to every variant row.

## Target Location 4

- File: `src/domain/futures/strategy/tiered_logging.py`
- Anchor:
  - `_format_directional_veto_line`

### Steps

1. Append contextual metrics when present.
2. Keep old formatting fields stable so current tests still pass with updated expectations.

## Recommended Test Placement

- `tests/unit/domain/futures/strategy/tiered_workflow/test_directional_veto.py`
  - state machine
  - summary metrics
  - config validation
  - causal rolling-return window
- `tests/unit/domain/futures/strategy/tiered_workflow/test_directional_veto_replay.py`
  - replay variants
  - adoption gate blockers
  - detail CSV output
  - negative long-loss metric fix
- `tests/unit/domain/futures/strategy/test_tiered_workflow.py`
  - focused integration around existing AWF veto insertion point
