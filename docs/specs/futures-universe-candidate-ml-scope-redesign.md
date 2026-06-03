# Futures Universe Candidate ML Scope Redesign

## Spec Type
`prd` / `quant` / `refactor`

## Problem Statement
The futures universe pipeline currently preserves a Stage 5 survivor panel as `training_panel`,
`inference_panel`, and `live_inference_panel`, while Stage 6 is the actual selected execution
membership. This leaves legacy alpha-ranking semantics in the candidate ML path and can expand
the loaded/validated panel far beyond the assets selected for trading. For candidate ML, a large
Stage 5 panel increases cross-sectional noise, dilutes per-symbol edge, and makes model training
less consistent with the dynamic universe constraints used by the backtest and optimizer.

The target behavior is:
- Stage 6 selected assets are the only candidate ML training, inference, and trading scope.
- Stage 5 remains an audit/research survivor set, not a default ML panel.
- Stage 3/4 gates reject weak execution candidates earlier.
- Cluster/beta/diversification diagnostics are carried into candidate ML features without
  introducing look-ahead bias.

## Design Decisions

### 1. ML Panel Scoping
Accept.

Candidate ML should operate on Stage 6 selected assets only. Stage 5 survivors are useful for
audit and research, but they should not be the default candidate event universe. The ML panel must
follow the same quarterly Stage 6 membership used by `universe_active_mask`.

### 2. Stage 3 and Stage 4 Hard Gates
Partially accept.

Strengthen execution feasibility defaults and add an optional OI/ADV crowding gate in Stage 3.
Do not make the gates so strict that Stage 6 cannot maintain `k_in` under volatile regimes. The
quality gate should verify a minimum selected count and fail loudly if Stage 3/4 reduce the pool
below viable ML support.

### 3. Cluster Metrics as ML Feature Feedback
Accept as causal static metadata.

Use `cluster_id`, cluster size, beta-vs-market, and anchor-cluster membership as candidate ML
features. These values must come from the universe snapshot as of the quarter boundary or from
columns injected into symbol frames before alignment. They must not be recomputed from OOS future
returns inside `build_candidate_dataset`.

## Target Files

Primary source files:
- `src/domain/futures/universe/config.py`
- `src/domain/futures/universe/filters.py`
- `src/domain/futures/universe/pipeline.py`
- `src/domain/futures/universe/models.py`
- `src/domain/futures/universe/storage.py`
- `src/application/futures/optimization/universe_service.py`
- `src/application/futures/optimization/strategy_service.py`
- `src/execution/opt_main_futures.py`
- `src/domain/futures/strategy/common/alignment.py`
- `src/domain/futures/strategy/candidate_dataset.py`

Tests:
- `tests/unit/domain/futures/universe/test_oi_adv_filter.py`
- `tests/unit/domain/futures/universe/test_selection.py`
- `tests/unit/domain/futures/universe/test_storage.py`
- `tests/unit/application/futures/optimization/test_universe_service.py`
- `tests/unit/application/futures/optimization/test_strategy_service.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`
- `tests/unit/domain/futures/strategy/test_candidate_dataset.py`

## Contracts

### `Stage3Config`
Replace the dataclass fields with backward-compatible defaults plus optional crowding controls:

```python
@dataclass(frozen=True, slots=True)
class Stage3Config:
    """Liquidity and execution feasibility gates."""

    min_adv_usdt_median: float = 50_000_000.0
    max_amihud_30d: float = 1.00e-9
    max_clip_to_adv: float = 0.0025
    enable_oi_adv_crowding_gate: bool = True
    max_oi_to_adv: float = 12.0
    screening_tier: str = "mid"
    screening_clip_usdt_by_tier: dict[str, float] = field(...)
    capacity_clip_usdt_list: tuple[float, ...] = (50_000.0, 100_000.0)
```

### `Stage4Config`
Tighten execution-cost defaults:

```python
@dataclass(frozen=True, slots=True)
class Stage4Config:
    """Execution-cost model gates."""

    max_execution_cost_bps: float = 35.0
    default_taker_fee_bps: float = 5.0
    default_half_spread_bps: float = 1.0
    spread_source_switch_date: str = "2020-01-01"
    pre2020_half_spread_bps: float = 2.5
    post2020_half_spread_bps: float = 1.0
    default_impact_coef_bps: float = 18.0
```

### `UniverseConfig`
Change the default strategy pool mode:

```python
strategy_pool_mode: str = "stage6_selected"
```

Allowed values:
- `"stage6_selected"`: production default.
- `"stage5_research"`: research-only opt-in for future experiments. Do not use this mode in
  `opt_main_futures.py` unless an explicit CLI/config switch is added and tested.

### `UniverseSnapshot`
Keep existing fields for compatibility, but change semantics:

```python
training_panel: tuple[str, ...] = field(default_factory=tuple)
inference_panel: tuple[str, ...] = field(default_factory=tuple)
live_inference_panel: tuple[str, ...] = field(default_factory=tuple)
historical_trading_panel: tuple[str, ...] = field(default_factory=tuple)
inference_panel_quarter_membership: dict[date, tuple[str, ...]] = field(default_factory=dict)
stage5_research_panel: tuple[str, ...] = field(default_factory=tuple)
```

Semantics:
- `training_panel`: current-quarter Stage 6 selected symbols.
- `live_inference_panel`: current-quarter Stage 6 selected symbols.
- `inference_panel`: historical quarterly union of Stage 6 selected symbols.
- `historical_trading_panel`: historical quarterly union of Stage 6 selected symbols.
- `inference_panel_quarter_membership`: quarter -> Stage 6 selected symbols.
- `stage5_research_panel`: current-quarter Stage 5 survivors for audit only.

`snapshot_to_payload` and `snapshot_from_payload` must persist and read `stage5_research_panel`
with default `()` for older snapshots.

### `apply_liquidity_stage`
Keep the public signature:

```python
def apply_liquidity_stage(
    frame: pd.DataFrame,
    config: Stage3Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Additional behavior:
- Read OI from the first available column in:
  `("oi_usdt_median", "sum_open_interest_value", "open_interest_usdt", "open_interest")`.
- Compute `oi_to_adv = oi_usdt / adv_usdt_median`.
- If `enable_oi_adv_crowding_gate` is true and OI exists, require
  `oi_to_adv <= max_oi_to_adv`.
- Missing OI must not reject a symbol.
- Add `oi_to_adv` to the Stage 3 report.
- Reject reason priority:
  `adv_too_low -> amihud_too_high -> clip_too_large_vs_adv -> oi_adv_crowded`.

### `build_universe`
Keep the public signature:

```python
def build_universe(
    *,
    as_of: str | date,
    tf: str,
    cfg: dict[str, Any] | UniverseConfig | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    previous_selection: tuple[str, ...] | None = None,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    ...
```

Required behavior:
- Preserve Stage 5 survivors in `stage5_research_panel`.
- Set `training_panel` and `live_inference_panel` from `s6`, not `s5`.
- Set `historical_trading_panel` only in the timeline layer, not inside a single snapshot build.
- Include `oi_usdt_median`, `sum_open_interest_value`, or `open_interest_usdt` in the ledger
  column request only if the ledger has the column. Do this through a helper that checks existing
  parquet schema to avoid requesting nonexistent physical columns.

### `load_or_build_universe_snapshot`
Keep the public signature.

Required behavior for cached snapshots:
- If `training_panel` or `live_inference_panel` is empty or has a length greater than
  `n_stage6_selected`, rebuild it from `snapshot.selected`.
- If `inference_panel` is read from legacy Stage 5 payload, do not reuse it as production ML scope.
- Set `stage5_research_panel` from the old `live_inference_panel` or Stage 5 report for audit.

### `discover_universe_timeline`
Keep the public signature.

Required behavior:
- `UniverseTimelineResult.symbols` remains the historical Stage 6 union.
- `inference_symbols` becomes the historical Stage 6 union.
- `inference_timeline` windows use Stage 6 members, not Stage 5.
- `inference_panel_quarter_membership` maps each quarter to Stage 6 members.
- OOS snapshot replacement sets:
  - `inference_panel=tuple(sorted(all_symbols))`
  - `historical_trading_panel=tuple(sorted(all_symbols))`
  - `inference_panel_quarter_membership` from Stage 6 timeline

### `_resolve_data_collection_symbols`
Keep the public signature:

```python
def _resolve_data_collection_symbols(
    *,
    run_config: FuturesRunConfig,
    discovered_symbols: list[str],
    inference_panel: tuple[str, ...],
    live_inference_panel: tuple[str, ...],
) -> tuple[str, ...]:
    ...
```

Required behavior:
- Use Stage 6 production scope only:
  `base_symbols = list(inference_panel or live_inference_panel or discovered_symbols)`.
- Add `FUTURES_ANCHOR_SYMBOLS` and `FUTURES_MACRO_INDEX_SYMBOLS`.
- Return stable de-duplicated order.
- Remove the hardcoded `scope = "stage5_passed"`.

### `_run_data_stage`
Keep the public signature.

Required behavior:
- Use `scope_name = "stage6_selected"` when either `inference_panel` or `live_inference_panel`
  is present because those panels are now Stage 6 scoped.
- `data_stage.valid_symbols` should normally match Stage 6 historical union plus anchors/macros
  that pass readiness, not Stage 5 survivors.
- Membership injection uses the Stage 6 timeline for both trading and inference masks. If
  `inference_timeline` is present, it must be Stage 6 equivalent.

### `_run_strategy_stage`
Keep the public signature.

Required behavior:
- `bridge_trading_symbols` must be the Stage 6 selected symbols passed from `_run_universe_stage`.
- `full_strategy_maps` may contain support symbols, but `run_active_strategy_output_bridge.symbols`
  must receive only Stage 6 trading symbols.
- Log both loaded support count and ML candidate scope count.

### `run_active_strategy_output_bridge`
Keep the public signature.

Required behavior:
- Stop deleting `trading_symbols`.
- Build `effective_symbols` from `trading_symbols or symbols`.
- If `training_panel` is provided, intersect `effective_symbols` with it.
- Intersect the final list with `preloaded_data_maps`.
- Raise `ValueError("candidate ML scope is empty")` if the final list is empty.

Pseudo-code:

```python
candidate_scope = list(trading_symbols or symbols)
if training_panel:
    allowed = set(training_panel)
    candidate_scope = [sym for sym in candidate_scope if sym in allowed]
effective_symbols = [sym for sym in dict.fromkeys(candidate_scope) if sym in preloaded_data_maps]
if not effective_symbols:
    raise ValueError("candidate ML scope is empty")
```

### `AlignedMarketData`
Add static metadata arrays:

```python
cluster_id_1d: NDArray[np.float32] | None = None
beta_vs_market_1d: NDArray[np.float32] | None = None
cluster_size_1d: NDArray[np.float32] | None = None
anchor_cluster_1d: NDArray[np.float32] | None = None
```

Implementation rule:
- Reuse `symbol_meta` as the collection mechanism.
- Read these frame columns when present:
  `cluster_id`, `beta_vs_market`, `cluster_size`, `anchor_cluster_member`.
- Populate the dedicated arrays from `symbol_meta`.

### `build_candidate_dataset`
Keep the public signature.

Required feature additions:
- Add four feature names after `funding_z20`:
  - `universe_cluster_id`
  - `universe_beta_vs_market`
  - `universe_cluster_size`
  - `universe_anchor_cluster_member`
- Use values from `aligned.cluster_id_1d`, `aligned.beta_vs_market_1d`,
  `aligned.cluster_size_1d`, and `aligned.anchor_cluster_1d`.
- Fallbacks:
  - `cluster_id`: `-1.0`
  - `beta_vs_market`: `0.0`
  - `cluster_size`: `1.0`
  - `anchor_cluster_member`: `0.0`
- Do not recompute clusters from event-time OOS returns in this function.

## Step-by-Step Logic

1. Update config defaults.
2. Add Stage 3 OI/ADV crowding logic and report column.
3. Change snapshot production semantics so Stage 6 selected symbols populate production ML panels.
4. Add `stage5_research_panel` to preserve Stage 5 survivor auditability.
5. Normalize cached legacy snapshots so old Stage 5 `live_inference_panel` cannot flow into
   production candidate ML.
6. Change quarterly timeline construction to use Stage 6 for `inference_*` fields.
7. Change data collection scope to Stage 6 union plus support symbols.
8. Change strategy bridge scope resolution to use `trading_symbols`/`training_panel`.
9. Carry universe cluster metadata through alignment.
10. Add cluster metadata features to candidate dataset.
11. Update unit tests to assert Stage 6 scoping and feature schema.

## Surgical Plan

### `src/domain/futures/universe/config.py`
`[ACTION: REPLACE]`
- In `Stage3Config`, update `min_adv_usdt_median`, `max_amihud_30d`,
  `max_clip_to_adv`, and add `enable_oi_adv_crowding_gate`, `max_oi_to_adv`.
- In `Stage4Config`, change `max_execution_cost_bps` to `35.0`.
- In `UniverseConfig`, change `strategy_pool_mode` default to `"stage6_selected"`.

### `src/domain/futures/universe/filters.py`
`[ACTION: ADD]`
- Add `_first_available_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None`
  near `_numeric_series`.
- In `apply_liquidity_stage`, compute OI/ADV only when an OI column exists.
- Add the OI/ADV predicate to `pass_mask`.
- Add `oi_to_adv` to `out` and `report`.

### `src/domain/futures/universe/models.py`
`[ACTION: ADD]`
- Add `stage5_research_panel` to `UniverseSnapshot` with a default empty tuple.
- Update comments for `training_panel`, `inference_panel`, and `live_inference_panel` to Stage 6
  production semantics.

### `src/domain/futures/universe/storage.py`
`[ACTION: REPLACE]`
- Persist `stage5_research_panel` in `snapshot_to_payload`.
- Read it in `snapshot_from_payload` with `payload.get("stage5_research_panel", [])`.

### `src/domain/futures/universe/pipeline.py`
`[ACTION: REPLACE]`
- In `build_universe`, define:

```python
stage5_passed_symbols = tuple(sorted(s5["symbol"].dropna().astype(str).tolist()))
stage6_selected_symbols = tuple(sorted(s6["symbol"].dropna().astype(str).tolist()))
```

- Set:

```python
training_panel=stage6_selected_symbols
live_inference_panel=stage6_selected_symbols
stage5_research_panel=stage5_passed_symbols
```

- In `load_or_build_universe_snapshot`, after creating `snapshot`, normalize:

```python
selected_symbols = tuple(str(meta.symbol) for meta in snapshot.selected)
legacy_stage5 = tuple(snapshot.live_inference_panel or _stage5_symbols_from_report(report))
snapshot = replace(
    snapshot,
    training_panel=selected_symbols,
    live_inference_panel=selected_symbols,
    stage5_research_panel=legacy_stage5,
)
```

Use `dataclasses.replace`; do not mutate the frozen dataclass.

### `src/application/futures/optimization/universe_service.py`
`[ACTION: REPLACE]`
- Replace `quarter_stage5 = frozenset(snapshot.live_inference_panel)` with Stage 6 `current_set`.
- Rename local variables from `stage5` to `ml_panel` or `stage6_panel`.
- Set `inference_symbols_set.update(current_set)`.
- Build `inference_windows` from Stage 6 `current_set` values.
- Update debug log key from `stg5=%d` to `ml=%d`.

### `src/execution/opt_main_futures.py`
`[ACTION: REPLACE]`
- In `_resolve_data_collection_symbols`, remove hardcoded `scope = "stage5_passed"` and select:

```python
base_symbols = list(inference_panel or live_inference_panel or discovered_symbols)
```

- In `_run_data_stage`, set `scope_name = "stage6_selected"` when either panel is present.
- In `_run_strategy_stage`, pass:

```python
training_panel=trading_symbols or tuple(data_stage.valid_symbols)
trading_symbols=trading_symbols or tuple(data_stage.valid_symbols)
```

to `run_active_strategy_output_bridge`.

### `src/application/futures/optimization/strategy_service.py`
`[ACTION: REPLACE]`
- Remove `trading_symbols` and `training_panel` from the `del (...)` block.
- Resolve `effective_symbols` from `trading_symbols or symbols`, optionally intersected with
  `training_panel`.
- Raise `ValueError("candidate ML scope is empty")` on empty scope.

### `src/domain/futures/strategy/common/alignment.py`
`[ACTION: REPLACE]`
- Extend `_meta_cols_to_read` with:
  `("beta_vs_market", "cluster_size", "anchor_cluster_member")`.
- Add dedicated arrays to `AlignedMarketData`.
- Populate dedicated arrays from `symbol_meta` before returning.

### `src/domain/futures/strategy/candidate_dataset.py`
`[ACTION: REPLACE]`
- Add static universe features to each `row_features` using `sym_idx`.
- Extend `feature_names` with the four universe feature names.
- Maintain feature schema stability for empty splits.

## Tests To Add Or Update

### `tests/unit/domain/futures/universe/test_oi_adv_filter.py`
- Replace helper-only OI tests with direct `apply_liquidity_stage` tests.
- Assert `oi_adv_crowded` rejects when `oi_to_adv > max_oi_to_adv`.
- Assert missing OI column does not reject.
- Assert `oi_to_adv` is present in the report.

### `tests/unit/application/futures/optimization/test_universe_service.py`
- Add a snapshot fixture where `selected=("BTCUSDT",)` and `live_inference_panel` contains
  legacy Stage 5 symbols.
- Assert `discover_universe_timeline().inference_symbols == result.symbols`.
- Assert inference timeline windows use Stage 6 symbols only.

### `tests/unit/execution/test_opt_main_futures_strategy_mode.py`
- Update `test_resolve_data_collection_symbols_uses_inference_panel` to expect Stage 6 panel
  semantics.
- Add a test proving `live_inference_panel` is used only when `inference_panel` is empty.

### `tests/unit/application/futures/optimization/test_strategy_service.py`
- Add a test where `symbols` contains Stage 5 survivors but `trading_symbols` contains only
  Stage 6. Patch `run_candidate_strategy_for_universe` and assert it receives Stage 6 only.
- Add an empty-scope error test.

### `tests/unit/domain/futures/strategy/test_candidate_dataset.py`
- Extend `_make_aligned()` with cluster/beta arrays.
- Assert new feature names exist.
- Assert row values match the aligned symbol metadata.
- Assert empty split still returns the same feature schema length.

## Verification

L1 commands after implementation:

```bash
uv run ruff check --fix src/domain/futures/universe/config.py src/domain/futures/universe/filters.py src/domain/futures/universe/models.py src/domain/futures/universe/storage.py src/domain/futures/universe/pipeline.py src/application/futures/optimization/universe_service.py src/application/futures/optimization/strategy_service.py src/execution/opt_main_futures.py src/domain/futures/strategy/common/alignment.py src/domain/futures/strategy/candidate_dataset.py tests/unit/domain/futures/universe/test_oi_adv_filter.py tests/unit/domain/futures/universe/test_selection.py tests/unit/domain/futures/universe/test_storage.py tests/unit/application/futures/optimization/test_universe_service.py tests/unit/application/futures/optimization/test_strategy_service.py tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/unit/domain/futures/strategy/test_candidate_dataset.py
```

```bash
uv run mypy src/domain/futures/universe/config.py src/domain/futures/universe/filters.py src/domain/futures/universe/models.py src/domain/futures/universe/storage.py src/domain/futures/universe/pipeline.py src/application/futures/optimization/universe_service.py src/application/futures/optimization/strategy_service.py src/execution/opt_main_futures.py src/domain/futures/strategy/common/alignment.py src/domain/futures/strategy/candidate_dataset.py
```

Focused tests:

```bash
uv run pytest tests/unit/domain/futures/universe/test_oi_adv_filter.py tests/unit/domain/futures/universe/test_selection.py tests/unit/domain/futures/universe/test_storage.py tests/unit/application/futures/optimization/test_universe_service.py tests/unit/application/futures/optimization/test_strategy_service.py tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/unit/domain/futures/strategy/test_candidate_dataset.py --tb=short
```

Coverage checks for changed source modules:

```bash
uv run pytest tests/unit/domain/futures/universe/test_oi_adv_filter.py --cov=src/domain/futures/universe/filters --cov-report=term-missing
```

```bash
uv run pytest tests/unit/domain/futures/strategy/test_candidate_dataset.py --cov=src/domain/futures/strategy/candidate_dataset --cov-report=term-missing
```

Smoke run:

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode strategy-smoke --symbols BTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01
```

Expected outcomes:
- Universe log reports production panels near Stage 6 size, not Stage 5 survivor count.
- Candidate ML bridge receives Stage 6 selected symbols only.
- Data loading may include anchors/macros, but candidate event generation excludes non-Stage 6
  symbols.
- No look-ahead is introduced: all cluster/beta features are sourced from PIT universe metadata.

## Risk Notes

- Stronger Stage 3/4 gates may reduce the candidate pool below viable ML sample support during
  stress regimes. `validate_universe_quality` should fail when Stage 6 cannot satisfy `k_in`
  rather than silently training on a tiny panel.
- Cached snapshots created with old Stage 5 semantics can contaminate runs unless normalized or
  rebuilt. The implementation must either rebuild on config-hash change or sanitize cached panels.
- Cluster features can become pseudo-label leakage if recomputed from OOS returns. Only snapshot
  metadata or PIT ledger-derived values are allowed.
- Reducing the panel can lower event count. The acceptance criterion is not more trades; it is
  consistency between selection, candidate ML, and optimizer/backtest membership.
