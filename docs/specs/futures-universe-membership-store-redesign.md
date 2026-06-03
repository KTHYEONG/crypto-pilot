# Futures Universe Membership Store Redesign

## Spec Type
`prd` / `quant` / `refactor`

## Problem Statement

The current universe persistence model stores each quarter as a materialized `UniverseSnapshot`
plus duplicated selected-symbol parquet and filter-report parquet artifacts. This makes a snapshot
both the domain DTO and the storage source of truth. The coupling has three costs:

- stale panel fields can bypass updated Stage 6 scope rules when old snapshot payloads are reused;
- per-symbol ML metadata can be lost if `SymbolMeta` does not mirror all selected-frame columns;
- timeline discovery must decode/reconstruct snapshot payloads even when it only needs membership.

The more efficient model is to keep `UniverseSnapshot` as an in-memory compatibility DTO, but move
the persistent source of truth to a partitioned, append-only membership store keyed by
`(tf, as_of, config_hash, data_manifest_hash, run_id)`.

## Design Decision

Use a `Parquet membership store` as the primary universe persistence layer.

Rejected alternatives:

- **No persistence, recompute every run:** clean but slow, and replay depends on immutable input
  manifests being available. This is acceptable as a debug mode only.
- **In-memory process cache only:** fast but cannot support replay, audit, or quarterly timeline
  reconstruction across process boundaries.
- **External DB/DuckDB store:** good query ergonomics, but unnecessary dependency and operational
  surface for the current artifact size.
- **Keep snapshot JSON/parquet as SSOT:** easy short-term, but keeps the stale-field problem and
  forces DTO evolution to carry every audit/query need.

Recommended behavior:

- Persist universe outputs as normalized per-symbol decision rows and one run manifest row.
- Materialize `UniverseSnapshot` from the store only when existing APIs require it.
- Preserve legacy snapshot readers/writers as compatibility wrappers during the migration.
- Use exact `config_hash` and `data_manifest_hash` matching for cache hits; otherwise rebuild.

## Quant Integrity Requirements

- Membership rows are PIT artifacts: all rows for an `as_of` date must be derived only from
  ledger rows with `knowledge_date <= as_of`.
- Candidate ML features must use `cluster_id`, `beta_vs_market`, `cluster_size`, and
  `anchor_cluster_member` from persisted membership decision rows, not from OOS-aligned frames.
- Quarterly membership masks must be generated from Stage 6 selected rows only.
- Stage 5 rows may exist in the store for audit and research, but must not enter production ML
  scope unless an explicit research mode is added later.

## Target Files

Primary source files:

- `src/domain/futures/universe/models.py`
- `src/domain/futures/universe/store.py` (new)
- `src/domain/futures/universe/storage.py`
- `src/domain/futures/universe/pipeline.py`
- `src/domain/futures/universe/__init__.py`
- `src/application/futures/optimization/universe_service.py`
- `src/execution/opt_main_futures.py`

Tests:

- `tests/unit/domain/futures/universe/test_store.py` (new)
- `tests/unit/domain/futures/universe/test_storage.py`
- `tests/unit/domain/futures/universe/test_quarterly_selection_audit.py`
- `tests/unit/application/futures/optimization/test_universe_service.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`

## Contracts

### Storage Layout

Primary root:

```python
DEFAULT_UNIVERSE_STORE_ROOT = LOG_DIR / "futures/universe/store/v1"
```

Partition layout:

```text
{root}/runs/tf={tf}/as_of={YYYY-MM-DD}/run_id={run_id}/manifest.parquet
{root}/runs/tf={tf}/as_of={YYYY-MM-DD}/run_id={run_id}/decisions.parquet
{root}/runs/tf={tf}/as_of={YYYY-MM-DD}/run_id={run_id}/filter_report.parquet
```

`run_id` is deterministic:

```python
run_id = sha256(f"{tf}|{as_of}|{config_hash}|{data_manifest_hash}").hexdigest()[:16]
```

### `SymbolMeta`

Extend the existing dataclass without removing fields:

```python
@dataclass(frozen=True, slots=True)
class SymbolMeta:
    symbol: str
    role: str
    adv_usdt: float
    execution_cost_bps: float
    funding_carry_8h: float
    beta_vs_market: float
    cluster_id: int
    tradeable_rank: int
    basis_annualized_mean: float | None
    basis_vol: float | None
    capacity_clip_usdt_list: tuple[float, ...]
    cluster_size: float = 1.0
    anchor_cluster_member: float = 0.0
```

Meaning:

- `cluster_size`: original selection-stage cluster size from the Stage 5 candidate pool.
- `anchor_cluster_member`: original selection-stage indicator for membership in an anchor-linked
  cluster, not `role == "anchor"`.

### `UniverseRunManifest`

Add in `src/domain/futures/universe/models.py`:

```python
@dataclass(frozen=True, slots=True)
class UniverseRunManifest:
    as_of: str
    tf: str
    schema_version: int
    run_id: str
    config_hash: str
    data_manifest_hash: str
    generated_at_utc: str
    ledger_confidence: str
    basket_ref: tuple[str, ...]
    basket_weights: tuple[float, ...]
    n_stage0: int
    n_stage1_pass: int
    n_stage2_pass: int
    n_stage3_pass: int
    n_stage4_pass: int
    n_stage5_pass: int
    n_stage6_selected: int
```

### Decision Frame Schema

`decisions.parquet` must contain these columns:

```python
UNIVERSE_DECISION_COLUMNS = (
    "as_of",
    "tf",
    "run_id",
    "config_hash",
    "data_manifest_hash",
    "symbol",
    "stage5_pass",
    "stage6_selected",
    "stage",
    "selection_reason",
    "role",
    "rank",
    "tradeable_score",
    "execution_pool_score",
    "adv_usdt_median",
    "execution_cost_bps",
    "funding_rate_8h",
    "beta_vs_market",
    "cluster_id",
    "cluster_size",
    "anchor_cluster_member",
    "basis_annualized_mean",
    "basis_vol",
    "capacity_clip_usdt_list",
    "reject_code",
    "final_rank",
    "generated_at_utc",
)
```

Rules:

- Stage 6 selected symbols have `stage6_selected == True`.
- Stage 5 survivors that are not selected have `stage5_pass == True` and
  `stage6_selected == False`.
- Hard-gated rejects can be included for audit with both flags false.
- Production ML panels are derived only from `stage6_selected == True`.

### Store API

Create `src/domain/futures/universe/store.py`.

```python
def compute_universe_run_id(
    *,
    as_of: str | date,
    tf: str,
    config_hash: str,
    data_manifest_hash: str,
) -> str:
    ...
```

```python
def write_universe_store_run(
    *,
    manifest: UniverseRunManifest,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
    root: Path = DEFAULT_UNIVERSE_STORE_ROOT,
) -> Path:
    """Write one deterministic universe run partition and return the run directory."""
```

```python
def load_universe_store_run(
    *,
    as_of: str | date,
    tf: str,
    config_hash: str,
    data_manifest_hash: str,
    root: Path = DEFAULT_UNIVERSE_STORE_ROOT,
) -> tuple[UniverseRunManifest, pd.DataFrame, pd.DataFrame] | None:
    """Return exact-hash run artifacts, or None if absent."""
```

```python
def materialize_snapshot_from_store(
    *,
    manifest: UniverseRunManifest,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    """Build the legacy DTO and selected frame from normalized store artifacts."""
```

```python
def build_decision_frame(
    *,
    manifest: UniverseRunManifest,
    stage5_frame: pd.DataFrame,
    stage6_frame: pd.DataFrame,
    report: pd.DataFrame,
) -> pd.DataFrame:
    """Build normalized decision rows from pipeline stage frames."""
```

### Existing Public API Compatibility

Keep signatures:

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

```python
def load_or_build_universe_snapshot(
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

Behavior changes:

- `snapshot_root` remains accepted for backward compatibility, but new writes use
  `DEFAULT_UNIVERSE_STORE_ROOT`.
- `load_or_build_universe_snapshot` first computes current `config_hash` and
  `data_manifest_hash`, then calls `load_universe_store_run`.
- Legacy snapshot files are fallback only. If legacy payload has stale/missing panel or metadata
  fields, rebuild and write the new store.

## Step-by-Step Logic

### Build Path

1. `build_universe` computes `config_hash`, `data_manifest_hash`, and deterministic `run_id`.
2. Pipeline executes Stage 1 through Stage 6 exactly as today.
3. Build `UniverseRunManifest`.
4. Build normalized `decisions` from `s5`, `s6`, and `report`.
5. Materialize `UniverseSnapshot` and `selected` from `manifest + decisions + report`.
6. Write the store run partition.
7. Optionally keep legacy flat selected/report artifacts only for compatibility, not as SSOT.

### Load Path

1. `load_or_build_universe_snapshot` computes current hashes.
2. Try exact-hash `load_universe_store_run`.
3. If found, materialize `UniverseSnapshot`, `selected`, and `report`.
4. If not found, call `build_universe`.
5. Do not use stale legacy snapshot payloads when hash mismatch exists.

### Timeline Path

1. `_discover_symbols_via_universe` continues to call `load_or_build_universe_snapshot`.
2. `discover_universe_timeline` uses `selected_frame["symbol"]` where `stage6_selected` is true.
3. `inference_panel_quarter_membership` is built from Stage 6 membership only.
4. Audit parquet generation may read from `decisions` rather than expanding `snapshot.rejected`.

### Candidate ML Metadata Path

1. `materialize_snapshot_from_store` populates `SymbolMeta.cluster_size` and
   `SymbolMeta.anchor_cluster_member` from decision rows.
2. `_universe_metadata_by_symbol` in `opt_main_futures.py` reads these fields directly.
3. It must not recompute `cluster_size` from the selected set.
4. It must not replace `anchor_cluster_member` with `role == "anchor"`.

## Surgical Plan

### `src/domain/futures/universe/models.py`

[ACTION: REPLACE]

- Extend `SymbolMeta` with:

```python
cluster_size: float = 1.0
anchor_cluster_member: float = 0.0
```

[ACTION: ADD]

- Add `UniverseRunManifest` dataclass as defined above.

### `src/domain/futures/universe/store.py`

[ACTION: ADD]

- Add constants:

```python
DEFAULT_UNIVERSE_STORE_ROOT = LOG_DIR / "futures/universe/store/v1"
UNIVERSE_DECISION_COLUMNS = (...)
```

- Add `compute_universe_run_id`, `write_universe_store_run`, `load_universe_store_run`,
  `build_decision_frame`, and `materialize_snapshot_from_store`.

Implementation constraints:

- Use `pd.to_parquet` and `pd.read_parquet`.
- Validate required columns before writing.
- Write to deterministic run directory.
- Do not overwrite another run with different hash under the same `run_id`.
- `materialize_snapshot_from_store` must set:
  - `training_panel = tuple(sorted(stage6 symbols))`
  - `live_inference_panel = tuple(sorted(stage6 symbols))`
  - `stage5_research_panel = tuple(sorted(stage5 symbols))`
  - `inference_panel = ()` for single-quarter materialization
  - `historical_trading_panel = ()` for single-quarter materialization

### `src/domain/futures/universe/storage.py`

[ACTION: REPLACE]

- Update `_symbol_meta_to_dict` and `_symbol_meta_from_dict` to preserve
  `cluster_size` and `anchor_cluster_member`.

[ACTION: KEEP]

- Keep `snapshot_to_payload`, `snapshot_from_payload`, `save_snapshot_json`,
  `load_snapshot_json`, `save_snapshot_parquet`, and `load_snapshot_parquet` as compatibility
  utilities.

### `src/domain/futures/universe/pipeline.py`

[ACTION: REPLACE]

- `_to_symbol_meta` must read:

```python
cluster_size=float(row.get("cluster_size", 1.0)),
anchor_cluster_member=float(row.get("anchor_cluster_member", 0.0)),
```

[ACTION: REPLACE]

- `_save_snapshot` should write the new store as primary persistence.
- Legacy flat parquet/json writes may remain behind the same function for compatibility.

[ACTION: REPLACE]

- `build_universe` should build `UniverseRunManifest` and `decisions`, then materialize return
  values from the store DTO path so build and load produce identical contracts.

[ACTION: REPLACE]

- `load_or_build_universe_snapshot` should prefer exact-hash store load and rebuild on miss.
- Legacy snapshot fallback is allowed only when all of these are true:
  - `schema_version` is current;
  - `config_hash` and `data_manifest_hash` match;
  - selected metadata contains `cluster_size` and `anchor_cluster_member`;
  - `inference_panel` is empty or Stage 6 equivalent.

### `src/domain/futures/universe/__init__.py`

[ACTION: ADD]

- Export `UniverseRunManifest`, `compute_universe_run_id`, `load_universe_store_run`,
  `write_universe_store_run`, and `materialize_snapshot_from_store`.

### `src/application/futures/optimization/universe_service.py`

[ACTION: REPLACE]

- Keep public timeline signatures.
- If audit generation needs per-symbol rows, source them from normalized decisions where available.
- Ensure timeline membership remains Stage 6 only.

### `src/execution/opt_main_futures.py`

[ACTION: REPLACE]

- `_universe_metadata_by_symbol` must use persisted snapshot metadata:

```python
metadata[symbol] = (
    float(meta.cluster_id),
    float(meta.beta_vs_market),
    float(meta.cluster_size),
    float(meta.anchor_cluster_member),
)
```

## Verification

Run L1 on modified source and tests:

```bash
uv run ruff check --fix src/domain/futures/universe/models.py src/domain/futures/universe/store.py src/domain/futures/universe/storage.py src/domain/futures/universe/pipeline.py src/domain/futures/universe/__init__.py src/application/futures/optimization/universe_service.py src/execution/opt_main_futures.py tests/unit/domain/futures/universe/test_store.py tests/unit/domain/futures/universe/test_storage.py tests/unit/domain/futures/universe/test_quarterly_selection_audit.py tests/unit/application/futures/optimization/test_universe_service.py tests/unit/execution/test_opt_main_futures_strategy_mode.py
uv run mypy src/domain/futures/universe/models.py src/domain/futures/universe/store.py src/domain/futures/universe/storage.py src/domain/futures/universe/pipeline.py src/domain/futures/universe/__init__.py src/application/futures/optimization/universe_service.py src/execution/opt_main_futures.py tests/unit/domain/futures/universe/test_store.py tests/unit/domain/futures/universe/test_storage.py tests/unit/domain/futures/universe/test_quarterly_selection_audit.py tests/unit/application/futures/optimization/test_universe_service.py tests/unit/execution/test_opt_main_futures_strategy_mode.py
```

Run focused tests:

```bash
uv run pytest tests/unit/domain/futures/universe/test_store.py tests/unit/domain/futures/universe/test_storage.py tests/unit/domain/futures/universe/test_quarterly_selection_audit.py tests/unit/application/futures/optimization/test_universe_service.py tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
```

Expected outcome:

- Store roundtrip preserves Stage 6 selected rows, Stage 5 research rows, and reject audit rows.
- Store cache hit requires exact config and manifest hashes.
- `UniverseSnapshot` materialized from the store preserves `cluster_size` and
  `anchor_cluster_member`.
- `discover_universe_timeline` produces Stage 6 membership windows only.
- `opt_main_futures._universe_metadata_by_symbol` does not recompute cluster metadata.

Optional smoke:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python src/execution/opt_main_futures.py --phase strategy --timeframe 4h --date 2026-05-01 --trials 1 --sync skip
```

Expected smoke outcome, excluding external Redis/Optuna storage availability:

- Universe log reports Stage 6 scoped production panels.
- Candidate ML bridge receives Stage 6 symbols only.
- No legacy Stage 5 `inference_panel` enters production ML scope.

## Acceptance Criteria

- [ ] `UniverseSnapshot` is no longer the persistent SSOT; it is materialized from normalized
  store artifacts.
- [ ] Exact hash cache lookup prevents stale config or stale data manifest reuse.
- [ ] `cluster_size` and `anchor_cluster_member` are preserved from selection output into ML
  features without semantic recomputation.
- [ ] Stage 5 survivor membership remains available for audit/research but cannot contaminate
  production candidate ML panels.
- [ ] Existing public APIs continue returning `tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]`
  where callers expect them.

## Risk Notes

- This is a medium-to-large refactor because storage, timeline, and ML metadata contracts touch
  multiple layers.
- Backward compatibility must be explicit. Legacy snapshot files should be treated as migration
  fallback, not cache authority.
- Multi-run partitions can accumulate storage. Add cleanup later if disk usage becomes material;
  do not add cleanup to this refactor.
- The store must not deduplicate by `as_of` alone. Hash-scoped run IDs are required to preserve
  replayability under config changes.
