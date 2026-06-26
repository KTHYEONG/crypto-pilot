# 🎯 Objective
매 trial마다 2.88s(72%)를 소비하던 `compute_bucket_realized_edges`를 `L2SimulationCache`에 1회만 계산하여 모든 200 trials에서 재사용. per-trial 시간 6.1s → 3.2s로 단축.

---

# 📦 Context & Dependencies

## Target Files
| File | Function | Change |
|---|---|---|
| `dataclasses.py:685` | `L2SimulationCache` | `bucket_edges_by_fold` 필드 추가 (default `()`) |
| `opt_main_futures.py:1280-1320` | `_run_tiered_l2_study` | cache 빌드 후 bucket edges 계산 → cache에 주입 |
| `awf_sim.py:1565-1608` | `_run_awf_simulation` | bucket 재계산 블록을 cache 읽기로 대체 |

## Key Data Shapes
- `bucket_edges_by_fold`: `tuple[dict[...], ...]` — fold 개수와 동일 길이. 각 dict는 `{(regime_code, family, tf): edge_bps}` 매핑.
- `compute_bucket_realized_edges(cache, aligned, fit_start, fit_end, regime_code_1d, cost_bps, min_n, shrinkage)` → `dict[tuple[int, str, str], float]`
- `compute_market_regime_context(aligned).code_1d` → `NDArray[np.int8]` [T]

## Precondition (cacheable 증명)
`compute_bucket_realized_edges`의 모든 입력은 trial params에 **독립적**:
- `cache` — trial 불변 (L2SimulationCache)
- `aligned` — trial 불변 (AlignedMarketData)
- `fit_start/fit_end` — fold 경계 (ctx.awf_folds 기반)
- `regime_code_1d` — aligned만으로 계산, trial 불변
- `cost_bps/min_n/shrinkage` — config 상수 (L2_ALLOC_SPACE에서 suggest되나, bucket routing 용도로는 cfg 디폴트)

---

# ✍️ Contract Changes

## L2SimulationCache (dataclasses.py:685)
```
class L2SimulationCache:
    # ... existing fields ...
    sleeve_to_tf: tuple[str, ...]
    # 신규:
    bucket_edges_by_fold: tuple[dict[tuple[int, str, str], float], ...] = ()
```

---

# 🛠️ Surgical Implementation Plan

## Step 1: Add field to dataclass

### Target
`src/domain/futures/strategy/tiered_workflow/dataclasses.py:722`

### Anchor (existing)
```python
    sleeve_to_tf: tuple[str, ...]  # [S] each sleeve's native TF (from strategy_id suffix)
```

### Algorithmic Flow
1. Line 722 이후에 `bucket_edges_by_fold: tuple[dict[tuple[int, str, str], float], ...] = ()` 추가
2. 기존 `return L2SimulationCache(...)` 구문들(awf_sim.py 2곳)은 변경 불필요 (기본값 `()`로 자동)

---

## Step 2: Precompute + inject in _run_tiered_l2_study

### Target
`src/execution/opt_main_futures.py:1280-1320` — cache 빌드 + ctx 생성 직후

### Anchor (existing)
```python
    if l2_sim_cache is None:
        l2_sim_cache = build_l2_simulation_cache(aligned, signal_batch, tf)
```

### Algorithmic Flow
```
# ... after l2_sim_cache is ready and _awf_folds_l2 is computed ...

# Precompute bucket edges once (trial-param independent)
from src.domain.futures.strategy.market_regime import compute_market_regime_context
from src.domain.futures.strategy.tiered_workflow.l2_meta import compute_bucket_realized_edges

_regime_code = compute_market_regime_context(aligned=aligned).code_1d
_bucket_edges: list[dict] = []
for _fold in _awf_folds_l2:
    if _fold.fit_start < _fold.oos_start:
        _be = compute_bucket_realized_edges(
            cache=l2_sim_cache, aligned=aligned,
            fit_start=_fold.fit_start, fit_end=_fold.oos_start,
            regime_code_1d=_regime_code,
            cost_bps=6.0, min_n=15, shrinkage=0.3,
        )
    else:
        _be = {}
    _bucket_edges.append(_be)
l2_sim_cache.bucket_edges_by_fold = tuple(_bucket_edges)

# ... then create ctx as before ...
```

**Key**: per-fold loop 내부에서 `compute_bucket_realized_edges` 호출. 비용: 3 folds × 0.9s = **2.7s 1회**.

---

## Step 3: replace recomputation with cache read in _run_awf_simulation

### Target
`src/domain/futures/strategy/tiered_workflow/awf_sim.py:1568-1608`

### Anchor (existing)
```python
    # L2 bucket routing: regime code + fit-leg 버킷 실현엣지 (1회)
    _l2_routing_mode = str(getattr(config, "l2_routing_mode", "bucket"))
    _regime_code_1d: NDArray[np.int8] = np.zeros(aligned.close_2d.shape[0], dtype=np.int8)
    bucket_edges_by_fold: list[dict[tuple[int, str, str], float]] = []
    if _l2_routing_mode == "bucket":
```

### Algorithmic Flow
```
1. cache.bucket_edges_by_fold가 비어있지 않으면 → 그대로 사용
2. 비어있으면(하위호환 fallback) → 기존 로직으로 compute_bucket_realized_edges 호출

Pseudo-code:
    _l2_routing_mode = str(getattr(config, "l2_routing_mode", "bucket"))
    if _l2_routing_mode == "bucket":
        if cache.bucket_edges_by_fold:
            bucket_edges_by_fold = list(cache.bucket_edges_by_fold)
        else:
            # fallback: 기존 로직 (regime code + per-fold compute)
            _regime_code_1d = compute_market_regime_context(...).code_1d
            for _fold in awf_folds:
                bucket_edges_by_fold.append(
                    compute_bucket_realized_edges(...)
                )
    else:
        bucket_edges_by_fold = [{} for _ in awf_folds]
```

### Key Constraints
- `bucket_edges_by_fold` 길이는 `awf_folds` 길이와 동일해야 함 (불일치 시 fallback)
- per-fold fit_start/fit_end는 `cache.bucket_edges_by_fold`에 내장되지 않으며, `compute_bucket_realized_edges` 호출 시 직접 전달됨 → precompute 시점과 runtime 시점의 fold 경계 동일성 보장
- `_regime_code_1d`는 bucket routing 이후 Step H (regime distribution check)와 fold 루프 내부에서도 사용됨 → **제거 불가, 변수 선언은 유지하되 조건부 계산으로 변경**

---

# 🧪 Test Scenario Design

## Scenario 1: Cache hit — no recomputation
- **Given**: `L2SimulationCache`에 `bucket_edges_by_fold`가 3개 fold 분량 pre-populated
- **When**: `_run_awf_simulation` 호출
- **Then**: `compute_bucket_realized_edges`가 호출되지 않음, `bucket_edges_by_fold` 리스트가 그대로 사용됨
- **Mock path**: `src.domain.futures.strategy.tiered_workflow.l2_meta.compute_bucket_realized_edges`
- **Test name**: `test_awf_sim_uses_cached_bucket_edges`

## Scenario 2: Cache miss fallback
- **Given**: `L2SimulationCache`에 `bucket_edges_by_fold`가 `()` (빈 tuple)
- **When**: `_run_awf_simulation` 호출
- **Then**: 기존 fallback 로직으로 `compute_bucket_realized_edges` 호출, 결과가 `bucket_edges_by_fold`에 저장됨
- **Test name**: `test_awf_sim_fallback_on_empty_cache`

## Scenario 3: Precompute in study
- **Given**: `_run_tiered_l2_study` 실행, mock `compute_bucket_realized_edges`
- **When**: cache 빌드 후
- **Then**: `compute_bucket_realized_edges`가 fold 수만큼 호출, 결과가 `l2_sim_cache.bucket_edges_by_fold`에 저장됨
- **Test name**: `test_l2_study_precomputes_bucket_edges`

---

# 🛡️ Verification

```bash
uv run ruff check --fix src/domain/futures/strategy/tiered_workflow/dataclasses.py src/domain/futures/strategy/tiered_workflow/awf_sim.py src/execution/opt_main_futures.py
uv run mypy src/domain/futures/strategy/tiered_workflow/dataclasses.py src/domain/futures/strategy/tiered_workflow/awf_sim.py src/execution/opt_main_futures.py
uv run pytest tests/unit/domain/futures/strategy/tiered_workflow/ -k "bucket or awf_sim" -v --tb=short
```

---

# 📊 Expected Impact

| metric | before | after |
|---|---|---|
| per-trial `bucket` 시간 | 2.88s | **~0.01s** (tuple → list 변환) |
| per-trial 총 시간 | 6.1s | **~3.2s** |
| 200 trials 소요 시간 (workers=6) | ~3.5분 | **~1.8분** |
| 1회 precompute 비용 | — | 2.7s |
| net 절감 | — | **200 × 2.87s - 2.7s ≈ 570s (9.5분)** |
