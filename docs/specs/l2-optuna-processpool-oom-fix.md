# 🎯 Objective
ThreadPool `as_completed` streaming이 유발한 Optuna 200-trials 속도 저하를 해결하기 위해, ProcessPool(fork) with OOM-safe batch processing으로 롤백하고, ctx 이중생성 및 불필요한 import 잔재를 정리한다.

---

# 📦 Context & Dependencies

## Target Files
| File | Function | Change |
|---|---|---|
| `src/execution/opt_main_futures.py:1251` | `_run_tiered_l2_study` | ThreadPool → ProcessPool(fork) 복원, OOM guard 추가, ctx 생성 2회→1회 |
| `src/execution/opt_main_futures.py:1267-1273` | imports | `_evaluate_l2_params_threadsafe` 제거, `_evaluate_l2_params` 추가 |
| `src/execution/opt_main_futures.py:1287-1326` | ctx creation | 1차 ctx 제거, 2차 ctx에 awf_folds 통합 |
| `src/execution/opt_main_futures.py:1430-1495` | parallel block | streaming ThreadPool → batch ProcessPool(fork) |
| `src/execution/opt_main_futures.py:109` | global | `_GLOBAL_L2_CTX` 복원 (`TieredContext | None`) |
| `src/domain/futures/optimization/workflow.py:2154` | `_evaluate_l2_params_threadsafe` | 유지 (제거 불필요, 향후 선택적 활용 가능) |

## Key Data Shapes
- `TieredContext`: `aligned`, `l2_sim_cache` (read-only in subprocess), `awf_folds` (pre-computed), `window`, `caps`, `cfg`, `tf`, `fixed_l1_params`
- `_GLOBAL_L2_CTX`: fork로 상속, CoW로 numba array 공유. 각 subprocess는 독립된 ctx 객체 참조.
- `_evaluate_l2_params(l2_params, ctx)`: ctx의 `l2_sim_cache`, `awf_folds`, `aligned`를 read-only로 사용.

## Memory Profile (OOM Guard 기준)
- Parent RSS: ~3-6 GB (l2_sim_cache: [T,N] + [T,S] matrices, aligned: close_2d[T,N])
- Per-subprocess additional RSS: ~0.5-1.5 GB (awf_simulation allocation)
- Formula: `mem_safe_workers = max(1, int((avail_gb - 2.0) / 1.5))`

---

# 🛠️ Surgical Implementation Plan

## Step 1: Clean up imports — `_run_tiered_l2_study` top

### Target
`src/execution/opt_main_futures.py:1267-1273`

### Anchor (existing)
```python
    from src.domain.futures.optimization.workflow import (
        TieredContext,
        _evaluate_l2_params_threadsafe,
        layer2_constraints_from_trial,
        objective_l2_growth,
        suggest_layered_params,
    )
```

### Algorithmic Flow
1. `_evaluate_l2_params_threadsafe` 제거
2. `_evaluate_l2_params` 추가
3. 최종: `TieredContext, _evaluate_l2_params, layer2_constraints_from_trial, objective_l2_growth, suggest_layered_params`

---

## Step 2: ctx 단일 생성 — remove double creation

### Target
`src/execution/opt_main_futures.py:1287-1326`

### Anchor (existing)
```python
    ctx = TieredContext(
        labeled_events=pd.DataFrame(),  # ...
        aligned=aligned,
        ...
        l2_sim_cache=l2_sim_cache,
    )

    # AWF folds pre-computation: ctx에 저장하여 ...
    _ho_ts = pd.Timestamp(window.holdout_start)...
```

### Algorithmic Flow
1. 첫 번째 ctx 생성(L1287-1296) 제거
2. folds pre-compute 후 ctx 생성 → `awf_folds=_awf_folds_l2` 포함된 ctx를 유일 ctx로
3. 최종: folds pre-compute → 단일 `ctx = TieredContext(... awf_folds=_awf_folds_l2)`

---

## Step 3: _GLOBAL_L2_CTX 복원

### Target
`src/execution/opt_main_futures.py:109` (module-level)

### Anchor (current)
```python
_REGIME_NAMES_SHORT: dict[int, str] = ...
```

### Algorithmic Flow
1. `_REGIME_NAMES_SHORT` 위에 `_GLOBAL_L2_CTX: TieredContext | None = None` 삽입
2. TYPE_CHECKING 블록에 `TieredContext` import 필요 → 확인 후 필요시 추가

---

## Step 4: ProcessPool(fork) batch processing 복원

### Target
`src/execution/opt_main_futures.py:1438-1495` (parallel block 전체)

### Anchor (existing — block to replace)
```python
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from threading import Lock

                from src.domain.futures.optimization.workflow import (
                    _evaluate_l2_params_threadsafe,
                )

                max_workers = min(batch_size, 4)
                ...  (entire ThreadPool streaming block)
```

### Algorithmic Flow
```
1. OOM guard 계산:
   avail_gb = psutil.virtual_memory().available / (1024**3)
   mem_safe = max(1, int((avail_gb - 2.0) / 1.5))
   cpu_cores = os.cpu_count() or 4
   max_workers = max(1, min(batch_size, cpu_cores, mem_safe))

2. Fork-safe eval function (로컬 정의):
   def _eval_in_subprocess(params, ctx):
       return _evaluate_l2_params(params, ctx)  # (value, attrs, elapsed)

3. _GLOBAL_L2_CTX = ctx  (module-level assign, fork 상속용)

4. try:
       mp_ctx = multiprocessing.get_context("fork")
       with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx):
           trial_idx = len(completed)
           while trial_idx < n_trials:
               current_batch = min(batch_size, n_trials - trial_idx)
               
               # study.ask() + suggest_layered_params()
               # executor.submit() batch
               # future.result() ordered wait
               # study.tell() and progress_cb()
               # trial_idx += 1
               
               gc.collect()
   finally:
       _GLOBAL_L2_CTX = None
```

### Key Constraints
- `_evaluate_l2_params` (기존, L2094-2097)의 `object.__setattr__(ctx, "l2_sim_cache", ...)` fallback path: l2_sim_cache가 항상 pre-built이므로 실행되지 않음. fork 환경에서 thread-safe.
- `ctx.awf_folds`: pre-computed → `_resolve_l2_signal_batch_and_folds`가 `build_walk_forward_folds` 재호출 안 함.
- `executor.submit(_evaluate_l2_trial_from_global, params)`: params만 pickle (작은 dict). ctx는 _GLOBAL_L2_CTX로 fork 상속 → serialization 없음.

---

## Step 5: _evaluate_l2_trial_from_global 함수 복원

### Target
`src/execution/opt_main_futures.py:113` (after `_btc_index_if_present`)

### Algorithmic Flow
```python
def _evaluate_l2_trial_from_global(params: dict[str, Any]) -> tuple[float, dict[str, Any], float]:
    if _GLOBAL_L2_CTX is None:
        raise ValueError("Global L2 context is not initialized")
    return _evaluate_l2_params(params, _GLOBAL_L2_CTX)
```

---

## Step 6: step 2 imports 내 _evaluate_l2_params_threadsafe 참조 정리

### Target
`src/execution/opt_main_futures.py:1442-1443` (removed — no longer needed in else block)

---

# 🔬 원인 분석 요약 (ThreadPool 실패 사유)

| 원인 | 설명 |
|---|---|
| **GIL 경합** | `evaluate_l2_trial`의 post-simulation Python 코드 (metric 계산, deployment calib, user_attrs 빌드, L2114-2151)에서 GIL 획득 경합 발생. numba 부분은 GIL 해제되나 후처리가 pure-Python → 실질 병렬도 <1.5x |
| **as_completed overhead** | 200 trials × `as_completed` 호출 → 매 호출마다 waiter 등록/해제 (condition lock acquisition × pending count). batch 방식 대비 ~200회 추가 overhead |
| **streaming refill delay** | 1 trial 완료 → break → refill → 다음 `as_completed` 호출까지 순차 지연. batch는 한 번에 2개 submit 후 동시 대기. |

---

# 📊 OOM Guard 공식

```
avail_gb = psutil.virtual_memory().available / (1024**3)
mem_safe = max(1, int(avail_gb / 1.2))
cpu_cores = os.cpu_count() or 4
max_workers = max(1, min(batch_size, cpu_cores, mem_safe))
```

- avail_gb < 1.2 → mem_safe = 1 → sequential fallback
- avail_gb = 4.0 → mem_safe = 3 → 적정 병렬 (OOM thrashing 회피)
- avail_gb ≥ 7.2 → mem_safe ≥ 6 → batch_size로 capped

---

# 🧪 Test Scenario Design

## Scenario 1: ProcessPool batch 동작 검증
- **Given**: `L2_OPTUNA_BATCH_SIZE=2`, mock study with 4 trials, mock `psutil.virtual_memory().available = 8GB`
- **When**: `_run_tiered_l2_study` 호출
- **Then**: `ProcessPoolExecutor(max_workers=2)` 생성, `executor.submit` 4회 호출, `study.tell` 4회 호출
- **Test name**: `test_l2_study_uses_process_pool_by_default`

## Scenario 2: OOM guard fallback to sequential
- **Given**: `psutil.virtual_memory().available = 2GB` (low memory), batch_size=2
- **When**: `_run_tiered_l2_study` 실행
- **Then**: batch_size → 1로 degrading, `study.optimize(n_jobs=1)` sequential path 사용, 경고 로그 출력
- **Test name**: `test_oom_guard_forces_sequential_on_low_memory`

## Scenario 3: ctx single creation 확인
- **Given**: `build_walk_forward_folds` mock → 호출 횟수 추적
- **When**: `_run_tiered_l2_study` 실행
- **Then**: `build_walk_forward_folds`가 정확히 1회만 호출 (기존 2회 → 1회)
- **Test name**: `test_ctx_created_once_with_folds`

## Scenario 4: _GLOBAL_L2_CTX cleanup
- **Given**: `_run_tiered_l2_study` 정상 완료
- **When**: 함수 반환 후
- **Then**: `_GLOBAL_L2_CTX is None` (finally block에서 정리됨)
- **Test name**: `test_global_l2_ctx_cleaned_after_study`

---

# 🛡️ Verification

```bash
uv run ruff check --fix src/execution/opt_main_futures.py
uv run mypy src/execution/opt_main_futures.py
uv run pytest tests/unit/execution/test_tiered_l2_optuna_integration.py -k "parallel or oom" -v --tb=short
```
