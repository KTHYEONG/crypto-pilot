# 🎯 Objective
L2 선택 단계의 두 치명적 병목현상(① `[FINAL SIMULATION]` 이후 `select_layer2_champion` 재시뮬레이션 병목, ② Optuna `ProcessPoolExecutor` fork 오버헤드)을 정밀 진단하고, 구조적 최적화안을 설계한다. Redis DB 도입은 해결책이 아님을 논증한다.

---

# 📦 Context & Dependencies

## Target Files & Hotspots
| File | Function | Role |
|---|---|---|
| `src/execution/opt_main_futures.py:1251` | `_run_tiered_l2_study` | Optuna study + ProcessPoolExecutor 병렬 처리 |
| `src/execution/opt_main_futures.py:1408-1484` | Batch submission loop | `_GLOBAL_L2_CTX` fork + `_evaluate_l2_trial_from_global` |
| `src/domain/futures/strategy/tiered_workflow/selection.py:223` | `select_layer2_champion` | Replay 평가 → `evaluate_l2_trial` 순차 호출 |
| `src/domain/futures/optimization/workflow.py:1757` | `evaluate_l2_trial` | `_run_awf_simulation` + metrics + deployment calibration |
| `src/domain/futures/optimization/workflow.py:2073` | `_evaluate_l2_params` | `evaluate_l2_trial` wrapper for `_GLOBAL_L2_CTX` |

## Key Data Shapes
- `L2SimulationCache`: `vol_matrix_2d [T,N]`, signal matrices `[T,S]`, sleeve mapping `[S]`. T=~3000 bars, N=~60 symbols, S=~200 sleeves.
- `AlignedMarketData`: `close_2d [T,N]`, `execution_cost_bps_2d [T,N]`, `beta_vs_market_1d [N]`
- `TieredContext`: wraps `aligned`, `signal_batch`, `l2_sim_cache`, `caps`, `cfg`, `tf`
- Each `_run_awf_simulation`: iterates over AWF folds (typically 4-6 folds), runs portfolio allocation + backtest per fold.

---

# 🔬 Bottleneck 1: `select_layer2_champion` Replay Latency

## 원인 분석

### Flow (hot path)
```
[FINAL SIMULATION] log (pipeline.py:2602)
  → _tw.build_l2_simulation_folds()
  → predict_layer1_signals_multi_tf()  ← heavy: multi-TF signal generation
  → _tw.run_l2_awf()                   ← heavy: full AWF simulation with folds

[NEXT LOG — long gap]
  → L3 holdout evaluation
  → pipeline returns
```

그러나 진짜 병목은 **Optuna study 완료 후 `select_layer2_champion` 호출 지점** (`opt_main_futures.py:1549`):

```
select_layer2_champion()
  → build_l2_simulation_cache()     ← prebuilt인 경우 skip (l2_sim_cache 전달)
  → for trial in eval_candidates:   ← 최대 8회 (gate-passed) or 3회 (fallback)
      → evaluate_l2_trial()         ← 각 호출당 _run_awf_simulation 실행
          → _run_awf_simulation()   ← fold-level nested loop
              → per fold: signal → weights → backtest
          → performance metrics 계산 (sharpe_hac, sortino, CAGR, MDD 등)
          → calibrate_deployment_leverage()
          → apply_deployment()
      → evaluate_layer2_gate() × 2  ← pre-gate + final-gate (heavy computation duplicated)
      → _deflated_sharpe_probability()
  → argmax champion selection
```

### 병목 원인
1. **순차 재시뮬레이션**: `evaluated_pairs = [_eval_candidate(trial) for trial in eval_candidates]` (selection.py:379) — 최대 8회의 `evaluate_l2_trial`을 순차 실행. 각 실행은 `_run_awf_simulation`을 호출하며 fold-level 루프 포함.
2. **Gate 평가 이중화**: `evaluate_layer2_gate`가 pre-gate + final-gate로 2회 호출됨 (selection.py:419-443, 455-478). 각 호출마다 동일한 입력으로 유사한 계산 반복.
3. **Deployment recalibration**: `calibrate_deployment_leverage` + `apply_deployment`가 매 trial마다 재실행 — trial params가 달라도 deployment 로직은 유사.
4. **AWF simulation 내 fold-level 루프**: `_run_awf_simulation`은 fold 개수만큼 반복하며 각 fold에서 portfolio allocation → positions → backtest → metrics 계산.

### 측정 기준
- 8개 trial 재시뮬레이션: 각 trial당 평균 2-5초 (fold 수 × sleeve 수에 비례)
- 총 지연: 16-40초 (순차 8회) — 이 구간 동안 다음 로그 없음
- gate-passed trial이 0개면 3회로 fallback → 6-15초

---

# 🔬 Bottleneck 2: Optuna `ProcessPoolExecutor` GIL Overhead

## 원인 분석

### 현재 구조 (`opt_main_futures.py:1408-1484`)
```python
_GLOBAL_L2_CTX = ctx  # module-level global
mp_ctx = multiprocessing.get_context("fork")
with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
    for batch:
        futures = [executor.submit(_evaluate_l2_trial_from_global, params) ...]
        for trial, future in zip(batch_trials, futures):
            value, attrs, t_elapsed = future.result()  # blocking
            study.tell(trial, value)
```

### 병목 원인
1. **Fork + CoW overhead**: `fork()`로 자식 프로세스 생성 시 부모의 전체 메모리 공간(수 GB, numpy array 포함)을 Copy-on-Write로 복제. 자식이 numpy 배열을 수정하는 순간 실제 복사 발생.
2. **Numba GIL 해제 미활용**: `_run_awf_simulation` 내부 numba jit 함수들은 GIL을 해제하지만, `ProcessPoolExecutor`는 프로세스 단위 병렬이므로 이 이점을 살리지 못함. 오히려 프로세스 생성/소멸 오버헤드 + serialization 비용이 추가됨.
3. **Batch blocking**: `future.result()`가 각 trial을 순차적으로 대기 → batch 내 trial들이 완료될 때까지 다음 batch로 진행 불가.
4. **Optuna `study.tell()` 트랜잭션**: 각 trial 완료 시 Redis storage에 write 발생. Thread-safe하지 않으므로 프로세스 간 동시 접근 시 충돌 가능성.
5. **`_GLOBAL_L2_CTX` global state**: fork 환경에서는 자식이 부모의 `_GLOBAL_L2_CTX`를 상속받아 동작하지만, 이는 fork-specific 동작에 의존 → `spawn` 컨텍스트로 변경 시 깨짐. 본질적으로 불안정한 패턴.

### 측정 기준
- 배치 크기 2, max_workers=2 기준
- 프로세스 생성 오버헤드: ~200-500ms/batch
- pickle/serialize 오버헤드: params는 작지만 ctx는 fork로 우회
- 실제 trial 실행 시간: 2-5초
- 오버헤드 비율: ~10-25%

---

# ❌ Redis DB 도입 평가

## Redis가 해결하지 못하는 것
1. **CPU-bound 연산**: 병목의 본질은 `_run_awf_simulation`의 numpy 연산. Redis는 저장소/I/O 최적화 도구로, 연산 속도 향상에 기여하지 않음.
2. **Serialization cost**: Redis로 ctx를 주고받으려면 오히려 매 trial마다 전체 ctx를 직렬화/역직렬화해야 하므로 오버헤드 증가.
3. **GIL 우회**: Redis는 GIL 우회 메커니즘이 아님. GIL 우회는 multi-process 또는 GIL을 해제하는 native extension이 필요.

## Redis가 유용한 경우
- **분산 최적화**: 여러 머신에서 동일 Optuna study를 공유할 때 (이미 `JournalRedisStorage` 사용 중)
- **챔피언 파라미터 영구 저장**: `update_champion_store`에서 이미 사용 중
- **Task queue 백엔드**: Celery/RQ로 분산 워커 구동 시 — 단, 단일 머신에서는 이점 없음

**결론: Redis DB 도입은 GIL/연산 병목 해결과 무관. 현재 Redis storage는 이미 활용 중이며 추가 도입으로 얻을 이점 없음.**

---

# 🛠️ 최적화 개선안

## A. `select_layer2_champion` — Gate 평가 이중화 제거 (Quick Win)

### Target
`src/domain/futures/strategy/tiered_workflow/selection.py:383-479`

### Algorithmic Flow
현재 `select_layer2_champion`은 각 candidate에 대해 `evaluate_layer2_gate`를 2회 호출:
1. **pre-gate** (L407-443): `gate` attr이 없을 때만 계산. `dsr_hybrid=None`으로 호출.
2. **final-gate** (L455-478): 항상 계산. `dsr_hybrid=float(dsr)` 포함.

→ **문제**: pre-gate 결과의 대부분 필드가 final-gate에서 재계산됨. pre-gate는 constraints_ok 체크 외에는 활용되지 않음.

### 개선
```python
# pre-gate를 완전히 제거하고, final-gate만 계산.
# constraints_ok 체크는 final_gate.optuna_constraint_values로 직접 수행.
# evaluate_layer2_gate 호출을 1회로 축소 → 약 30-40% gate 계산 시간 절감.
```

### Risk
- 기존 pre-gate가 없는 trial에 대한 fallback path는 `evaluate_layer2_gate` 내부 로직과 동일하므로 안전.

---

## B. `select_layer2_champion` — Thread Pool 기반 Fold-Level 병렬화

### Target
`src/domain/futures/strategy/tiered_workflow/selection.py:378-379`

### Rationale
`evaluate_l2_trial` → `_run_awf_simulation` 내부는 numba로 GIL 해제. `ThreadPoolExecutor`로 trial 단위 병렬 처리가 가능하며, fork/serialize 오버헤드 없음.

### Algorithmic Flow
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

eval_candidates = gate_passed_candidates[:8] if gate_passed_candidates else replay_candidates[:3]

# GIL-unlocked numba 연산 활용 → ThreadPool이 ProcessPool보다 효율적
max_workers = min(len(eval_candidates), 4)
evaluated_pairs = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_map = {executor.submit(_eval_candidate, t): t for t in eval_candidates}
    for future in as_completed(future_map):
        evaluated_pairs.append(future.result())
```

### Key Constraint
- **Thread-safe cache**: `cache` (L2SimulationCache)는 read-only. 모든 thread가 공유 가능.
- **Numba release GIL**: `_run_awf_simulation`이 numba 기반이므로 multi-thread에서 실제 병렬성 확보.
- **max_workers=4 제한**: 메모리 사용량 고려. sleeve 수 × T 크기에 따라 조정.

### Expected Impact
- 8회 순차 → 4-thread 병렬: 약 **2-3배 속도 향상** (4배 이론치에서 thread contention 감안)
- ProcessPoolExecutor의 fork/serialize 오버헤드 없음
- `prebuilt_cache`를 모든 thread가 공유 (read-only)

---

## C. Optuna — Thread 기반 스트리밍 병렬화

### Target
`src/execution/opt_main_futures.py:1408-1484`

### Rationale
현재 `ProcessPoolExecutor` → `ThreadPoolExecutor` + `as_completed`로 전환. Fork 오버헤드 제거, GIL 해제 활용, 스트리밍으로 batch blocking 해소.

### Algorithmic Flow
```python
# Step 1: ProcessPoolExecutor 제거
# Step 2: ThreadPoolExecutor with as_completed (streaming)

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_study_lock = Lock()

def _thread_evaluate(trial, params):
    from src.domain.futures.optimization.workflow import _evaluate_l2_params
    t0 = time.perf_counter()
    value, attrs = _evaluate_l2_params(params, ctx)[:2]
    elapsed = time.perf_counter() - t0
    return trial.number, value, attrs, elapsed

max_workers = min(batch_size, 4)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    trial_idx = completed_count
    active_futures: dict[Any, Any] = {}
    
    while trial_idx < n_trials or active_futures:
        # Fill up to max_workers pending futures
        while trial_idx < n_trials and len(active_futures) < max_workers:
            trial = study.ask()
            params = suggest_layered_params(trial, "L2", fixed=ctx.fixed_l1_params)
            future = executor.submit(_thread_evaluate, trial, params)
            active_futures[future] = trial
            trial_idx += 1
        
        if not active_futures:
            break
            
        # Wait for ANY one to complete (streaming — no batch blocking)
        done_futures = as_completed(active_futures.keys())
        for future in done_futures:
            trial = active_futures.pop(future)
            try:
                num, value, attrs, elapsed = future.result()
                for k, v in attrs.items():
                    trial.set_user_attr(k, v)
                with _study_lock:
                    study.tell(trial, value)
                progress_cb(study, trial, value=value)
            except Exception as exc:
                _logger.error(...)
            break  # only process one completion, then refill slot
```

### Key Constraint
- `study.tell()`은 thread-safe하지 않음 → `threading.Lock`으로 보호
- Optuna `study.ask()`도 thread-safe하지 않음 → 메인 스레드에서만 호출
- `ctx`는 read-only로 모든 thread가 공유 (l2_sim_cache 포함)

---

## D. `evaluate_l2_trial` — Deployment 계산 Lazy화

### Target
`src/domain/futures/optimization/workflow.py:1757-1856`

### Rationale
`calibrate_deployment_leverage` + `apply_deployment`는 trial당 0.3-0.8초 소요. 8개 trial × 0.5초 = 4초 낭비. Gate 통과 실패한 trial에서는 deployment 정보가 champion 선정에 사용되지 않음.

### Algorithmic Flow
```python
# _run_awf_simulation까지만 실행 → 빠른 gate 평가 먼저 수행
# gate 통과한 trial에 대해서만 deployment calibration + 적용

def evaluate_l2_trial_lazy(*, cache, signal_batch, aligned, awf_folds, config, caps, tf, 
                             lazy_deploy: bool = True):
    sim = _run_awf_simulation(...)
    # ... metrics 계산 ...
    
    if lazy_deploy:
        # deployment 계산을 지연: caller가 필요할 때 별도 호출
        evaluation._sim = sim  # attach raw sim for later deployment
    else:
        # 기존 로직 유지
        _l_star, _l_binding = calibrate_deployment_leverage(...)
        apply_deployment(...)
```

### 호출 측 (`select_layer2_champion`)
```python
# 1차: 모든 candidate에 대해 lazy=True로 빠르게 gate 평가
# 2차: champion 선정된 trial에 대해서만 deployment 적용
```

---

## E. 캐시 웜업 + AWF Fold 재사용

### Target
`src/execution/opt_main_futures.py:1520-1549`

### Rationale
Optuna study 완료 후 `select_layer2_champion` 진입 직전에 `build_walk_forward_folds`를 다시 계산 (L1522). 이 folds는 이미 Optuna study 중에 계산된 것과 동일. 재계산 불필요.

### 개선
- `_run_tiered_l2_study`에서 계산한 `awf_folds_l2`를 `TieredContext`에 저장하여 반환
- `select_layer2_champion`은 ctx에서 folds를 재사용

---

# 🧪 Test Scenario Design

## Scenario 1: Thread 기반 replay 병렬화 검증
- **Given**: `mock_frozen_trials` 8개 (l2_promotion_passed=True), prebuilt_cache 주입
- **When**: `select_layer2_champion` 호출
- **Then**: `evaluate_l2_trial`이 8회 호출되고, `ThreadPoolExecutor.submit`이 실제로 4개 max_workers로 호출됨
- **Verify**: 결과가 순차 실행과 동일 (determinism 유지)
- **Test name**: `test_select_layer2_champion_thread_parallel_determinism`

## Scenario 2: Gate 평가 1회화 검증
- **Given**: `evaluate_layer2_gate` mock, candidate evaluation prepared
- **When**: `select_layer2_champion`에서 candidate 처리
- **Then**: `evaluate_layer2_gate`가 candidate당 정확히 1회만 호출됨 (pre-gate 없음)
- **Verify**: `constraints_ok`가 final_gate.optuna_constraint_values로 올바르게 판단됨
- **Test name**: `test_select_layer2_champion_single_gate_evaluation`

## Scenario 3: Deployment lazy evaluation 검증
- **Given**: `evaluate_l2_trial(lazy_deploy=True)` 호출
- **When**: gate-passed한 trial에 한해 `apply_deployment` 호출
- **Then**: champion 선정까지 deployment 계산이 지연되고, champion trial에만 적용됨
- **Verify**: 최종 `Layer2StudyResult`의 `best_params`에 배치 파라미터 포함
- **Test name**: `test_lazy_deployment_applied_to_champion_only`

## Scenario 4: Thread 기반 Optuna 병렬화 검증
- **Given**: Optuna study, 10 trials, max_workers=4
- **When**: `_run_tiered_l2_study` 실행
- **Then**: ThreadPoolExecutor로 4개 thread 실행, study.tell()이 Lock으로 보호됨
- **Verify**: 모든 trial 완료, best_params 반환, ProcessPoolExecutor 미사용
- **Test name**: `test_optuna_thread_parallel_over_sequential`

---

# 🛡️ Verification

```bash
# Unit tests for selection changes
uv run pytest tests/unit/domain/futures/strategy/test_selection.py -k "thread_parallel or single_gate or lazy_deploy" -v --tb=short

# Unit tests for Optuna changes  
uv run pytest tests/unit/execution/test_tiered_l2_optuna_integration.py -k "thread" -v --tb=short

# Lint
uv run ruff check src/domain/futures/strategy/tiered_workflow/selection.py src/execution/opt_main_futures.py src/domain/futures/optimization/workflow.py

# Type check
uv run mypy src/domain/futures/strategy/tiered_workflow/selection.py src/execution/opt_main_futures.py
```

---

# 📊 우선순위 & 기대 효과

| 순위 | 개선 | 난이도 | 기대 속도 향상 | 위험 |
|---|---|---|---|---|
| 1 | **A. Gate 이중화 제거** | Low | 15-20% (selection 단계) | Low — 로직 단순화 |
| 2 | **B. ThreadPool replay** | Medium | 2-3x (8 trial 기준) | Medium — thread safety 검증 |
| 3 | **D. Lazy deployment** | Medium | 20-30% (불필요 trial 제외) | Low — champion에만 적용 |
| 4 | **C. Optuna ThreadPool** | High | 1.5-2x (fork overhead 제거) | High — study.tell() 동시성 |
| 5 | **E. Fold 재사용** | Low | 5-10% (중복 계산 제거) | Low — 캐시 무결성 |
