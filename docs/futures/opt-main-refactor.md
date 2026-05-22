# Futures Optimization Runner 리팩토링 설계도

**작성일**: 2026-05-22  
**대상**: `src/execution/opt_main_futures.py`  
**목표**: 실행 파일을 얇은 composition layer로 축소하고, 유니버스/데이터/전략/백테스트/최적화 책임을 테스트 가능한 모듈로 분리한다.

---

## 1. 현재 진단

`opt_main_futures.py`는 현재 1200라인 이상이며 아래 책임을 한 파일에서 동시에 수행한다.

| 책임 | 현재 위치 | 문제 |
|---|---|---|
| CLI pre-parse/main parse | `main()` 내부 | sync/universe/strategy/optuna 설정이 섞여 실행 모드별 계약을 검증하기 어렵다. |
| 유니버스 분기 타임라인 | `_discover_evolution_symbols()` | `docs/futures/universe.md`의 PIT snapshot 책임과 실행 orchestration 책임이 섞인다. |
| membership mask 생성 | `_inject_membership_masks_into_maps()` | 백테스트 입력 준비 책임인데 execution runner에 존재한다. |
| 데이터 충분성 검증 | `main()` + `opt_data_utils.py` | 데이터 readiness 결과가 runner 흐름에 직접 결합되어 재사용 테스트가 어렵다. |
| strategy smoke | `_run_strategy_smoke()` | 전략 검증 harness가 runner 내부 private 함수로 묶여 있다. |
| ML/legacy alpha gate | `main()` Step 2 | legacy alpha component filter 계약을 active strategy path와 같은 위치에서 처리한다. |
| Optuna contract/run summary | `main()` 후반부 | 최적화 실행과 observability persistence가 섞인다. |

가장 중요한 findings는 다음이다.

1. `No alpha components survived`는 신뢰 가능한 alpha 품질 판정이 아니다. non-strategy 경로에서 `strategy_runtime.bridge.run_ml_pipeline_for_universe()`가 `strategy_cfg=None`이면 빈 `MLPipelineOutput()`를 반환하고, runner의 G-ALPHA hard gate가 이를 `survive=0/post=0`으로 해석한다.
2. `ml_cfg["FUTURES_ML_ALPHA_BACKEND"] = "factory_v1"`는 현재 실행 경로에 실질적으로 반영되지 않는다. `src/domain/futures/ml_pipeline/__init__.py`는 active API를 `strategy_runtime.bridge`로 라우팅하므로 legacy/factory 설정값만으로 alpha backend가 연결되지 않는다.
3. `docs/futures/strategy.md`는 legacy alpha/HMM 없는 deterministic strategy 검증을 active 방향으로 정의한다. 따라서 legacy alpha mining/HMM full path는 active runner의 기본값이 아니라 명시적 legacy runner로 격리해야 한다.

---

## 2. 리팩토링 원칙

1. `src/execution/opt_main_futures.py`는 orchestration entrypoint만 담당한다.
2. active path는 `quick-backtest`와 `strategy` path만 지원한다.
3. legacy alpha/HMM/factory mining path는 `legacy/`로 이동하고, active runner에서 import하지 않는다.
4. 데이터 수집은 백테스트 요구 기간(`fetch_start`, `is_start`, `oos_start`, `end`)과 warmup/intrabar 요구를 readiness 계약으로 검증한 뒤 진행한다.
5. 유니버스 membership 변경은 백테스트 입력 준비 계층에서 생성하고, optimizer/backtest engine은 동일한 mask 배열만 소비한다.
6. 모든 단계는 독립 테스트 가능한 request/result dataclass 계약을 갖는다.

---

## 3. 목표 아키텍처

```text
src/execution/opt_main_futures.py
  -> parse CLI
  -> build FuturesRunConfig
  -> FuturesOptimizationRunner.run(config)
  -> process exit code

src/application/futures/optimization/
  config.py              # CLI/config dataclass, validation
  runner.py              # high-level use-case orchestration
  data_readiness.py      # required window, warmup, 1m coverage gate
  universe_service.py    # PIT quarterly timeline + quality gate
  strategy_service.py    # quick/strategy mode ML bridge invocation
  optimization_service.py# Optuna phase execution wrapper
  reporting.py           # run summary, p7, optuna contract, telemetry

src/domain/futures/
  universe/*             # PIT universe pure/domain logic
  strategy/*             # active deterministic strategies
  strategy_runtime/*     # MLPipelineOutput bridge only for active strategy
  backtest_preparation.py# membership/kill/entry masks + alignment contracts
  optimization/*         # optimizer/phase runner/evaluator
  validation/*           # promotion gates

src/domain/futures/legacy/
  ml_pipeline/*
  alpha_factory/*
  strategy_ml.py
  execution/opt_main_futures_legacy.py
```

`application` 계층은 domain 모듈을 조립하지만 domain 내부 계산을 직접 구현하지 않는다. 이 경계가 생기면 `opt_main_futures.py`는 100라인 이하의 thin entrypoint로 축소된다.

---

## 4. 책임 분리 상세

### 4.1 CLI 및 설정

신규 파일: `src/application/futures/optimization/config.py`

```python
@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    tf: str
    reference_date: str | None
    symbols: tuple[str, ...] | None
    trials: int
    mode: Literal["quick-backtest", "strategy", "strategy-smoke"]
    strategy: Literal["momentum_v0", "eh_st_v1"] | None
    sync_mode: Literal["full_history_master", "elite_fast"]
    skip_universe: bool
    skip_data_sync: bool
    force_universe_rebuild: bool
```

검증 규칙:

| 조건 | 처리 |
|---|---|
| `mode="strategy"`인데 `strategy is None` | fail-fast |
| `mode="quick-backtest"`와 `strategy` 동시 지정 | fail-fast |
| `mode="full"` | 제거. legacy runner에서만 허용 |
| `alpha_only`, `hmm_only` | active CLI에서 제거, legacy CLI로 이동 |

### 4.2 유니버스 서비스

신규 파일: `src/application/futures/optimization/universe_service.py`

책임:

- `get_quarterly_window(reference_date)` 결과를 기준으로 분기별 `UniverseSnapshot` 조회.
- `previous_selection`을 전달해 hysteresis/dwell 계약 보존.
- union symbols와 timeline을 반환.
- `validate_universe_quality()`를 이 파일로 이동.

계약:

```python
@dataclass(slots=True, frozen=True)
class UniverseTimelineResult:
    symbols: tuple[str, ...]
    timeline: Mapping[date, frozenset[str]]
    snapshot: UniverseSnapshot
    report: pd.DataFrame
```

로그:

- `[UNIVERSE-TIMELINE] quarter=... selected=... new=... dropped=... retained=...`
- `[UNIVERSE-QUALITY] median_cost_bps=... median_adv=... dropout_rate=... pass=...`

### 4.3 데이터 readiness

신규 파일: `src/application/futures/optimization/data_readiness.py`

책임:

- `sync_mode=full_history_master` 기본값 유지.
- 필요한 기간 전체 수집: `fetch_start <= first_dt`, `last_dt >= end`.
- warmup bars, IS/OOS bars, 1m execution coverage 검증.
- 리포트 저장과 per-symbol 상세 로그.

계약:

```python
@dataclass(slots=True, frozen=True)
class DataWindowContract:
    fetch_start: date
    is_start: date
    oos_start: date
    end: date
    tf: str
    warmup_bars: int
    require_exec_1m: bool
```

`opt_data_utils.filter_symbols_by_data_sufficiency()`는 이 계층으로 이동하거나 wrapper만 남긴다.

로그:

- `[DATA-SYNC] mode=full_history_master start=... end=... symbols=...`
- `[DATA-SUFFICIENCY] symbol=... required_is_bars=... actual_is_bars=... exec_1m_coverage=... pass=...`
- `[DATA-READY] total=... passed=... dropped=... report=...`

### 4.4 membership mask

현재 `_inject_membership_masks_into_maps()`는 runner에서 제거하고 `backtest_preparation.py` 또는 신규 `src/domain/futures/universe/membership.py`로 이동한다.

출력 컬럼:

| 컬럼 | 의미 |
|---|---|
| `universe_active_mask` | 해당 decision bar에서 유니버스 소속이면 1 |
| `universe_entry_warm_mask` | 신규 진입 후 warmup 완료 시 1 |
| `membership_kill_signal` | 퇴출 시점 강제 청산 signal |
| `entry_block_mask` | 비소속 또는 warmup 미완료 진입 차단 |
| `kill_signal` | 기존 kill과 membership kill의 max |

필수 규칙:

- 심볼 비교는 canonical symbol(`BTCUSDT`)로만 수행한다.
- 데이터 로더 입력 심볼 포맷은 변경하지 않는다.
- optimizer와 backtest engine은 동일 mask 배열을 소비한다.

테스트:

- `BTCUSDT` timeline과 `BTC/USDT` data key가 동일 membership으로 판정되는지 확인.
- 분기 퇴출 시 `membership_kill_signal`이 최초 비활성 bar에 1회 발생하는지 확인.
- 신규 진입 후 warmup 이전 `entry_block_mask=1`인지 확인.

### 4.5 strategy service

신규 파일: `src/application/futures/optimization/strategy_service.py`

책임:

- active strategy만 실행한다.
- `quick-backtest`는 명시적으로 neutral `MLPipelineOutput()`를 반환한다.
- `strategy`는 `StrategyConfig`를 만들어 `strategy_runtime.bridge.run_ml_pipeline_for_universe()`를 호출한다.
- legacy alpha/HMM gate는 호출하지 않는다.

active gate:

| 모드 | alpha 검증 |
|---|---|
| `quick-backtest` | empty alpha 허용 |
| `strategy` | `alpha_panel` non-empty, `alpha_long/alpha_short` 존재, merge 후 non-zero 검증 |
| `strategy-smoke` | 단일 backtest plumbing 검증 |

제거 대상:

- `G-ALPHA v8.0` hard kill
- `alpha_component_filter` 기반 survive/post count gate
- `alpha_only`, `hmm_only`
- `FUTURES_ML_ALPHA_BACKEND="factory_v1"` 강제 설정

legacy 경로에서만 유지:

- alpha component FDR/DSR/OOS gate
- GP miner/factory alpha
- HMM-only/alpha-only execution mode
- `No alpha components survived` 메시지

### 4.6 최적화 서비스

신규 파일: `src/application/futures/optimization/optimization_service.py`

책임:

- `MLPhaseDContext` 생성.
- `precompute_ml_optimization_context()`.
- `run_v43_phase_optimization_skeleton()`.
- candidate selection, champion guard, final OOS evaluation 호출.

Optuna contract는 `reporting.py`에서 생성하되, trials/worker/sampler/pruner 설정은 `FuturesRunConfig`와 `resolve_futures_parallel_policy()` 결과로 명시한다.

로그:

- `[OPTUNA-CONTRACT] total_trials=... phases=... workers=... storage=...`
- `[OPTIMIZATION] phase=... trials=... completed=... pruned=... failed=...`

---

## 5. legacy 이관 범위

아래 로직은 active runner에서 제거하고 `src/domain/futures/legacy/` 또는 `src/execution/legacy/`로 이동한다.

| 대상 | 이동 위치 | 이유 |
|---|---|---|
| `alpha_only`, `hmm_only` CLI | `src/execution/legacy/opt_main_futures_legacy.py` | `docs/futures/strategy.md`의 active strategy 방향과 충돌 |
| `FUTURES_ML_ALPHA_BACKEND="factory_v1"` | legacy runner | 현재 active bridge에서 소비하지 않는 설정 |
| `G-ALPHA v8.0` component survival gate | legacy alpha validation module | strategy mode의 `alpha_long/alpha_short` 계약과 무관 |
| `log_alpha_component_summary()` 의존 active gate | legacy reporting | component index 기반 리포트는 active deterministic strategy에 필요 없음 |
| `src/domain/futures/alpha_factory/*` shim | `src/domain/futures/legacy/alpha_factory/*`만 직접 사용 | 신규 active code import 금지 |
| `src/domain/futures/ml_pipeline/*` legacy shim | legacy namespace로 격리 | active bridge와 이름 충돌 방지 |

호환이 필요한 경우 `src/execution/legacy/opt_main_futures_legacy.py`에서만 legacy shim을 import한다. active package에서는 legacy import가 발생하면 테스트에서 실패해야 한다.

---

## 6. 단계별 마이그레이션 계획

### Phase 0: 현행 동작 고정

- 현재 `quick-backtest`, `strategy momentum_v0`, `strategy-smoke` smoke command를 기록한다.
- 기존 dirty 변경분 중 membership/data sufficiency 패치는 별도 커밋 단위로 고정한다.
- `No alpha components survived`는 `legacy_path_not_supported_in_active_runner`로 재분류한다.

### Phase 1: 얇은 runner 도입

- `FuturesRunConfig`, `parse_args()`, `FuturesOptimizationRunner` 추가.
- 기존 `opt_main_futures.py`는 새 runner를 호출하도록 축소.
- 기능 이동 없이 wrapper만 먼저 둔다.

### Phase 2: 유니버스/데이터/membership 이동

- `_discover_evolution_symbols()` -> `universe_service.py`
- `validate_universe_quality()` -> `universe_service.py`
- `_inject_membership_masks_into_maps()` -> domain membership/preparation 계층
- sufficiency gate -> `data_readiness.py`

### Phase 3: active strategy path 정리

- `mode`를 `quick-backtest | strategy | strategy-smoke`로 제한.
- `full`, `alpha_only`, `hmm_only` 제거.
- non-strategy full path는 legacy runner로만 실행 가능하게 변경.

### Phase 4: Optuna/reporting 분리

- Optuna study setup, phase budget, run summary persistence를 service/reporting으로 이동.
- `opt_main_futures.py`는 exit code와 최종 요약만 출력.

### Phase 5: legacy 격리 테스트

- active source에서 `src.domain.futures.legacy`, `alpha_factory`, legacy `ml_pipeline` import가 없는지 테스트.
- legacy runner는 별도 smoke만 유지하고 active CI에서 제외하거나 optional marker로 분리한다.

---

## 7. 테스트 설계

### 7.1 Unit tests

| 테스트 | 목적 |
|---|---|
| `test_run_config_rejects_legacy_modes` | active CLI에서 `alpha_only/hmm_only/full` 차단 |
| `test_universe_timeline_uses_previous_selection` | 분기 hysteresis 입력 전달 확인 |
| `test_membership_mask_symbol_canonicalization` | `BTCUSDT`와 `BTC/USDT` mismatch 방지 |
| `test_data_window_contract_requires_warmup_and_oos` | warmup/IS/OOS 부족 시 fail |
| `test_strategy_service_rejects_empty_strategy_alpha` | strategy mode에서 빈 alpha 차단 |
| `test_quick_backtest_allows_neutral_output` | quick mode에서는 빈 alpha 허용 |
| `test_active_runner_has_no_legacy_imports` | active path legacy import 금지 |

### 7.2 Smoke commands

```bash
uv run python src/execution/opt_main_futures.py \
  --quick-backtest \
  --skip-universe --skip-data-sync \
  --symbols BTCUSDT --trials 1 --tf 4h \
  --reference-date 2026-05-01
```

```bash
uv run python src/execution/opt_main_futures.py \
  --strategy momentum_v0 \
  --skip-data-sync \
  --trials 1 --tf 4h \
  --reference-date 2026-05-01 \
  --sync-mode full_history_master
```

```bash
uv run python src/execution/opt_main_futures.py \
  --mode strategy-smoke \
  --strategy momentum_v0 \
  --skip-data-sync \
  --trials 1 --tf 4h \
  --reference-date 2026-05-01
```

### 7.3 Acceptance criteria

- `opt_main_futures.py`는 100라인 이하.
- active runner에서 legacy import 0건.
- strategy mode에서 `alpha_panel`과 per-symbol `alpha_long/alpha_short` merge 검증 로그가 출력된다.
- data readiness report가 항상 생성된다.
- membership mask active/warm ratio가 0으로 붕괴하지 않는다.
- Step 3 Optuna 진입 실패 시 reason code가 `data_not_ready`, `strategy_alpha_empty`, `no_candidate`, `legacy_mode_disabled` 중 하나로 분류된다.

---

## 8. 문서 정합성 업데이트 필요

`docs/futures/strategy-code.md`는 일부 코드 사실이 현재와 다르다. 예를 들어 `merge_ml_output_into_data_maps()`가 no-op이라고 되어 있으나 현재는 merge 구현이 존재한다. 리팩토링 Phase 0에서 아래 문서를 갱신해야 한다.

| 문서 | 필요한 변경 |
|---|---|
| `docs/futures/strategy-code.md` | bridge 현재 상태, active mode, legacy 이관 정책 반영 |
| `docs/futures/backtest-logic.md` | `src/execution/opt_main_futures.py` 역할을 smoke entry에서 thin runner로 변경 |
| `docs/futures/universe.md` | membership mask가 runner가 아닌 preparation/domain 계층에서 생성됨을 반영 |

---

## 9. 최종 상태

리팩토링 완료 후 active 실행 구조는 다음처럼 단순해야 한다.

```text
CLI
  -> FuturesRunConfig
  -> UniverseTimelineResult
  -> DataReadinessResult
  -> Prepared membership masks
  -> StrategyOutput or NeutralOutput
  -> OptimizationResult
  -> RunReport
```

legacy alpha/HMM/factory path는 별도 runner에서만 접근 가능하다. active futures optimization은 `docs/futures/strategy.md`의 deterministic strategy 계약과 `docs/futures/backtest-engine.md`의 execution semantics를 기준으로 검증된다.
