# Binance Futures 백테스트 아키텍처 (v3.3 - Engine/Execution Focus)

**최종 업데이트**: 2026-05-22  
**핵심 목적**: 전략 품질과 분리된 백테스트 엔진의 정확성, 재현성, 실행 현실성 보장.

## 0. 2026-05-22 실행 계약 업데이트

- `src/execution/opt_main_futures.py`는 thin wrapper가 아니라 stage 오케스트레이션 엔트리포인트로 동작한다.
- 최적화 실행 계약은 `run_tracker.log_optuna_contract()`와 `collect_run_summary_from_study()`로 추적한다.
- run summary/contract에서 관리하는 핵심 항목:
  - `requested_trials_per_phase`
  - `planned_total_trials`
  - `trials_per_phase`
  - `completed_trials_per_phase`
  - `sampler_by_phase`
  - `worker_by_phase`
  - `storage_url`
- **백테스팅 데이터 로딩의 오프라인화**:
  - 병렬 백테스팅 스레드 가동 도중 동적으로 생성되는 API 요청에 의한 네트워크 포트 고갈을 해결하기 위해 백테스트 실행 단계(`load_single_symbol_data`)는 오직 디스크 캐시(Parquet)만을 100% 사용하여 기동되도록 강제함.
  - `DataCollector.collect_and_save(fetch_network=False)` 및 `collect_1m_ohlcv(fetch_network=False)` 호출을 강제하여, 캐시 누락 시 실시간 API 호출 시도를 엄격하게 스킵/차단함으로써 연산의 고속화와 소켓 자원의 극대화된 안정성을 동시에 확보함.

---

## 1. 핵심 아키텍처 및 데이터 흐름

백테스트는 전략 내부 학습 로직과 분리된 **계약 기반 파이프라인**으로 동작한다.

```
run_backtest_pipeline(config, snapshot, prepared_data) -> WalkForwardResult
```

* **결정론적 재현성**: 동일 `config` + 동일 `UniverseSnapshot` + 동일 입력 parquet에서 동일 결과를 재현한다.
* **룩어헤드 차단**: `knowledge_date > as_of` 데이터는 전처리 단계에서 배제한다.

### 4개 레이어 분리 구조

```
┌─────────────────────────────────────────────────────────────┐
│ [Layer A: Data Preparation]                                │
│ - 1h/4h/1m 정렬, funding 정합, kill mask 생성              │
│ - UniverseSnapshot과 시간축 동기화                         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [Layer B: Walk-Forward & Optimization]                     │
│ - Inner AWF(K=5, IS=24M), Atomic 6M block 평가             │
│ - Score 계산, Hard Gate 판정, DSR 통제                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [Layer C: Portfolio & Execution]                           │
│ - 0.25x Kelly, 5-cap projection, minNotional quantization  │
│ - Coarse(4h) + Intrabar(1m) 체결/펀딩/청산 시뮬레이션       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [Layer D: Promotion & Registry]                            │
│ - Atomic/OOS gate + Intrabar decay + AUM ladder 검증       │
│ - Champion 승격/보류 결정                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 디렉토리 구조 및 모듈 매핑

`src/domain/futures/` 기준 백테스트 엔진 책임 분리:

| 파일 | 역할 |
|---|---|
| `backtest_preparation.py` | 입력 패널 정렬(4h/1m/funding/kill) 및 실행 계약 검증 |
| `portfolio/execution_sim.py` | Numba 기반 coarse/intrabar 체결 시뮬레이터 |
| `portfolio/portfolio_constructor.py` | Kelly scaling, cap projection, 양자화 |
| `portfolio/friction_model.py` | Coarse pre-charge 비용 추정 |
| `portfolio/risk_controls.py` | Dual decay, drawdown overlay, no-trade buffer |
| `optimization/evaluator.py` | `compute_v3_score`, DSR 관련 통계 |
| `optimization/optimizer.py` | Inner AWF 탐색, trial 목적함수 계산 |
| `validation/walk_forward.py` | WalkForwardConfig, leg 집계, hard gate 연결 |
| `validation/atomic_blocks.py` | non-overlap 6M block 평가 |
| `validation/unified_gates.py` | `V3HardGates` 판정 |
| `validation/champion_registry.py` | sequential promotion gate, champion 비교 |
| `validation/boundary_contract.py` | `PurgeBarsRegistry` fail-fast 검증 |
| `src/execution/opt_main_futures.py` | 전체 orchestration 및 smoke entry |

---

## 3. 데이터 계약 및 정합성 규칙

### 3.1 시간 해상도 계약

* Decision grid: `4h` UTC closed bar.
* Execution grid: `1m` intrabar.
* Base feature grain: `1h`.

### 3.2 핵심 배열 계약

| 변수 | shape | 용도 |
|---|---|---|
| `close_2d` | `[B4h, N]` | 의사결정 종가 |
| `target_weights_2d` | `[B4h, N]` | 리밸런스 목표 비중 |
| `exec_o/h/l/c_1m` | `[B1m, N]` | intrabar 체결 경로 |
| `funding_event_mask_1m` | `[B1m, N]` | 8h funding 이벤트 |
| `funding_rate_1m` | `[B1m, N]` | 이벤트 funding rate |
| `kill_signal_2d` | `[B4h, N]` | 거래 금지/강제 제외 마스크 |
| `sigma_3d` | `[B4h, N, N]` | 공분산/리스크 추정 입력 |

### 3.3 결측/비정상 처리 계약

* 단일 결측 바는 직전 값 forward-fill 후 거래량 `0` 처리.
* 연속 결측 2개 이상 또는 강제 제외 이벤트는 `kill_signal=1`.
* NaN/Inf는 neutral weight 또는 entry skip로 처리하고, 계산 실패는 fail-fast로 기록한다.

### 3.4 Look-ahead 및 Purge 계약

* 체결은 항상 의사결정 바 다음 시점(`t -> t+1`)에서만 반영.
* `boundary_purge_bars` 공식:

```text
max(label_horizon, meta_label_horizon, stateful_fit_leakage, execution_delay)
```

* 모든 stateful 모듈은 `purge_bars` 등록이 필수이며, 미등록은 실행 거부한다.

### 3.5 오프라인 캐시 및 네트워크 격리 계약 (Offline-Only Backtesting)

* **원칙**: 백테스팅 기동 중(특히 Optuna 병렬 탐색 시)의 모든 데이터 로드(`data_loader.py`의 `collect_and_save` 및 `collect_1m_ohlcv`)는 오직 디스크 캐시(Parquet)만을 100% 사용하여 처리한다.
* **네트워크 차단**: `fetch_network=False`를 강제 적용하여 캐시 누락 시 실시간 API 호출 시도를 엄격하게 스킵하고 즉각 캐시 데이터만 슬라이싱하여 반환한다.
* **이유**: 병렬 백테스팅 스레드들이 각각 CCXT 소켓을 생성해 발생시키던 WSL 네트워크 포트 고갈 현상을 방지하고, 네트워크 레이턴시를 0으로 만들어 백테스팅 연산 속도를 극대화하기 위함이다.
* **선결 조건 및 수집 전략**:
  - 백테스팅에 필요한 `1h`, `1d`, `4h`, `funding` 데이터는 백테스팅 기동 전인 **1.5단계(`run_historical_sync(sync_4h=True, sync_1m=False)`)에서 단 한 번에 일괄 선 수집(Pre-fetch)**합니다.
  - **1m 데이터 Targeted Pre-fetch**: 시계열 크기가 매우 방대한 1m 데이터의 경우, 전체 후보군 수집 오버헤드를 막기 위해 1.5단계 수집 대상에서 완전히 생략합니다. 이후 **유니버스 7단계 필터링을 최종 통과한 백테스팅 대상 심볼군(`load_symbols`)에 대해서만 3단계 데이터 로드 직전에 콕 찝어 Targeted Pre-fetch를 실행**하여 영속화함으로써, 성능 향상과 대역폭 보전을 완벽하게 양립시킵니다.

---

## 4. Walk-Forward 평가 파이프라인

백테스트 엔진의 핵심 평가는 아래 7단계로 고정한다.

### Stage 0: 입력 준비 검증 (Readiness)

* `UniverseSnapshot` 해시, 입력 기간 커버리지, 필수 컬럼 존재성 확인.
* 실패 시 최적화 진입 전 중단.

### Stage 1: Fold 스케줄 생성 (WF Scheduler)

* Inner AWF: `IS=24M, K=5, leg≈5M` (`FUTURES_AWF_K_LEGS=5`, opt_config.py 기준).
* Outer Rolling OOS: `IS=24M, OOS=6M, step=3M` (관측 전용).
* Atomic blocks: `6M non-overlap` (승격 통계 전용).

### Stage 2: Trial 탐색 및 목적함수 계산

* `compute_v3_score` 고정 가중치로 trial score 계산.
* DSR 다중검정 보정(`n_trials_eff`) 적용.

### Stage 3: Hard Gate 판정

핵심 게이트:
* `min_positive_leg_ratio >= 0.55`
* `worst_leg_tw_floor >= 0.85`
* `mean_leg_tw_floor >= 1.015`
* `DSR >= 0.60`
* `funding_drag_ceiling <= 0.30`

### Stage 4: Intrabar 재평가

* coarse 결과를 1m 경로로 재평가해 decay와 MDD를 검증.
* dual decay:
  * `percent_decay >= -15%` (coarse CAGR > 0일 때)
  * `absolute_decay_bps_yr >= -500`

### Stage 5: Atomic OOS 승격 판정

* `pass_ratio >= 70%` (예: 11개 block 중 8개 이상).
* `median_log_growth`, `worst_block_mdd` 동시 점검.

### Stage 6: Capacity/AUM ladder 검증

* `AUM=[10k, 50k, 100k, 250k, 500k]`.
* 승격 필수 통과: `50k, 100k, 250k`.

---

## 5. 실행 시뮬레이션 및 비용 모델

### 5.1 비용 모델

```text
roundtrip_cost_bps =
  fee_bps + spread_bps + impact_bps + tick_cost_bps + latency_buffer_bps + funding_proxy_bps
```

* spread는 `bookDepth` 기반(가용 구간), 과거 구간은 fallback estimator 사용.
* impact는 square-root 형태(`k=0.5` 기준).

### 5.2 Intrabar 청산/체결 규약

* Binance Vision에 `mark_price_1m`이 없어, 청산 판정은 `exec_low_1m/exec_high_1m` 기준을 표준으로 사용한다.
* Long 청산: `exec_low_1m`.
* Short 청산: `exec_high_1m`.
* funding은 8h 이벤트 시점의 보유 포지션에만 적용.

### 5.3 포지션 양자화 계약

```text
qty = floor(target_weight * equity / (price * step_size)) * step_size
if qty * price < min_notional:
    qty = 0
```

* 기본 `min_notional=20 USDT`.
* 잔여 비중은 다음 리밸런스에서 재흡수.

### 5.4 리스크 오버레이

* 5-cap: gross/net/beta/per-symbol/vol.
* drawdown overlay:
  * rolling 30d 손실 > 10%: gross scale `0.7`
  * rolling 30d 손실 > 15%: gross scale `0.4`

---

## 6. Universe 연동 규칙

### 6.1 스냅샷 입력 계약

* `UniverseSnapshot(as_of)`는 읽기 전용 입력으로 사용.
* 백테스트 중 스냅샷 재작성 금지.

### 6.2 멤버십 변경 처리

* 퇴출 심볼은 다음 1m open에서 target `0`으로 강제 청산.
* 신규 진입 심볼은 warm-up 구간에서 신호만 계산하고 거래는 지연한다.

### 6.3 OI/ADV crowding 연동

* `daily/metrics` 기반 OI/ADV 필터를 지원한다.
* 2020-09 이전 구간은 OI 데이터 부재로 비활성 처리한다.

---

## 7. Champion 승격 및 리포트 계약

### 7.1 Sequential Promotion Gate

```text
1) Inner AWF hard gate pass
2) Atomic block pass ratio pass
3) Intrabar decay/MDD pass
4) AUM ladder mandatory tiers pass
5) Existing champion과 우선순위 비교
```

### 7.2 Champion 비교 우선순위

1. `atomic_oos_pass_ratio`
2. `capacity_ceiling`
3. `median_log_growth`
4. `worst_block_mdd`
5. `intrabar_absolute_decay_bps_yr`

### 7.3 필수 산출 리포트

* 성과: CAGR, MDD, Calmar, Sortino, DSR.
* 실행: EV/Cost, turnover_cost_ratio, funding_drag.
* 안정성: positive_leg_ratio, worst_leg_tw, atomic pass ratio.
* 수용력: AUM pass/fail, capacity ceiling.

---

## 8. 테스트 코드 명세 압축 (`docs/futures/back-code.md` 반영)

### 8.1 완료된 핵심 검증

* Phase 1~14 구현 완료.
* 실데이터 경로 구조 이슈 3건 수정 완료:
  * metrics 헤더 누락 시 컬럼명 복구 (`binance_vision.py`)
  * 1m 수집 경로 Vision 우선 + API 보완 (`data_loader.py`)
  * `_run_portfolio_numba_block` 호출 시그니처 불일치 수정 (`opt_main_futures.py`)
* `--quick-backtest` 추가로 전략 모듈과 분리된 엔진 경로 검증 가능.

### 8.2 실행 아규먼트 및 Quick Backtest 통과 기준

실행 명령 파라미터는 `--mode` 아규먼트를 통해 동작 모드를 지정하며, `--quick-backtest` alias 플래그를 제공합니다.

*   `--mode` 지원 옵션:
    *   `quick-backtest` (기본값): 초고속 엔진 경로/정합성 검증용.
    *   `strategy`: 최적화 전략 탐색용.
    *   `strategy-smoke`: 단기 전략 스모크 테스트용.
    *   `full`: 전체 파이프라인 E2E 풀 코스 수행용.

실행 명령 예시:

```bash
uv run python -m src.execution.opt_main_futures \
  --skip-universe --skip-data-sync \
  --symbols BTCUSDT --trials 1 --tf 4h \
  --mode quick-backtest
```

또는 동일한 alias 플래그 사용:

```bash
uv run python -m src.execution.opt_main_futures \
  --skip-universe --skip-data-sync \
  --symbols BTCUSDT --trials 1 --tf 4h \
  --quick-backtest
```

판정 기준:
* RuntimeError/TypeError 없이 Optimization 단계 진입.
* Backtest/Walk-forward 경로가 끝까지 실행.
* `no_completed_trials`는 현재 quick 모드에서 허용 결과.

### 8.3 남은 검증 항목

* full path(quick 미사용) 1회 완주 확인.
* `tests/integration -k smoke_backtest` 기준 실데이터 E2E 고정.
* full path에서 `WalkForwardResult`/gate/candidate 생성까지 확인.

---

## 9. 적용 대상

* `src/execution/opt_main_futures.py`
* `src/domain/futures/backtest_preparation.py`
* `src/domain/futures/optimization/*`
* `src/domain/futures/validation/*`
* `src/domain/futures/portfolio/*`
* `src/core/utils/binance_vision.py`
* `src/domain/futures/data_loader.py`

---

## 10. 버그 수정 이력 (v3.2 → v3.3)

### 10.1 `_build_funding_event_arrays_1m` IndexError (2026-05-22 수정)

- **파일**: `src/domain/futures/optimization/opt_data_utils.py:256`
- **증상**: `np.searchsorted` 결과 `pos == n`(배열 끝 초과)일 때 `exec_ms[pos]` 직접 인덱싱 → `IndexError: index N is out of bounds for axis 0 with size N`. 모든 심볼 로드 실패 → `data_not_ready`.
- **원인**: NumPy `&` 연산은 short-circuit 없음. `(pos < n)` 조건이 True여야만 `exec_ms[pos]`가 안전하지만, 조건 평가 전에 우측 항목이 먼저 실행됨.
- **수정**: `pos_safe = np.clip(pos, 0, n - 1)` 삽입 후 `exec_ms[pos_safe]`로 인덱싱. `pos < n` 조건과 AND로 out-of-bounds pos는 valid=False 처리.

### 10.2 `RegimeConfig.enabled` 기본값 오류 (2026-05-22 수정)

- **파일**: `src/domain/futures/strategy/config.py`
- **증상**: `enabled=True`가 기본값인데 `src.domain.futures.strategy.regime` 모듈이 삭제된 상태 → `ModuleNotFoundError`. strategy mode 전체 진입 불가.
- **원인**: Regime provider 모듈이 아키텍처 정리 과정에서 제거됐으나 Config 기본값이 갱신되지 않음.
- **수정**: `enabled: bool = False`로 변경. 모듈 재구현 시 P2 승격 이후 활성화 예정.
