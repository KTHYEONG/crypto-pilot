# Binance Futures 백테스트 아키텍처 (v3.1 - Engine/Execution Focus)

**최종 업데이트**: 2026-05-20  
**핵심 목적**: 전략 품질과 분리된 백테스트 엔진의 정확성, 재현성, 실행 현실성 보장.

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
│ - Inner AWF(K=8, IS=24M), Atomic 6M block 평가             │
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

---

## 4. Walk-Forward 평가 파이프라인

백테스트 엔진의 핵심 평가는 아래 7단계로 고정한다.

### Stage 0: 입력 준비 검증 (Readiness)

* `UniverseSnapshot` 해시, 입력 기간 커버리지, 필수 컬럼 존재성 확인.
* 실패 시 최적화 진입 전 중단.

### Stage 1: Fold 스케줄 생성 (WF Scheduler)

* Inner AWF: `IS=24M, K=8, leg=3M`.
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

### 8.2 Quick Backtest 통과 기준

실행 명령:

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
