---
title: Binance Futures Backtest Engine
domain: futures-backtest
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/backtest/
  - src/domain/futures/portfolio/execution_sim.py
change_triggers:
  - src/domain/futures/backtest/**
  - src/domain/futures/portfolio/execution_sim.py
last_verified: 2026-05-25
---

# Binance Futures Backtest Engine

## 1. Overview
1m Intrabar 경로 기반의 보수적 체결, 수학적/회계적 무결성 보장, Look-ahead 편향 원천 차단을 목적으로 하는 이중 해상도(Dual-Resolution) 시뮬레이터입니다. 최적화(`optimizer.py`)와 백테스트(`backtest_engine.py`) 환경 간 실행 Semantics를 완전히 통일합니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `backtest_engine.py` | `PortfolioBacktestEngine` 오케스트레이션 및 결과 집계 |
| `backtest_preparation.py` | 1h->4h 집계 및 1m Window 인덱스 매핑 (Data Alignment) |
| `execution_sim.py` | Numba `@njit` 가속 기반 실행 코어 (수학적 체결 및 상태 업데이트) |
| `portfolio_constructor.py` | Kelly scaling, cap projection, 양자화 |
| `optimizer.py` | 백테스트와 동일한 Numba 코어를 호출하는 하이퍼파라미터 최적화 경로 |

---

## 3. Data Flow

```text
[Data Alignment (1h/1m Raw)] 
  -> [Decision Layer (4h): target_weights 산출] 
  -> [Execution Layer (1m): Intrabar 경로 시뮬레이션] 
  -> [Accounting: PnL, Fees, Funding 집계] 
  -> [Output: Trades, Equity Curve]
```

---

## 4. Business Rules

### Must Follow
- **Conservative Principle:** 동일 1m 바 내에서 불리한 가격 체결을 우선시(Gap-down/Gap-up 처리).
- **Scheduled Rebalance:** 각 Decision Bar의 첫 번째 1m Bar Open 시점에 즉시 실행.
- **Single Source of Truth Costs:** `src/core/settings.py`의 `round_trip_cost_bps()`를 모든 모듈이 공유.
- **Offline-Only Backtesting:** 백테스팅 중(Optuna 탐색 시 포함) 모든 데이터 로드는 100% 디스크 캐시(Parquet)만 사용하며 네트워크 호출(`fetch_network=False`)을 엄격히 차단함.

### Must Not Do
- **Look-ahead Bias:** T 시점 신호는 반드시 (T+1) 시점 이후의 Execution 윈도우에서 작동해야 함.
- **Portfolio Control:** ML 전략이 직접 주문/레버리지를 제어하지 말 것 (Alpha Supplier 역할만 수행).

---

## 5. Detailed Specifications

### 5.1 Numba Input Arrays (Core Contract)
최대 성능을 위해 엔진은 Pandas를 배제하고 다음의 Numpy 2D Array(`[B, N]`)를 코어 루프에 주입합니다.

| 변수 | 해상도 | 설명 |
|---|---|---|
| `close_2d`, `atr_2d` | 4h (Decision) | 의사결정 시점의 종가 및 변동성 |
| `target_weights_2d` | 4h (Decision) | 리밸런싱 목표 비중 |
| `exec_o/h/l/c_1m` | 1m (Execution)| Intrabar 체결 시뮬레이션 경로 |
| `funding_event_mask`| 1m (Execution)| 펀딩 정산 시점 마스크 (1=발생) |

### 5.2 Optuna Execution Contract
최적화 실행은 `run_tracker.log_optuna_contract()`를 통해 다음 항목을 추적하여 재현성을 보장합니다.
- `requested_trials_per_phase`, `planned_total_trials`
- `sampler_by_phase`, `worker_by_phase`
- `storage_url` (Redis or SQLite WAL)

### 5.2.1 Redis Storage Resolution Contract (2026-05-24)
- `setup_optuna_storage()`는 아래 우선순위로 Redis endpoint를 결정합니다.
- `FUTURES_REDIS_URL`
- `FUTURES_REDIS_HOST` + `FUTURES_REDIS_PORT` + `FUTURES_REDIS_DB` + `FUTURES_REDIS_PASSWORD` + `FUTURES_REDIS_TLS`
- `OPT_FUTURES_CONFIG["FUTURES_REDIS_URL"]`
- storage 생성 전에 TCP preflight를 수행합니다.
- `FUTURES_REDIS_CONNECT_TIMEOUT_SEC` (기본 `1.5`)
- `FUTURES_REDIS_CONNECT_RETRIES` (기본 `2`)
- 시작 로그 `[OPTUNA-STORAGE] scheme=... host=... port=... db=...`로 실제 적용 endpoint를 출력합니다.

### 5.3 Walk-Forward Evaluation Pipeline (7 Stages)
1. **Readiness:** 데이터 커버리지 및 필수 컬럼 존재성 검증.
2. **WF Scheduler:** Inner AWF(IS=24M, K=5) 및 Outer Rolling(IS=24M, OOS=6M) 스케줄링.
3. **Trial Search:** `compute_v3_score` 및 DSR 다중검정 보정 적용.
4. **Hard Gate:** `min_positive_leg_ratio >= 0.55`, `worst_leg_tw_floor >= 0.85` 등 검증.
5. **Intrabar Decay:** Coarse 대비 Intrabar 결과의 열화율(`percent_decay >= -15%`) 검증.
6. **Atomic OOS:** 6M Non-overlap 블록 승격 판정 (`pass_ratio >= 70%`).
7. **Capacity Ladder:** AUM 단계별(50k, 100k, 250k 등) 수용력 검증.

---

## 6. Examples

### Accounting 항등식 검증
- **Input:** `Initial: 10,000`, `Fees: 50`, `PnL: +200`, `Funding: -10`
- **Output:** `Final Balance: 10,140` (`Final == Init - ΣFees - ΣCarry + ΣPnL`)

---

## 7. Testing Expectations
- **Unit test:** `execution_sim.py`의 Numba 루프 개별 로직(Liquidation, Stop-loss 등) 검증.
- **Semantics Match:** 최적화 파라미터가 실제 백테스트와 오차 없이 동일한 Equity Curve를 생성하는지 확인.
- **Deterministic Test:** 동일 시드/데이터에 대해 부동소수점 오차 없는 완전 동일 결과 출력 확인.
- **Pipeline Safety:** `opt_main_futures.py`의 CLI/pipeline 분기(`strategy`, `strategy-smoke`, 오류 경로)는 오프라인 stub 기반으로 검증.
- **No Live Network in Tests:** 백테스트/최적화 테스트 스위트는 실거래소 API 자격증명/네트워크에 의존하지 않음.
