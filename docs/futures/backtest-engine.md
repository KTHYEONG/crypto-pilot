# Binance Futures 백테스트 엔진 아키텍처 (Futures Backtest Engine)

**최종 검증/확정**: 2026-05-19  
**핵심 설계 목적**: 1m Intrabar 경로 기반의 보수적 체결, 수학적/회계적 무결성 보장, Look-ahead 편향 원천 차단, 최적화/백테스트 엔진 간 실행 Semantics 완전 통일.

---

## 1. 핵심 아키텍처 및 철학 (Architecture & Philosophy)

본 백테스트 엔진은 기존 OHLC(Open-High-Low-Close) 기반 Coarse 엔진이 가지는 한계(동일 캔들 내 Stop-Loss와 Take-Profit 발생 순서의 불확실성 등)를 극복하기 위해 설계된 **이중 해상도(Dual-Resolution) 시뮬레이터**입니다.

- **Decision Timeframe (ex: 4h)**: 타겟 비중(Target Weights)을 산출하고 포트폴리오를 리밸런싱할지 결정하는 기준 시점입니다. (1h Raw Data를 닫힌 바 기준으로 파생 생성)
- **Execution Timeframe (1m)**: 실제 가격 경로를 따라가며 체결, 청산, 펀딩, 스탑로스(Stop-Loss) 등을 시뮬레이션하는 고해상도 실행 윈도우입니다.

```text
┌─────────────────────────────────────────────────────┐
│ [공통 Preparation Layer: Data Alignment]             │
│ - 1h Base Grain -> 4h 파생 Decision Bar 집계         │
│ - Decision Bar ↔ 1m Execution Window 매핑            │
│ - Funding / Kill / Volume block 정렬                 │
└──────────────────┬──────────────────────────────────┘
                   │ PreparedBacktestInputs (2D Numpy Arrays)
                   ▼
┌─────────────────────────────────────────────────────┐
│ [Execution Core: 포트폴리오 시뮬레이션 (Numba)]         │
│ ├─ Intrabar 1m Mode (Default / Accurate)            │
│ │   - 1m High/Low 경로 기반 Stop / Liquidation 판정    │
│ │   - Event 기반 Funding 적용                         │
│ └─ Coarse OHLC Mode (Fallback / Fast)               │
│     - 4h Open/High/Low 기반의 근사 체결               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ trades_df, equity_curve, final_bal
```

---

## 2. 디렉토리 구조 및 모듈 매핑

`src/domain/futures/` 내 백테스트 관련 파일은 단일 책임을 갖도록 분리되어 있습니다.

| 파일명 | 주된 역할 및 책임 | 주요 외부 라이브러리 |
|---|---|---|
| `backtest_engine.py` | `PortfolioBacktestEngine` 인스턴스화, 하이퍼파라미터 파싱 및 오케스트레이션 | `pandas`, `numpy` |
| `backtest_preparation.py` | 공통 Execution Input 준비, 1h->4h 집계 및 1m Window 인덱스 매핑 | `numpy` |
| `portfolio/execution_sim.py` | Numba `@njit` 가속 기반 실행 코어 (수학적 체결 및 상태 업데이트 루프) | `numba`, `numpy` |
| `portfolio/portfolio_constructor.py` | 알파, HMM 결합, 공분산 기반의 `precompute_rebalance_weights` 산출 | `numpy` |
| `optimization/optimizer.py` | `PortfolioBacktestEngine`과 **동일한 Numba 코어**를 호출하는 최적화 경로 | `numba`, `numpy` |

---

## 3. 데이터 계약 및 Numba 메모리 구조 (Data Contracts)

최대 성능을 위해 엔진의 코어 로직(`execution_sim.py`)은 `pandas` 구조를 배제하고 Numpy 2D Array(`shape: [n_bars, n_symbols]`)만을 사용합니다.

### 3.1 인덱스 및 매핑 규칙
* **Index Mapping**: `prepare_backtest_inputs`가 생성합니다. Decision Bar의 인덱스 `i`에 대해, 실행되어야 할 1m 바들의 시작과 끝 인덱스는 `exec_bar_start_1m_idx[i]`와 `exec_bar_end_1m_idx[i]`로 정의됩니다.

### 3.2 핵심 Numba Input Arrays (`shape = [N, M]`)
| 변수명 | 해상도 | 설명 |
|---|---|---|
| `close_2d`, `atr_2d` | 4h (Decision) | 직전 닫힌 바(closed bar)의 종가 및 ATR (신호 산출용) |
| `target_weights_2d` | 4h (Decision) | `i` 시점에 도달해야 할 심볼별 목표 자본 비중 |
| `exec_open/high/low/close_1m` | 1m (Execution)| 1m 바의 OHLC 경로 (실제 슬리피지 및 체결가 산출용) |
| `funding_event_mask_1m` | 1m (Execution)| 1m 바에 펀딩 이벤트 발생 여부 (1=발생, 0=미발생) |
| `funding_rate_event_1m` | 1m (Execution)| 이벤트 발생 시의 실제 펀딩비율 |

---

## 4. Intrabar 실행 시뮬레이션 알고리즘 (Algorithm Flow)

`execution_sim.py` 내의 `backtest_target_weights_intrabar_numba` 함수는 다음의 로직 흐름(Tick-by-Tick 에 준하는 1m Loop)을 따릅니다. 동일한 1m 바 내에서도 철저한 **보수성 원칙(Conservative Principle)**을 적용합니다.

### Step-by-Step 1m Bar 처리 로직
매 Decision Bar `i`에 대하여, 해당하는 1m 윈도우 `[start_1m, end_1m]`를 순회합니다.

1. **Scheduled Rebalance (정기 리밸런싱)**:
   * 윈도우의 **첫 번째 1m Bar의 Open** 시점(`exec_open_1m[start_1m]`)에 즉시 실행됩니다.
   * `target_weights_2d[i]`를 맞추기 위해 델타 비중만큼 매수/매도하며 비용(`fees`, `slippage`)을 선차감합니다.
2. **관측 가능 이벤트 (Kill / Max Hold)**:
   * `kill_signal` 발생 또는 `max_hold_bars` 초과 시 현재 1m Bar의 `Open` 가격으로 즉시 시장가 강제 청산합니다. (Stop-loss보다 우선 판별)
3. **강제 청산 판정 (Liquidation)**:
   * 포지션 유지 증거금(`maintenance_margin_rate`) 기준의 파산가(Liquidation Price)를 계산합니다.
   * `exec_low_1m` (Long) 또는 `exec_high_1m` (Short)이 청산가를 터치하면 포지션이 즉각 소멸되며 `liquidation_fee`가 부과됩니다.
4. **Stop Loss & Trailing Stop 판정**:
   * 청산가를 터치하지 않았다면, 스탑로스 도달 여부를 확인합니다.
   * **Long 포지션 스탑로스**:
     * `exec_open_1m <= stop_price` : 갭 하락(Gap-down). 불리하게 **`exec_open_1m * (1 - slippage)`** 에 체결.
     * `exec_low_1m <= stop_price` : 정상적인 꼬리 터치. **`stop_price * (1 - slippage)`** 에 체결.
   * **Short 포지션 스탑로스**:
     * `exec_open_1m >= stop_price` : 갭 상승(Gap-up). 불리하게 **`exec_open_1m * (1 + slippage)`** 에 체결.
     * `exec_high_1m >= stop_price` : 정상 터치. **`stop_price * (1 + slippage)`** 에 체결.
5. **Funding Event 처리**:
   * `funding_event_mask_1m`이 1인 바인 경우, 포지션 명목 가치(Notional)에 `funding_rate_event_1m`를 곱하여 `fund_fee_stored` 장부에 누적합니다.

---

## 5. 마찰 비용 및 회계 모델 (Friction & Accounting)

현실적인 거래 환경 시뮬레이션을 위해 엔진은 다음과 같은 모델을 적용합니다.

### 5.1 체결 마찰 비용 (Execution Cost)
어떠한 거래든(진입, 청산, 스탑로스) 아래의 수식에 의해 비용이 발생합니다.
* **진입 단가(Long)** = `Price * (1 + slippage_rate)`
* **청산 단가(Long)** = `Price * (1 - slippage_rate)`
* **수수료(Taker)** = `Notional(수량 * 단가) * taker_fee`
※ Slippage는 단순 퍼센트 비율 외에도 1m 거래량 기반의 `Volume Impact` (Square-root 모델)로 확장될 수 있도록 인터페이스가 준비되어 있습니다.

### 5.2 펀딩비 규약 (Funding Physics)
* 부호 규약: 양수(+) 펀딩비 환경에서 **Long은 비용 지불(-)**, **Short은 수익 수취(+)**
* 계산식: `Funding Fee = Pos_Amount * Mark_Price * Funding_Rate * (1 if Long else -1)`
* 펀딩 장부 누적: 매 이벤트마다 장부에 누적되며, 포지션이 최종 청산(Exit)될 때 계좌의 Realized PnL에 정산됩니다.

---

## 6. 수학적 무결성 검증 (9 Pillars of Integrity)

엔진은 테스트(`backtest-test.md`)를 통해 아래 9대 원칙이 훼손되지 않음을 증명합니다.

1. **노출 한도 (Exposure Cap)**: Gross Exposure와 Concurrent Symbol 한도를 정확히 스케일다운하여 준수.
2. **청산 방지 (Bankruptcy Guard)**: `current_equity ≤ 0` 발생 즉시 전체 청산 및 파산 처리.
3. **마찰 정밀도 (Cost Precision)**: Turn-over 시의 수수료/슬리피지 수학적 일치.
4. **갭 처리 (Price Gaps)**: 캔들 간 갭 시 지정가(Stop)가 아닌 시장가(Open)로 체결하여 유리한 조작 방지.
5. **펀딩 부호 (Funding Signs)**: Short/Long 의 정확한 보유 비용 및 수익 누적.
6. **Look-ahead 차단**: T 신호는 반드시 (T+1) 시점 이후의 Execution 1m 윈도우에서 작동.
7. **자금 보존 항등식 (Conservation of Money)**:
   `Final_Balance == Initial_Balance - Σ(Fees) - Σ(Carry_Costs) + Σ(Realized_Net_PnL)`
8. **NaN/Inf 격리**: 입력 데이터 내 NaN 존재 시 해당 심볼 진입 스킵 및 전체 계좌 오염 방지.
9. **결정론 (Determinism)**: 동일 시드/데이터에 대해 부동소수점 오차 없이 완전 동일한 Equity Curve 출력.

---

## 7. 품질 검증 가드레일 (Acceptance Criteria)

새로운 알파 로직이나 최적화 모듈을 개발할 때, 엔진 사용의 유효성을 보장하는 가드레일입니다.

1. **엔진 Semantics 완전 일치**: 최적화 환경(`optimizer.py`)에서 뽑은 파라미터가 백테스트 환경(`backtest_engine.py`)과 완전히 동일한 Core Numba Loop를 사용해야 합니다.
2. **스트레스 테스트 내성**: 50개 이상의 멀티 심볼 난수 비중(Random Weights) 주입 시에도 NaN 자산이나 Negative Margin 에러가 발생하지 않아야 합니다.
3. **Coarse vs Intrabar 편차 확인**: 최적화를 위한 Coarse 엔진 결과가 실제 배포 전 Intrabar 1m 엔진에서 크게 이탈하지 않는지 검증해야 합니다.