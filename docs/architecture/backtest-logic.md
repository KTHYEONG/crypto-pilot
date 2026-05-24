---
title: Backtest Logic & Semantics
domain: futures-backtest-logic
type: architecture
status: active
priority: high
ai_read_policy: always
related_paths:
  - src/domain/futures/portfolio/execution_sim.py
  - src/domain/futures/portfolio/friction_model.py
change_triggers:
  - src/domain/futures/portfolio/execution_sim.py
  - src/domain/futures/portfolio/friction_model.py
last_verified: 2026-05-24
---

# Backtest Logic & Semantics

## 1. Overview
백테스트 엔진의 핵심 체결 로직 및 회계 처리 방식을 정의합니다. 특히 1m 단위의 Intrabar 시뮬레이션에서의 가격 우선순위와 마찰 비용 계산을 다룹니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `backtest_target_weights_intrabar_numba` | 1m 경로 기반의 핵심 시뮬레이션 루프 |
| `friction_model.py` | 수수료 및 슬리피지 수학적 모델링 |
| `risk_controls.py` | 마진 콜 및 강제 청산 로직 구현 |

---

## 3. Data Flow

```text
[1m OHLC Path] -> [Trigger Check (Stop/Liq)] -> [Fill Price Calculation] 
  -> [Account State Update (Equity/Margin)] -> [Trade Logging]
```

---

## 4. Business Rules

### Must Follow
- **Execution Priority:** 1) Liquidation -> 2) Kill/Max Hold -> 3) Stop Loss -> 4) Rebalance 순으로 우선순위 적용.
- **Conservative Fills:** 1m 바 내에서 가장 불리한 가격(Low for Long Stop, High for Short Stop)을 기준으로 체결가 산정.
- **Rounding Strategy:** 모든 수량 산출 시 `step_size` 기반 내림(Floor) 적용하여 가용 증거금 초과 방지.

### Must Not Do
- **Instant Fills:** 캔들 간 가격 갭 발생 시 지정가(Stop) 체결 금지. 반드시 Open 가격으로 슬리피지를 포함하여 체결.

---

## 5. Detailed Specifications

### 5.1 9 Pillars of Integrity (무결성 원칙)
1. **Conservation of Money:** `Final_Balance == Initial_Balance - Σ(Fees) - Σ(Carry) + Σ(Realized_PnL)`
2. **Exposure Cap:** Gross Exposure 및 Concurrent Symbol 한도를 정확히 준수.
3. **Liquidation Guard:** 포지션별 청산가 도달 또는 계좌 레벨 `equity ≤ 0` 시 즉시 파산 처리.
4. **Cost Precision:** `settings.py`의 비용 모델과 시뮬레이터 간 수치적 일치.
5. **Price Gaps:** 캔들 간 갭 시 지정가가 아닌 시장가(Open) 체결로 유리한 조작 방지.
6. **Funding Signs:** Long 지불(-), Short 수취(+) 부호 규약 준수.
7. **Look-ahead 차단:** T 신호는 반드시 T+1 이후의 실행 윈도우에서만 작동.
8. **NaN/Inf 격리:** 입력 데이터 오염 시 해당 심볼 진입 스킵 및 계좌 보호.
9. **Determinism:** 동일 조건에서 소수점 오차 없는 완전 동일 결과 출력.
10. **B1 Canonical Cost Model (Label-Objective Split):** `src/domain/futures/strategy/labels.py`의 `build_label_panel()`은 반드시 GROSS alpha(cost=0.0)를 출력해야 하며, 거래 비용 차감은 정확히 한 번 `src/execution/opt_main_futures.py`의 objectives 마찰층(friction + EV_HURDLE)에서만 발생해야 함. 이중 차감(double-deduction) 방지를 위해 필수 불변식.

### 5.2 Position Quantization Formula
```python
qty = floor(target_weight * equity / (price * step_size)) * step_size
if qty * price < min_notional:
    qty = 0
```
- **min_notional:** 기본 20 USDT 적용.

### 5.3 Execution Cost Model
- **진입/청산 단가:** `Price * (1 ± slippage_rate)`
- **수수료(Taker):** `Notional * taker_fee_rate`
- **Total Round-trip Cost:** `2 * (fee + slippage) + impact + tick_cost`

---

## 6. Examples
- **Input:** Stop Loss at 100, Next bar Open at 95 (Gap-down)
- **Output:** Filled at `95 * (1 - slippage)` (보수적 체결 원칙)

---

## 7. Testing Expectations
- **Friction Precision Test:** 수수료/슬리피지 계산이 `settings.py`의 `round_trip_cost_bps()`와 일치하는지 확인.
- **Sequence Test:** 동일 바 내에서 Liquidation과 Stop-loss가 동시 발생 시 Liquidation이 우선 처리되는지 확인.
- **Boundary Test:** `min_notional` 경계값에서 주문이 정상적으로 차단되거나 실행되는지 확인.
