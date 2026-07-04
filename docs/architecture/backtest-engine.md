---
title: Futures Backtest Engine Architecture
domain: futures.backtest
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
last_verified: 2026-06-10
---

# 1. Purpose
Numba JIT 코어 기반의 이중 해상도(4h 의사결정, 1m 실행) 백테스트 시뮬레이터를 제공하여 최적화 단계와 최종 검증 단계 간의 수학적 일치성과 완벽한 정렬을 보장한다.

# 2. Core Logic & Math

### Execution Priorities (Intrabar 1m)
1. **Margin Call (청산)**: $Equity \leq 0$ 발생 시 강제 청산 처리.
2. **Time Stop**: 최대 보유 시간 초과 시 청산.
3. **Stop Loss (손절)**: 보수적 체결 적용 (예: Gap-Down 시 $O_{1m} < \text{Stop}$ 이면 $O_{1m}$에 체결).
4. **Rebalance / Entry**: 신규 진입 및 리밸런싱 실행.

### Position Quantization
- **수량 계산**: $Q = \lfloor \frac{W_{\text{target}} \cdot E}{P \cdot \text{step\_size}} \rfloor \cdot \text{step\_size}$
- **최소 수량 필터**: $Q \cdot P \geq \text{min\_notional}$

### Accounting Identity
- **계좌 평가액 산식**:
  - $E_{T} = E_{0} - \sum \text{Fees} - \sum \text{Funding} + \sum \text{RealizedPnL} + \text{UnrealizedPnL}_{T}$

### Look-ahead Prevention
- $T$ 시점의 의사결정은 오직 $P_{T}$ 이하의 과거 데이터만 사용.
- 실제 거래 실행은 $(T, T+1]$ 구간의 1m 고해상도 시계열을 따라 순차적으로 진행.

# 3. Architecture Flow

```mermaid
graph TD
    A[4h Decision Weights] --> C[Data Alignment]
    B[1m Intrabar OHLCV] --> C
    C --> D[Numba Core: execution_sim]
    D --> E{Intrabar Loop}
    E --> F[Trigger: Liquidation/Stop]
    E --> G[Trigger: Scheduled Rebalance]
    F --> H[Fill at Conservative Price]
    G --> H
    H --> I[Update Account State]
    I --> J[Log Trade / Funding]
    E -.=>|Next 1m Bar| E
    J --> K[Final Equity Curve & Trade Log]
```

# 4. Core Variables & I/O

| Type | Variable | Description |
|---|---|---|
| **Input** | `target_weights_2d` | $[B_{4h}, N]$ 포트폴리오 목표 비중 배열. 범위: `[-1.0, 1.0]` |
| **Input** | `exec_ohlc_1m` | $[B_{1m}, N]$ 세부 1m OHLCV 가격 데이터 |
| **Input** | `funding_event_mask`| $[B_{1m}, N]$ 펀딩비 정산 발생 시점 마스크 |
| **Param** | `round_trip_cost_bps` | 왕복 거래비용 (slippage + fee 등) |
| **Param** | `min_notional` | 거래소 최소 주문 금액 제한 |
| **Output**| `Equity Curve` | 시계열 계좌 평가액 $E_{T}$ |
| **Output**| `Trades Log` | 모든 체결 가격, 수량, 수수료 내역이 기록된 원장 |

# 5. Edge Cases & Handling
- **마진 부족**: 목표 비중이 계좌 평가액을 초과할 경우, $Q$는 최대 가용한 마진 범위 내로 클램핑되어 과배치를 방지함.
- **청산과 손절의 동시 발생**: 1m 봉 내에서 고가/저가가 청산 기준과 손절 기준을 동시에 충족할 경우, 청산(Margin Call) 로직을 우선 처리하여 포트폴리오를 전량 회수함.
