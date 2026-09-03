---
title: MHS Architecture - 05. Execution Replay & Simulated Ledger
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/execution/ledger.py
  - src/mhs/execution/strategy_replay.py
  - src/mhs/execution/contracts.py
  - src/mhs/pipeline/stages/replay.py
  - src/mhs/params.py
change_triggers:
  - src/mhs/execution/*.py
  - src/mhs/pipeline/stages/replay.py
last_verified: 2026-09-03
---

# 05. 체결 리플레이 및 모의 재고 원장 (Simulated Inventory Ledger)

## 1. 개요 (Overview)
포트폴리오의 실현 손익은 단순히 시계열 가격 곱셈으로 계산될 수 없습니다. 실제 환경에서는 체결 지연, 슬리피지, 펀딩비 지불 주기, 수수료, 그리고 체결 전후의 현금 잔고 변화가 복합적으로 작용합니다.

MHS는 1시간 단위 신호를 5분봉 고해상도 격자 상에서 체결 리플레이하고, **모의 실행 원장([`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py))**을 단일 진실 원천(Single Source of Truth)으로 삼아 모든 PnL과 리스크 지표를 엄밀하게 산출합니다.

---

## 2. 3가지 체결 프록시 바운드 (Execution Proxy Bounds)

MHS는 전략의 체결 민감도와 비용 내구성을 다각도로 검증하기 위해 3가지 체결 바운드를 제공합니다 ([`ExecutionSpec`](file:///home/kth/crypto-pilot/src/mhs/contracts.py)):

| 체결 모드 | 적용 기준 | 상세 체결 메커니즘 | 역할 및 목적 |
|---|---|---|---|
| **`OHLCV_IMMEDIATE_TAKER`** | **Primary Research GO (기본)** | 주문 발생 즉시 해당 5분봉 Close 가격으로 Taker 즉시 체결 | 전략의 시장 참여율(`participation_warnings`)이 $10^{-9}$ 수준으로 극히 미미하므로, 시장 충격을 피하기 위해 Passive 대기를 수행할 경제적 이유가 없다는 실측 연구에 따라 고정된 **공식 기준** |
| **`OHLCV_STRICT_PROXY`** | Patient Reference (참고용) | 5분봉 High/Low가 지정가를 관통(Trade-through)할 때만 Maker 인정, 30분 미체결 시 Taker Fallback | 인내형(Patient) 지정가 주문의 체결률 및 슬리피지 비교를 위한 보조 지표 |
| **`SPREAD_AND_COST_X3`** | Cost Stress Bound (스트레스) | Immediate-Taker 조건과 동일하나 비용만 **3배**(Maker 6bps, Taker 15bps, Slippage 9bps) 가산 | 극단적인 유동성 위기 및 수수료 급등 상황에서의 전략 알파 생존력 검증 |

---

## 3. Simulated Inventory Ledger: 단일 진실 원천 원장

[`src/mhs/execution/ledger.py`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py)의 [`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py)는 매 이벤트 타임스탬프마다 엄격한 회계 순서에 따라 원장을 갱신합니다:

```text
[새로운 타임스탬프 t 진입]
       │
       ▼ 1. 마크-투-마켓 평가 (MTM Valuation)
직전 보유 수량(Units)을 현재 시점 마크 가격(Mark Price)으로 평가
       │
       ▼ 2. 펀딩비 결제 (Accrued Funding Charge)
해당 구간에 발생한 펀딩비(Funding Rate)를 직전 보유 가액(Units * Mark)에 비례하여 현금(Cash)에서 정산
       │
       ▼ 3. 주문 의도 상쇄 (Intent Netting)
새로 생성된 목표 포지션과 현재 보유 수량을 비교하여 실체결 필요 수량(Delta Qty) 산출
       │
       ▼ 4. 프록시 체결 및 수수료 차감 (Proxy Fill Application)
체결 가격(Fill Price) 및 체결 수수료(Fee Bps)를 반영하여 수량(Units) 및 현금(Cash) 갱신:
  Cash_t = Cash_{t-1} - (Delta_Qty * Fill_Price + Fee)
       │
       ▼ 5. 자산 및 회전율 집계 (Equity & Turnover Calculation)
총 자산 = Cash + Σ (Units * Mark Price)
회전율 = Σ |Delta_Qty * Fill_Price| / Pre-trade Equity
```

### 불변식 및 안전장치
- **Causal Timing Invariant**: 체결 프록시는 자신의 발생 타임스탬프 이전의 PnL을 취하거나 손실을 낼 수 없습니다.
- **Fill-Mark Parity Guard ([`FILL_MARK_PRICE_PROTECTION_BAND = 0.05`](file:///home/kth/crypto-pilot/src/mhs/params.py))**:
  - 체결 가격과 마크 가격 간의 로그 괴리율이 5%를 초과할 경우, 이상 호가로 판단하여 방어 게이트를 작동합니다.

---

## 4. 17차 연율화 수정 (The 17th Annualization Correction)

### 1) 버그의 원인
- MHS의 전략 신호는 1시간봉(`1h`)이지만, 실제 체결 원장([`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py))은 정밀한 슬리피지 측정을 위해 5분봉(`5m`, `execution_timeframe=5m`) 해상도로 리플레이됩니다.
- 그러나 원장의 집계 모듈에서 5분봉 시계열에 1시간봉 연율화 상수([`_PERIODS_PER_YEAR_1H = 8760`](file:///home/kth/crypto-pilot/src/mhs/params.py))를 적용하던 불일치 버그가 존재했습니다.

### 2) 교정 및 조치
- 5분봉 해상도에 일치하는 정확한 연율화 상수([`_PERIODS_PER_YEAR_5M = 365 * 24 * 12 = 105,120`](file:///home/kth/crypto-pilot/src/mhs/params.py))로 교정되었습니다.

### 3) 교정 결과 및 영향
- **Sharpe 비율 및 Research GO 판정**: 일봉 수익률 기반으로 측정되는 Sharpe 비율 및 Research GO 승인 여부에는 일절 변동이 없습니다.
- **자산 증식 지표의 정상 복원**: 기하 연복리 수익률(CAGR), 연간 회전율, 부트스트랩 최대 낙폭(MDD) 등 자산 증식 수치가 정확히 교정되었습니다 (CAGR 0.63% $\rightarrow$ **7.84%**로 정상 복원).

---

## 5. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 데이터 클래스 |
|---|---|---|
| 모의 재고 원장 | [src/mhs/execution/ledger.py](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py) | `simulated_inventory_ledger` |
| 전략 인지 체결 리플레이 | [src/mhs/execution/strategy_replay.py](file:///home/kth/crypto-pilot/src/mhs/execution/strategy_replay.py) | `strategy_aware_execution_replay` |
| 체결 계약 및 결과 규격 | [src/mhs/execution/contracts.py](file:///home/kth/crypto-pilot/src/mhs/execution/contracts.py) | `SimulatedInventoryLedgerResult`, `ExecutionDataGap` |
| 리플레이 파이프라인 스테이지 | [src/mhs/pipeline/stages/replay.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/replay.py) | `run_replays` |
| 5분봉 연율화 상수 및 보호 밴드 | [src/mhs/params.py](file:///home/kth/crypto-pilot/src/mhs/params.py) | `PERIODS_PER_YEAR_5M`, `FILL_MARK_PRICE_PROTECTION_BAND` |
