# Rolling Library Admission Walk-Forward Replay Result (`technical-5symbol-rolling`)

## 1. 실행 요약 (Execution Summary)

- **실행 프로필 (Profile)**: `technical-5symbol-rolling`
- **데이터 스냅샷 기준 시각 (`as_of`)**: `2026-07-07 20:00+00:00`
- **실행 시간 (Wall-Clock Execution Time)**: **약 28.5초**
- **실행 결과 상태 (Status)**: `paper` (Sealed Walk-Forward Execution)

---

## 2. 전체 백테스트 및 신뢰성 지표 (Overall Performance & Reliability)

| 측정 지표 (Metric) | 수치 및 결과 (Value) |
| :--- | :--- |
| **CAGR (연간 복리 수익률)** | **+25.73%** (0.2573) |
| **MDD (최대 낙폭)** | **-15.99%** (-0.1599) |
| **Sharpe Ratio (샤프 지수)** | **1.224** |
| **Sortino Ratio (소르티노 지수)** | **1.815** |
| **Profit Factor (수익인자)** | **1.406** |
| **Win Rate (승률)** | **52.4%** |
| **Total Closed Trades (총 체결 트레이드 수)** | **311** |
| **90% LCB CAGR (하한 신뢰구간 CAGR)** | **+0.95%** |
| **관측 신뢰성 검증 Verdict** | `FAIL` (LCB > 0 이나 일부 분기 변동성 임계 초과) |
| **스트레스 테스트 Verdict** | `PASS` (비용 및 수수료 1.5배 가중치 견딤) |

---

## 3. 분기별 리밸런싱 및 승자 제안 내역 (Quarterly Rebalances)

| 분기 리밸런싱 일자 | 윈도우 관측 기간 (Observed Range) | 배치 적용 기간 (Deployment Period) | 선택 상태 |
| :--- | :--- | :--- | :--- |
| `2024-07-01 00:00` | `2022-05-28 ~ 2024-06-30` | `2024-07-01 ~ 2024-09-30` | `COMPLETE` |
| `2024-10-01 00:00` | `2022-08-28 ~ 2024-09-30` | `2024-10-01 ~ 2024-12-31` | `COMPLETE` |
| `2025-01-01 00:00` | `2022-11-28 ~ 2024-12-31` | `2025-01-01 ~ 2025-03-31` | `COMPLETE` |
| `2025-04-01 00:00` | `2023-02-26 ~ 2025-03-31` | `2025-04-01 ~ 2025-06-30` | `COMPLETE` |

---

## 4. 원장 및 스냅샷 검증 (Ledger Provenance)

- **Rebalance Ledger File**: [`docs/results/rebalance_ledger.jsonl`](file:///home/kth/my_coin_traider/docs/results/rebalance_ledger.jsonl)
- **Current Pointer File**: [`docs/results/current_rolling_profile.json`](file:///home/kth/my_coin_traider/docs/results/current_rolling_profile.json)
