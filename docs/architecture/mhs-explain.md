# Multi-Horizon Market State (MHS) 시스템 아키텍처 및 체결 가이드

## 1. 개요 및 목적 (Overview & Purpose)

**Multi-Horizon Market State (MHS)**는 여러 거래 보유 기간(Horizon)의 횡단면(Cross-Sectional) 알파 신호를 독립된 북(Book)으로 구축하고, 5분봉 단위의 실제 계약수량 기반 **모의 실행 원장(Simulated Inventory Ledger)**을 통해 전략의 유효성을 검증하는 **Phase 1 알파 연구 및 실행 파이프라인**입니다.

### 핵심 역할
- **다중 호라이즌 신호 앙상블**: Short-term Reversal(단기 반전)과 Long-term Momentum(장기 모멘텀) 신호를 병렬적으로 생성하여 포트폴리오를 구성합니다.
- **인과적(Causal) 실행 리플레이**: 미래 편향(Look-ahead bias)을 엄격히 배제한 Point-In-Time (PIT) 마크 평가 및 5분봉 proxy 체결을 시뮬레이션합니다.
- **단일 진실 원천(Single Source of Truth) 원장**: Target weight 기반의 근사 PnL이 아닌, 체결 이벤트와 펀딩비, 마크-투-마켓(MTM) 가치 평가가 완전 통합된 수량/현금 원장(`SimulatedInventoryLedger`)을 통해 최종 PnL 및 리스크를 정밀 측정합니다.

> [!IMPORTANT]
> MHS의 **Research GO** 승인은 과거 데이터 기반의 연구 유효성 검증일 뿐이며, 실거래 배포(Execution GO), 소액 실전 테스트(Pilot GO), 또는 자본 확대(Scale GO) 승인과는 명확히 분리됩니다.

---

## 2. 전체 시스템 아키텍처 (System Architecture)

MHS는 신호 생성(1시간봉)과 체결 리플레이(5분봉)의 시간 격자를 분리하여 효율성과 체결 정밀도를 동시에 달성합니다.

```text
 1시간(1h) OHLCV + Funding + PIT Lifecycle
        │
        ├─ Trailing 720-bar quote volume ──► Liquid-Half Eligibility (유동성 상위 50%)
        ├─ Fast Reversal (48h) / Slow Momentum (72h~504h 앙상블) 북 생성
        ├─ Portfolio Rebalance Trigger (추적오차 20% 이상 시에만 수량 변경)
        └─ Market Beta Neutralization & Regime Tilt 적용
        │
        ▼ Decision 시각별 Top-30 PIT Execution Roster (Schmitt-Trigger 2.0x 적용)
        │
        ├─ 5분(5m) OHLCV high/low/close ──► Proxy Fill (Immediate-Taker / Strict Limit)
        ├─ Historical Mark Price Cache ──► Causal MTM Valuation & Funding Charge
        └─ Timestamped Fill Events ──────► Simulated Inventory Ledger (원장)
                                                │
                                                ▼
                                   3-Fold Level 2 Anchored Purged Validation
                                                │
                                                ▼
                                    Research GO 게이트 판정 및 결과 리포트
```

---

## 3. 데이터 및 Point-In-Time (PIT) 거버넌스

### 데이터 규격 및 PIT 규칙
1. **신호 패널**: `1h` OHLCV의 `open`, `close`, `quote_vol`을 사용하며 `2021-01-01`부터 시점 정렬을 수행합니다.
2. **유동성 필터 (Liquid-Half)**: 각 결정 시각마다 최근 720개 관측치(`720h`)의 quote volume 단면 중앙값(Median) 이상인 심볼만 거래 대상 자격(Eligible)을 가집니다.
3. **실행 로스터 (Execution Roster)**:
   - 유효 심볼 중 quote volume 상위 30개(`execution_universe_size=30`)를 실제 주문을 재생할 로스터로 선별합니다.
   - **Schmitt-Trigger 히스테리시스**: 랭킹 경계 부근에서 자산이 잦게 락인/탈락하며 발생하는 진동 매매를 방지하기 위해, 진입 30위 / 탈락 60위(`2.0x Exit Multiplier`)의 이중 임계값을 적용합니다.
4. **결손 데이터 엄격 배제 (Fail-Closed)**:
   - Binance API/Vision 아카이브 결손 심볼(`MHS_SOURCE_GAP_EXCLUDED_SYMBOLS`: `SLPUSDT`, `CTKUSDT` 등)은 사전 제외됩니다.
   - 보유 포지션이나 활성 주문에 데이터 갭이 발생하면 암묵적 0 패딩이나 추정을 하지 않고 해당 구간을 무효화(`INVALID_PRIMARY_LEDGER`)합니다.

### 실행 데이터 수집 (CLI)
```bash
# 5분봉 실행 데이터 manifest 생성 및 수집
PYTHONPATH=. uv run python -m src.cli.main data collect mhs-execution \
  --timeframe 5m --start 2021-01-01 --end 2025-12-31 --execute
```
*(관련 모듈: [mhs_execution_collection.py](file:///home/kth/crypto-pilot/src/application/data/mhs_execution_collection.py))*

---

## 4. 신호 산출 및 포트폴리오 구조 (Signal & Portfolio Construction)

MHS는 Fast Reversal 북과 Slow Momentum 북을 독립적으로 생성한 뒤 자본 배분을 수행합니다.

### 1) 신호 및 자본 배분 (Capital Blend Weight)
- **Fast Reversal**: 48시간 수익률 반전 신호 (Sign = -1)
- **Slow Momentum**: 168시간(기본) 모멘텀 신호 (Sign = +1)
- **최신 자본 배분 (`PHASE_1_BOOK_BLEND_WEIGHTS`)**:
  Discovery qualification 실측 데이터 검증 결과, Fast Reversal은 전 비용 구간에서 t-stat 조건(|t| >= 2.0)을 미달하여 **자본 비중 0.0 (0%)**, Slow Momentum에 **자본 비중 1.0 (100%)**을 배분합니다.

### 2) 알파 엔진 개선 사항 (Alpha Engine Enhancements)
기존 정적 랭크 방식의 한계를 극복하기 위해 아래 핵심 기법들이 배선되어 있습니다:

- **Portfolio Rebalance Trigger (RC-1)**:
  종목별 데드밴드는 달러 중립성과 단위 그로스(Unit Gross = 1.0)를 파괴하는 문제가 존재했습니다. 이를 해결하기 위해 포트폴리오 전체의 원웨이 추적 오차(Tracking Error)가 임계값(`0.20`, 20%) 이상 벌어질 때만 타겟 수량을 한 번에 업데이트하는 `portfolio_rebalance_trigger`를 적용합니다.
- **Horizon Ensemble (RC-2)**:
  단일 168시간 모멘텀 신호의 고분산 문제를 완화하기 위해 72h부터 504h까지 19개 호라이즌의 달러 중립 북을 동일 가중 평균하는 `horizon_ensemble` 모드를 지원합니다.
- **Market Beta Neutralization (RC-4)**:
  `causal_market_beta` (720바 롤링 OLS)를 산출하고 `beta_neutralize_weights`를 통해 포트폴리오가 시장 전체의 시장 베타(Market Directional Risk)에 노출되지 않도록 직교화(Orthogonalization)합니다.
- **Crash Regime Tilt Overlay**:
  BTCUSDT 단일 자산 바스켓의 인과적 추세 z-score 기반으로 크래시 레짐 시 시장 방향성 틸트를 보수적으로 혼합(`crash_regime_tilt_weights`)하는 기능을 지원합니다.

---

## 5. Historical Mark Price 및 체결 리플레이 (Execution Replay)

### Historical Mark Price
- Mark price는 신호, 랭킹, 체결 판정에는 영향을 주지 않으며, 오직 **Valuation(평가), MTM PnL, Funding Charge** 계산에만 사용됩니다.
- `cache_required` 모드: `data/futures/markPriceKlines/<timeframe>/<symbol>.parquet` 데이터를 사용하며 1시간 마크 캔들은 1시간 지연 후 Causal Forward-Fill합니다. Mark gap 발생 시 즉시 Fail-Closed 처리됩니다.

### Execution Proxy Bounds
실제 OHLCV 시뮬레이션 체결은 세 가지 경계로 산출됩니다:

1. **`OHLCV_IMMEDIATE_TAKER` (Primary Research GO)**:
   - 주문 생성 즉시 5분봉 Close 가격으로 Taker 체결을 가정합니다.
   - 전략의 분당 참여율(`participation_warnings`)이 1e-9 수준으로 무시 가능한 수준이므로, 시장 충격을 피하기 위해 Passive 대기를 수행할 경제적 근거가 없다는 연구 결과에 따라 **Primary 기준**으로 고정되었습니다.
2. **`OHLCV_STRICT_PROXY` (Patient Reference)**:
   - Limit intent가 이후 high/low를 관통(Trade-through)할 때만 Maker 체결로 인정하고, 30분 타임아웃 시 Taker Fallback을 적용합니다. (참고용 지표)
3. **`SPREAD_AND_COST_X3` (Cost Stress Bound)**:
   - Primary 체결과 동일한 Immediate-Taker 조건에서 비용만 **3배**(Maker 6.0bps, Taker fee 15.0bps, Slippage 9.0bps) 스트레스를 가하여 전략의 비용 내성을 검증합니다.

---

## 6. Simulated Inventory Ledger & 17차 연율화 수정

모든 PnL 및 리스크 지표의 단일 진실 원천은 [src/mhs/execution.py](file:///home/kth/crypto-pilot/src/mhs/execution.py)의 `simulated_inventory_ledger` 함수입니다.

### 타임스탬프별 처리 순서
1. **MTM Evaluation**: 직전 이벤트 이후 보유 수량을 마크 가격으로 마크-투-마켓 평가.
2. **Accrued Funding Charge**: 해당 구간 발생 펀딩비를 전 시점 수량 x 마크 가액 기준으로 차감/지급.
3. **Intent Netting**: Fast/Slow 및 리밸런스 신호를 실제 보유 수량 기준으로 상쇄(Netting).
4. **Proxy Fill Application**: 타임스탬프 순으로 체결 이벤트 및 수수료(Fee)를 반영하여 수량 및 현금(Cash) 갱신.
5. **Equity & Turnover Calculation**: Equity = Cash + (Units * Mark Price), Turnover = `sum(|qty * fill_price| / pre_trade_equity)`.

### 17차 연율화 버그 수정 (`mhs_execution_annualization_fix`)
- **원인**: 5분봉 격자(`execution_timeframe=5m`) 체결 원장에 1시간봉 연율화 상수(`_PERIODS_PER_YEAR_1H = 8760`)를 적용하던 집계 버그 존재.
- **수정**: 5분봉 격자에 맞춘 연율화 상수(`_PERIODS_PER_YEAR_5M = 365 * 24 * 12 = 105,120`)가 올바르게 적용되도록 수정.
- **효과**: Sharpe 비율(일봉 기반)이나 Research GO 승인 여부는 변함이 없으나, 연복리 수익률(CAGR), 연간 회전율, 꼬리위험(Bootstrapped MDD) 등 자산 증식 수치가 정확히 교정되었습니다 (CAGR 0.63% ──► **7.84%**).

### Two-Pass Replay & Strategy P&L Volatility Targeting
Primary 실행은 2단계(Two-Pass)로 진행됩니다:
- **1-Pass (`pre_vol_target_reference`)**: Unscaled 가중치로 1차 시뮬레이션을 실행하여 전략 자체의 일별 P&L 실현 변동성을 측정.
- **2-Pass (Primary)**: 측정된 P&L 변동성을 바탕으로 인과적 Volatility-Targeting(Barroso & Santa-Clara 2015) 스케일을 가중치에 적용하여 Momentum Crash를 방어한 뒤 최종 보고서를 작성합니다.

---

## 7. Research GO 검증 체계 및 최신 상태

### 3-Fold Level 2 Anchored Purged Validation
과적합 방지를 위해 3개 구간의 Anchored Fold에서 검증을 수행합니다:
- **Fold 0**: 2021~2022 Train ──► **2023 Validation** (168시간 Purge/Embargo)
- **Fold 1**: 2021~2023 Train ──► **2024 Validation** (168시간 Purge/Embargo)
- **Fold 2**: 2021~2024 Train ──► **2025 Validation** (168시간 Purge/Embargo)

### Research GO 5대 통과 조건
1. Pre-screen cost tiers (4.18bp base, 6.07bp stress) 결과 리포트 제출.
2. Immediate-Taker 원장의 Daily Autocorrelation-Adjusted Sharpe **≥ 0.6** 충족.
3. Cost-stressed (`SPREAD_AND_COST_X3`) Immediate-Taker Stress Sharpe **> 0** 충족.
4. Cap 30% Sharpe 및 양의 순 연환산 수익률 조건 충족.
5. Phase degeneracy, relevant missing data, termination, concentration 등 5개 진단 리포트 완비 (Silent exclusion 금지).

### 최신 실행 진단 결과 (17차 기준)

| 항목 | 16차 (연율화 버그) | **17차 (최신 수정)** | 비고 |
| :--- | ---: | ---: | :--- |
| `primary_autocorr_sharpe` | 0.5257 | **0.5257** | 0.6 미달 (GO 차단 원인) |
| `primary_naive_sharpe` | 0.1333 | **0.4616** | √12배 상승 교정 |
| `primary_net_ann` | 0.0081 | **0.0972** | 12배 상승 교정 |
| `primary_geometric_cagr` | 0.0063 | **0.0784 (7.84%)**| 12.4배 상승 교정 |
| `primary_max_drawdown` | -0.2269 | **-0.2269 (-22.69%)**| 무변화 |
| `primary_annualized_turnover`| 3.56 | **42.68** | 12배 상승 교정 |
| `stress_naive_sharpe` (x3 cost) | +0.0410 | **+0.1420** | 3.46배 상승 |
| **`research_go.eligible`** | `false` | **`false`** | **Research GO 기각** |

### Fold별 상세 (17차 실측치)
- **2023년 (Fold 0)**: Autocorr Sharpe **+0.8046**, CAGR **+9.36%**, Stress Sharpe +0.2111 (통과)
- **2024년 (Fold 1)**: Autocorr Sharpe **-0.2672**, CAGR **-4.99%**, Stress Sharpe -0.8201 (**미달 — 시장 반전장/붕괴장 손실**)
- **2025년 (Fold 2)**: Autocorr Sharpe **+1.5047**, CAGR **+48.22%**, Stress Sharpe +0.9310 (통과)

### 현재 상태 결론
- **구현 상태**: 실행 리플레이, 모의 원장, Volatility Targeting, 앙상블 및 리밸런스 트리거 코드 구현 **100% 완료 및 정상 동작**.
- **판정 상태**: 전략의 전체 Sharpe(0.5257 < 0.6) 및 2024년 Fold1 성능 미달로 인해 **Research GO 실패 (`eligible=false`)**.

---

## 8. 주요 코드 진입점 (Key Code Entry Points)

| 책임 / 역할 | 파일 경로 | 주요 함수 / 클래스 |
| :--- | :--- | :--- |
| **MHS 계약 & 파라미터** | [src/mhs/contracts.py](file:///home/kth/crypto-pilot/src/mhs/contracts.py) | `BookSpec`, `ExecutionSpec`, `PHASE_1_BOOK_BLEND_WEIGHTS` |
| **북 구성 & 리밸런스** | [src/mhs/books.py](file:///home/kth/crypto-pilot/src/mhs/books.py) | `rank_weight_book`, `portfolio_rebalance_trigger`, `equal_weight_book_ensemble` |
| **레짐 & 베타 직교화** | [src/mhs/regime.py](file:///home/kth/crypto-pilot/src/mhs/regime.py) | `causal_market_beta`, `beta_neutralize_weights`, `crash_regime_tilt_weights` |
| **체결 & 모의 원장** | [src/mhs/execution.py](file:///home/kth/crypto-pilot/src/mhs/execution.py) | `simulated_inventory_ledger`, `strategy_aware_execution_replay` |
| **평가 & Sharpe / Fold** | [src/mhs/evaluation.py](file:///home/kth/crypto-pilot/src/mhs/evaluation.py) | `autocorrelation_adjusted_sharpe`, `compute_deployment_readiness`, `phase_1_anchored_purged_folds` |
| **MHS Orchestration** | [src/application/research/mhs/evaluation.py](file:///home/kth/crypto-pilot/src/application/research/mhs/evaluation.py) | `MhsHorizonDiagnosticReport`, `run_mhs_horizon_diagnostic` |
| **5m/1m 데이터 수집** | [src/application/data/mhs_execution_collection.py](file:///home/kth/crypto-pilot/src/application/data/mhs_execution_collection.py) | `collect_mhs_execution_data` |
| **MHS CLI Command** | [src/cli/commands/research/mhs.py](file:///home/kth/crypto-pilot/src/cli/commands/research/mhs.py) | `mhs_horizon_diagnostic` |

