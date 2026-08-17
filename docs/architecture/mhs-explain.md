# Multi-Horizon Market State (MHS) 시스템 아키텍처 및 체결 가이드

## 1. 개요 및 주 목적 (Overview & Core Purpose)

**Multi-Horizon Market State (MHS)**는 여러 거래 보유 기간(Horizon)의 횡단면(Cross-Sectional) 알파 신호를 독립된 북(Book)으로 구축하고, 5분봉 단위의 실제 계약수량 기반 **모의 실행 원장(Simulated Inventory Ledger)**을 통해 전략의 유효성을 검증하는 **Phase 1 알파 연구 및 실행 파이프라인**입니다.

### 🎯 MHS의 주 목적
1. **Target Weight 근사 PnL의 착시 제거**: 목표 가중치(Target weight) 변경만으로 수익이 나는 것처럼 보이는 백테스트 착시를 막기 위해, 실제 주문 체결, 마크-투-마켓(MTM) 평가, 펀딩비, 슬리피지/수수료가 완전 통합된 수량/현금 원장(`SimulatedInventoryLedger`)으로 순수 PnL을 정밀 측정합니다.
2. **미래 편향(Look-ahead bias) 배제**: 매 결정 시점(Point-In-Time, PIT)에서 관측 가능한 유동성 및 마크 가격만을 인과적(Causal)으로 이용합니다.
3. **Kelly 및 동적 사이징의 배제 (Raw Alpha 검증)**:
   - MHS Phase 1 파이프라인에는 Fractional Kelly, 동적 레버리지 배율, 포트폴리오 사이징 로직이 **의도적으로 배제**되어 있습니다.
   - 이는 신호 자체의 순수한 우위(Edge)가 검증되지 않은 상태에서 사이징 로직을 얹어 백테스트 곡선을 왜곡하는 과적합(Overfitting)을 막고, **1.0x Gross 자본(자기자본 100%) 상에서 전략의 순수한 알파 생존력**을 쌩얼(Raw State)로 평가하기 위함입니다.

### 📌 단계별 파이프라인 (MHS의 위치)
- **Phase 1: Research GO (현재 MHS 단계)** 👈 **[현재 위치]**
  - 1.0x Gross 자본, Taker 체결/3배 비용 스트레스 조건에서 순수 알파 신호가 Sharpe ≥ 0.6 및 자본 보존을 달성하는지 검증
- **Phase 2: Execution GO**
  - L1/L2 오더북 데이터, 실시간 Limit/Taker 체결 딜레이 및 Market Impact 검증
- **Phase 3: Pilot GO / Scale GO**
  - Fractional Kelly, optimal leverage, 자금 배분(Sizing) 모델 적용 및 실전 자금 투입

---

## 2. 전체 시스템 아키텍처 (System Architecture)

MHS는 신호 생성(1시간봉)과 체결 리플레이(5분봉)의 시간 격자를 분리하여 효율성과 체결 정밀도를 동시에 달성합니다.

```text
 1시간(1h) OHLCV + Funding + PIT Lifecycle
        │
        ├─ 3단계 심볼 선정 (결손 제외 ──► 720h 유동성 상위 50% ──► Top-30 Roster & 히스테리시스)
        ├─ Fast Reversal (48h) / Slow Momentum (72h~504h 앙상블) 북 생성
        ├─ Portfolio Rebalance Trigger (추적오차 20% 이상 시에만 수량 변경)
        └─ Market Beta Neutralization & Regime Control (BTC 틸트 + P&L Vol Targeting)
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

## 3. 데이터 및 Point-In-Time (PIT) 심볼 선정 방식

MHS는 미래 데이터를 보고 종목을 고르는 편향을 막기 위해 매 매매 결정 시각(1시간마다) 3단계 필터링을 거쳐 거래 대상을 선별합니다.

```text
전체 바이낸스 선물 심볼 
   │
   ▼ [1단계] 데이터 결손 & 상장 이력 필터 (Source Gap Guard)
결손 심볼(SLP, CTK 등) & 미상장/상장 직후 자산 제거
   │
   ▼ [2단계] 유동성 반분 필터 (Liquid-Half Eligibility)
최근 30일(720시간) 거래대금 중앙값 이상인 유동성 상위 50% 추출
   │
   ▼ [3단계] 최종 실행 로스터 & 히스테리시스 (PIT Top-30 Roster + Schmitt-Trigger)
실제 5분봉 시뮬레이션 체결을 수행할 상위 30개 종목 확정 (진동 매매 방지)
```

### 🔍 3단계 심볼 선정 상세 흐름
1. **1단계: 데이터 결손 및 상장 이력 필터 (Source Gap Guard)**
   - Binance API/Vision 아카이브에 4시간 이상 결손이 발생한 심볼(`SLPUSDT`, `CTKUSDT`, `LITUSDT` 등 `MHS_SOURCE_GAP_EXCLUDED_SYMBOLS`)은 사전 제외됩니다.
   - 해당 시점에 새로 상장되어 최소 720시간 데이터가 쌓이지 않은 신규 코인도 제외합니다.
2. **2단계: 유동성 반분 필터 (Liquid-Half Eligibility)**
   - 매 결정 시각마다 최근 720시간(`720h`, 30일) 거래대금(Quote Volume)을 계산하고, 전체 코인 중 **거래대금 단면 중앙값(Median) 이상인 상위 50% 코인**만 1차 거래 자격 심볼(Eligible)로 표시합니다.
3. **3단계: 최종 실행 로스터 30개 선별 (Schmitt-Trigger 히스테리시스)**
   - 1차 유효 코인 중 거래대금이 가장 높은 **상위 30개 심볼 (`execution_universe_size=30`)**을 실제 주문을 재생할 로스터로 최종 선정합니다.
   - **Schmitt-Trigger 히스테리시스 (2.0x Exit Multiplier)**: 랭킹 경계 부근에서 자산이 잦게 락인/탈락하며 발생해 수수료를 갉아먹는 진동 매매(Churning)를 방지하기 위해, **진입은 상위 30위 이내**, **탈락은 60위(30 * 2.0) 밖으로 밀려날 때**만 로스터에서 빼는 이중 스위치를 적용합니다.

---

## 4. 신호 산출 및 호라이즌 탐색·조합 방식

### 1) Fast Reversal vs Slow Momentum 북 구조
- **Fast Reversal**: 48시간(`48h`) 수익률을 측정하여 오버슈팅된 코인을 매도(Short), 과도하게 떨어진 코인을 매수(Long)합니다 (`Sign = -1`).
- **Slow Momentum**: 72시간부터 504시간까지의 긴 기간 동안 오른 코인을 매수, 내린 코인을 매도합니다 (`Sign = +1`).

### 2) 호라이즌 탐색 및 디스커버리/퀄리피케이션 게이트 (Discovery / Qualification Gate)
- 호라이즌 후보군 선정 시, 단 한 해만 특출나게 수익을 낸 후보가 채택되는 과적합을 막기 위해 **Worst-Year Robustness 검증**을 거칩니다 (`select_horizon_by_discovery_qualification`).
- **Discovery 구간 (2021~2022년)**: 연도별 oriented net t-stat의 **최소값(Worst-year)**이 통계적 유의성 바닥(`|t| >= 2.0`)을 통과하는 호라이즌만 1차 선택합니다.
- **Qualification 구간 (2023년)**: 분리된 2023년 데이터에서 동일한 방향 부호와 `|t| >= 2.0` 조건을 재확인하여 최종 승인합니다. Fast Reversal은 이 게이트를 통과하지 못해 자본 비중 0%로 조정되었습니다.

### 3) 호라이즌 동일가중 앙상블 (Horizon Ensemble, RC-2)
- 백테스트에서 단 하나의 호라이즌(예: 딱 168시간)만 고르면(Argmax 선택), 특정 시점에는 잘 맞지만 시장 환경이 바뀌면 붕괴하는 고분산(High Variance) 문제가 발생합니다.
- 이를 해결하기 위해 Slow 모멘텀은 **72h부터 504h까지 24시간 간격의 19개 호라이즌**을 전부 계산한 뒤 **동일 가중 평균(`equal_weight_book_ensemble`)**합니다.
- 호라이즌 간 의견이 일치할 때만 포지션이 커지고, 의견이 갈리면 비중이 축소되어 예측 안정성이 향상됩니다.

### 4) 실측 데이터 기반 자본 배분 (Phase 1 Capital Blend)
- 디스커버리 게이트 실측 검증 결과에 따라 `PHASE_1_BOOK_BLEND_WEIGHTS`는 **Fast Reversal 자본 비중 0.0 (0%)**, **Slow Momentum 앙상블 자본 비중 1.0 (100%)**을 배분합니다.

### 5) 포트폴리오 제어 기법 (Alpha Engine)
- **Portfolio Rebalance Trigger (RC-1)**: 종목별 데드밴드가 달러중립성을 파괴하는 문제를 막기 위해, 포트폴리오 추적 오차가 임계값(`0.20`, 20%) 이상 벌어질 때만 타겟 수량을 한 번에 업데이트하는 `portfolio_rebalance_trigger`를 적용합니다.
- **Market Beta Neutralization (RC-4)**: `causal_market_beta` (720바 롤링 OLS)를 산출하고 `beta_neutralize_weights`를 통해 포트폴리오가 시장 전체의 방향성 위험에 노출되지 않도록 직교화(Orthogonalization)합니다.

---

## 5. 레짐(Regime)의 2중 관여 방식

횡단면 모멘텀 전략은 하락장 폭락 시 숏스퀴즈가 터지거나 자산 간 상관관계가 1로 수렴하며 큰 손실(Momentum Crash)을 보는 약점이 있습니다. MHS는 **2가지 축의 레짐 제어**로 이를 방어합니다.

### 1축: 참조 자산 추세 기반 크래시 레짐 틸트 (`crash_regime_tilt_weights`)
- **고정 참조 자산 (BTCUSDT)**: 심볼 구성 변화로 인한 착시를 막기 위해 시장 대표성이 가장 높은 `BTCUSDT` 단일 자산을 레짐 지표로 고정 사용합니다.
- **방향성 틸트 (Directional Tilt) 혼합**: BTC의 최근 추세 인과적 Z-score를 계산하여 시장 전체가 하락 크래시 레짐에 진입하면, 달러 중립 북에 **하락 방향성 틸트(`alpha` 비율만큼)**를 혼합합니다. 이는 하락장에서 숏 포지션 부담을 자연스럽게 완화(Offset)하여 **하락장 꼬리 위험(Tail Risk)을 방어**합니다.

### 2축: 전략 자체 P&L 변동성 타겟팅 (`Strategy P&L Volatility Targeting`)
- **전략 P&L 실현 변동성 추적**: 코인 개별 변동성이 아닌, **"MHS 전략 포트폴리오 자체의 최근 21일 일별 P&L 실현 변동성"**을 측정합니다.
- **동적 비중 축소 (Two-Pass Replay)**: 전략의 P&L 변동성이 장기 평균(Median)보다 급격히 튀면 모멘텀 붕괴 레짐으로 판단하여, `중앙값 변동성 / 최근 변동성` 비율만큼 전체 노출(Gross Exposure)을 인과적으로 축소(최대 0.2까지)하여 **Momentum Crash를 방어**합니다.

---

## 6. Historical Mark Price 및 체결 리플레이 (Execution Replay)

### Historical Mark Price
- Mark price는 신호, 랭킹, 체결 판정에는 영향을 주지 않으며, 오직 **Valuation(평가), MTM PnL, Funding Charge** 계산에만 사용됩니다.
- `cache_required` 모드: `markPriceKlines` 1시간 마크 캔들을 1시간 지연 후 Causal Forward-Fill하며, Mark gap 발생 시 즉시 Fail-Closed 처리됩니다.

### Execution Proxy Bounds
1. **`OHLCV_IMMEDIATE_TAKER` (Primary Research GO)**:
   - 주문 생성 즉시 5분봉 Close 가격으로 Taker 체결을 가정합니다.
   - 전략의 분당 참여율(`participation_warnings`)이 1e-9 수준으로 무시 가능한 수준이므로, 시장 충격을 피하기 위해 Passive 대기를 수행할 경제적 근거가 없다는 연구 결과에 따라 **Primary 기준**으로 고정되었습니다.
2. **`OHLCV_STRICT_PROXY` (Patient Reference)**:
   - Limit intent가 이후 high/low를 관통(Trade-through)할 때만 Maker 체결로 인정하고, 30분 타임아웃 시 Taker Fallback을 적용합니다. (참고용 지표)
3. **`SPREAD_AND_COST_X3` (Cost Stress Bound)**:
   - Primary 체결과 동일한 Immediate-Taker 조건에서 비용만 **3배**(Maker 6.0bps, Taker fee 15.0bps, Slippage 9.0bps) 스트레스를 가하여 전략의 비용 내성을 검증합니다.

---

## 7. Simulated Inventory Ledger & 17차 연율화 수정

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

---

## 8. 진단 테스팅 & Research GO 검증 체계

### 1) 9대 합성 스트레스 시나리오 (Synthetic Stress Scenarios)
MHS는 단순 과거 재현을 넘어 아래 9가지 결정론적 스트레스 상황을 독립적으로 검증합니다 (`synthetic_stress_scenarios`):
1. `BTC_DOWN_10`: 비트코인 10% 급락
2. `BTC_DOWN_20`: 비트코인 20% 급락
3. `ALT_BETA_UP`: 알트코인 베타 급증
4. `XS_CORRELATION_ONE`: 자산 간 횡단면 상관관계 1 수렴
5. `SPREAD_AND_COST_X3`: 스프레드 및 체결 비용 3배 폭등
6. `PASSIVE_FILL_DEGRADATION`: 지정가 체결률 저하
7. `FUNDING_EXTREME`: 펀딩비 극단적 치솟음
8. `LIQUIDITY_DETERIORATION_50PCT`: 50% 심볼 유동성 급감
9. `VENUE_API_OUTAGE_30M`: 30분간 거래소 API 장애

### 2) 꼬리 민감도 및 윈저화 (Tail Sensitivity & Winsor Curve)
- 특정 대형 이벤트가 전체 수익률을 착시시키지 않았는지 `tail_sensitivity_curve`로 진단합니다.
- 수익률 캡(50%, 30%, 20%, 10%) 윈저화(Winsorization) 손익 곡선, 상위 1개/5개/1% 이벤트 수익 기여도(`top1_event_share`), 최악 이벤트 제외 Sharpe(`leave_worst_event_out_sharpe`)를 측정합니다.

### 3) 배포 준비도(Deployment Readiness) 부트스트랩 검증
- **Stationary Block Bootstrap** (168시간/1주일 블록 크기, 2,000회 리플레이)을 통해 MDD 20% 초과 확률(`probability_mdd_over_20pct`), MDD 30% 초과 확률, 최종 자산 손실 확률 및 레버리지 파산 확률(Ruin Probabilities)을 추정합니다.

### 4) 3-Fold Level 2 Anchored Purged Validation 및 최신 결과
- **Fold 0**: 2021~2022 Train ──► **2023 Validation** (168시간 Purge/Embargo)
- **Fold 1**: 2021~2023 Train ──► **2024 Validation** (168시간 Purge/Embargo)
- **Fold 2**: 2021~2024 Train ──► **2025 Validation** (168시간 Purge/Embargo)

#### 최신 실행 진단 결과 (위원회 레짐 적응형 트랜치 기본값, 2026-08-17 실측)

`committee_capital=True` + `committee_regime_adaptive_tranche=True`(둘 다 CLI 기본값, ADR_20260817_MHS_COMMITTEE_REGIME_ADAPTIVE_TRANCHE)가 k=5 위원회 북 자신의 causal trailing lag-1 자기상관이 음수(whipsaw)인 결정행만 3행 트랜치로 평활하고 양수(추세지속)인 행은 raw를 채택해, 고정 평활 하나로는 항상 한쪽 레짐을 희생시키던 트레이드오프를 제거했다.

| 항목 | 17차 (레짐 적응형 이전) | **위원회 레짐 적응형 (최신)** | 비고 |
| :--- | ---: | ---: | :--- |
| `primary_autocorr_sharpe` | 0.5257 | **1.0792** | 0.6 플로어 여유 통과 |
| `primary_geometric_cagr` | 0.0784 (7.84%) | **0.1923 (19.23%)** | |
| `primary_max_drawdown` | -0.2269 (-22.69%) | **-0.1705 (-17.05%)** | |
| `deployment_readiness.calmar` | 0.35 | **1.128** | |
| `stress_naive_sharpe` (x3 cost) | +0.1420 | **+0.8334** | |
| `research_go.folds_passed` | 2/3 | **3/3** | |
| **`research_go.reason_codes`** | 알파 미달 + 데이터 결손 다수 | **`['UNSPECIFIED_POLICY']` 단일** | 정책 임계값 미등록만 남음 |

- **Fold별 실측**: 2023년 autocorr Sharpe 1.158(통과), 2024년 0.853(통과), 2025년 3.387(통과) — 세 폴드 모두 0.6 플로어를 여유 있게 상회.
- **최종 상태**: 알파/사이징 레벨의 병목은 해소되었고, Research GO의 유일한 차단 사유는 `MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`) 미등록에 따른 `UNSPECIFIED_POLICY`뿐 — 통계 게이트가 아닌 정책 등록 절차 이슈.
- P&L 변동성 타겟팅 비활성화(`--no-pnl-vol-target`) 실측 결과 2025-07-20 시점 `CAPITAL_INVARIANT_BREACH`(자본 완전 파괴)로 실패 — 현재 gross exposure 억제(평균 53%)는 인위적 제약이 아니라 실측 검증된 꼬리위험 방어선.

---

## 9. 주요 코드 진입점 (Key Code Entry Points)

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

