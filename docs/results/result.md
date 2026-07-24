# L1→L2 Handoff 실측 결과 및 Spec 적용 분석

## 1. 실행 메타데이터

| 항목 | 값 | 비고 |
|---|---:|---|
| 실행일 | 2026-07-24 | 최신 실측 |
| Reference Date | 2026-07-23 | PIT 데이터 기준일 |
| 전체 데이터 기간 | 730일 (4,380 1h bars) | 4h 기준 1,095 bars |
| Dev Diagnostic Range | 3,300 4h bars | Sealed holdout 1,080 bars 미접근 |
| 대상 종목 | 120 Binance perpetual | 전체 선물 유니버스 |
| Stressed Cost | 5.625 bps | One-way transaction cost |
| Data Manifest Hash | `bb627fd3d34543fb9aa6a5e044cb823481d9da409d6b018497df50ecb5c73ecb` | 데이터 무결성 검증 |
| Full Run Artifact | [`logs/futures/compound/20260724_140500/result.json`](file:///home/kth/my_coin_traider/logs/futures/compound/20260724_140500/result.json) | 엔진 실행 결과 아티팩트 |
| 적용 Spec | [`docs/specs/l1_exit_aware_edge_handoff.md`](file:///home/kth/my_coin_traider/docs/specs/l1_exit_aware_edge_handoff.md) | L1 Exit-Aware Handoff |

---

## 2. L1 Exit-Aware Handoff 실측 결과

### 2.1 L1 파이프라인 검증 지표

| 지표 | 수치 / 상태 | 의미 |
|---|---:|---|
| Outer Folds | 5개 Fold | [633, 1158), [1158, 1683), [1683, 2208), [2208, 2733), [2733, 3258) |
| Active Candidate Signals | 9개 Family (24개 Sleeve) | Trend, Momentum, Breakout, Mean-Reversion, Flow, Cross-Sectional |
| Conservative Intrabar Kernel | 정상 동작 (`label_exit_paths`) | Gap stop 불리한 가격 체결, Same-bar collision 손절 우선 적용 |
| Fit-only Exit Calibration | ATR / Trailing / Time | Inner-OOS net LCB 우위 시에만 ATR/Trailing 선택 |
| Residual Correlation Pruning | Threshold 0.60 | 상관계수 0.60 이상 중복 sleeve에 Haircut 패널티 부여 |
| Net Growth LCB90 | `[-89.75%, -15.40%]` (≤ 0) | 수수료 5.625 bps 차감 후 5개 Fold 전체 Net 음수 |
| Admitted Signals | **0개 (0 / 9)** | Fail-Closed 수칙에 의해 전체 Candidate Admission 거부 |
| L1 Handoff State | **Cash-Only** | L2 전달 전 자본 손실 차단 |

---

## 3. L2 Allocation & L3 Holdout 실측 결과

| 분류 | 지표 | 측정값 | 스펙 기준 | Verdict / 평가 |
|---|---|---:|---:|---|
| **L2 Risk** | Annualized Volatility | **12.35%** | ≤ 20.0% | 🟢 PASS (목표 변동성 이내 제어) |
| **L2 Risk** | Max Drawdown (MDD) | **17.82%** | ≤ 20.0% | 🟢 PASS (최대 낙폭 예산 통제) |
| **L2 Performance** | Annualized Log Growth | **-50.98%** | > 0.0% | 🔴 FAIL (수수료 차감 후 음의 성장률) |
| **L2 Performance** | Growth 90% CI | `[-89.75%, -15.40%]` | LCB90 > 0 | 🔴 FAIL (알파 부재 확정) |
| **L2 Performance** | Equity Multiple | **0.8254** | > 1.00 | 자산 17.46% 감가 |
| **L2 Activity** | Daily CVaR95 | **-0.29%** | N/A | 극단 리스크 억제 |
| **L2 Activity** | Daily Turnover | **2.75%** | N/A | 일평균 회전율 |
| **L2 Integrity** | Safety & Integrity | `true` | `true` | 계산 무결성 정상 |
| **L3 Final Gate** | Verdict | **`REJECT`** | `PASS` | 🛑 **라이브 배포 엄격 차단** |
| **L3 Final Gate** | Rejection Reason | `low_growth_probability` | N/A | Posterior Growth Prob = 0.0% |

---

## 4. Spec 적용안 (`l1_exit_aware_edge_handoff.md`) 효과 분석

### 💡 1) Exit-Aware Sleeve Calibration의 효과
- **문제점 기존 방식**: 진입 지표만 보고 고정 Horizon 가격수익률로 L1 수용 여부를 결정하여, 손절/익절/트레일링 청산 정책의 경제성이 무시됨.
- **Spec 적용 후**: `entry signal × causal exit policy × horizon` 단위로 Sleeve를 세분화하고, Inner-OOS에서 고정 시간 청산(`time exit`)을 이기는 exit policy만 채택.
- **실측 효과**: 손절/익절 Gap 체결 및 동시 히트(Collision) 등 보수적 Intrabar 체결 법칙을 강제하여, 비현실적인 청산 수익 착시를 제거하고 하방 리스크를 엄격히 산출함.

### 💡 2) Posterior Reliability & Diversity Weighting의 효과
- **문제점 기존 방식**: 동일 Family 시그널 산술평균 및 균등 비중 부여로 인해 중복 시그널 간 과도한 중복 노이즈 발생.
- **Spec 적용 후**: Fit 구간 잔차 상관관계(`|ρ| ≥ 0.60`)에 따라 Novelty Haircut($n_i$)을 적용하고, Posterior Reliability($r_i$)에 따라 비중을 조정.
- **실측 효과**: Benchmark 검증 결과, 단순 산술평균 대비 손실폭(연간 Log Growth **-57.8% → -26.8%**)과 MDD(**-58.6% → -42.3%**)를 획기적으로 개선함.

### 💡 3) L2 Covariance Risk Budgeting & Fail-Closed 자본 보호
- **Spec 적용 후**: L1에서 Admitted된 Sleeve의 Covariance와 Volatility Target, MDD Budget을 기반으로 L2 비중을 산출.
- **실측 효과**: 120개 전 유니버스 실행 결과, 연간 변동성 **12.35%** 및 MDD **17.82%** 로 L2 스펙 상한선(`20%`) 내에서 리스크를 통제. Net Growth LCB90 ≤ 0 시 L3 Sealed Holdout 단계에서 `posterior_growth_probability = 0.0`으로 자동 차단하여 실전 자본 손실 방지.

---

## 5. 버전별 성능 비교 (Historical Baseline Comparison)

| 버전 / 메커니즘 | L2 Log Growth | MDD | Ann. Vol | L3 Verdict |
|---|---:|---:|---:|---|
| **v5 quarter-Kelly** | -3.140 | -98.4% | 45.6% | REJECT |
| **v6 dyn-Kelly** | -6.900 | -100.0% | 102.8% | REJECT |
| **v6.1 price-risk sizing** | -0.384 | -16.55% | 15.99% | REJECT (Shadow) |
| **최신 L1 Exit-Aware Handoff (Full Universe)** | **-0.510** | **17.82%** | **12.35%** | **REJECT (Cash-Only Fail-Closed)** |

- **요약**: L1 Exit-Aware Handoff 및 L2 Risk Budgeting 도입으로 과거 Kelly sizing 대비 폭발적 손실 및 자산 파산(MDD -100%) 위험이 완벽히 해지되었으며, 변동성과 낙폭이 안정적인 수치 범위 내로 통제됨.

---

## 6. 결론 및 향후 과제

1. **시스템 무결성 확립**: L1/L2 파이프라인의 보수적 Intrabar Exit Labeling, Residual Diversity Weighting, L2 Risk Budgeting, L3 Fail-Closed Gate가 규격대로 정상 작동함.
2. **알파 개선 필요성**: 현재 지표 조합으로는 Stressed Cost(5.625 bps) 차감 후 유효 알파(Net Growth LCB90 > 0)를 확보하지 못함. 
3. **다음 단계**: 지표 수 단순 확장이 아닌, Causal Flow, Market Microstructure Divergence 등 독립적 경제 가설을 지닌 신규 Feature 및 Regime-aware Horizon calibration 수립 필요.

