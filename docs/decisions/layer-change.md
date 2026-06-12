---
title: 계층적 하이브리드 아키텍처로의 전환 (Tiered Hybrid Architecture)
domain: futures/strategy
type: adr
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/tiered_workflow.py
  - src/domain/futures/strategy/walk_forward.py
  - src/domain/futures/portfolio/signal_composer.py
  - src/domain/futures/strategy/cs_rank.py
  - src/domain/futures/portfolio/portfolio_constructor.py
  - src/domain/futures/optimization/workflow.py
  - src/execution/opt_main_futures.py
  - docs/architecture/allocation.md
last_verified: 2026-06-11
---

## [2026-06-12] Tiered aligned scope 교정 (Method B — Scope Mismatch Fix)
- **Delta:** `opt_main_futures.py:788` `align_data_maps` 인자를 `data_stage.valid_symbols(63)` → `effective_trade_syms = Stage6 OOS ∩ data_maps(12)`로 교체. `Layer1Result`에 `n_trade_scope: int = 0` 관측성 필드 추가. `tiered_logging.py` Valid Symbols/N 행 `n_trade_scope` 표시.
- **Rationale:** `breadth = n_valid / 63 = 0.168` 구조적 gate 미달(< 0.30) — 분모가 inference union(63) vs 신호 생성 범위(12) scope mismatch. bridge의 `effective_symbols = Stage6 ∩ data_maps`와 tiered의 `aligned.symbols`를 동일 범위로 일치시킴. 교정 후 breadth 0.168→0.883, coverage 0.0%→93.3%으로 두 gate 통과. 잔류 blocker: t-stat 1.64 < 1.96 (alpha 품질).
- **Trade-off:** aligned_tiered 63→12 축소로 CS feature 해상도 감소. 대신 breadth 분모의 의미론적 정확성 확보.

## [2026-06-11] Tiered Pipeline 첫 실행 진단 — CPCV Val IC 음수 확인

- **Delta:** `--phase signal` 실행으로 Tiered pipeline 진입 확인(`USE_CS_RANK_ENGINE=True` 로그). CPCV Fold 1: N=10,907, Global μ=23.43 bps, Val IC=-0.0043 (archetype_regime). Regime lift proof: passed=False, nw_tstat=-0.359 → archetype_only fallback. 이후 fold: Val IC=-0.051.
- **Rationale:** 기존 앙상블(B0) 병목과 동일 — **feature 예측력 부재(cross-sectional Rank IC≈0)**. Tiered 구조 자체는 정상 동작하나 입력 alpha가 없어 L1 gate 통과 불가 예상.
- **Edge Cases:** CPCV fold별 score_calibration+variant_prior 재피팅으로 10분+ 소요 (병목). 실행 완료 전 중단.

## [2026-06-11] USE_CS_RANK_ENGINE 기본값 True + 실행 통합 완료

- **Delta:** `OPT_FUTURES_CONFIG["USE_CS_RANK_ENGINE"] = True` (기본값 변경). tiered 분기를 bridge 호출 직후로 이동 — `ml_out.labeled` (Triple-Barrier 이벤트)를 `labeled_events`로 사용, `align_data_maps`로 `aligned` 생성. Phase D allocation 스킵 후 `return None`.
- **Rationale:** bridge labeling 인프라 재사용으로 독립 이벤트 생성 불필요. `--phase signal/alo/full` 기존 flag와 자동 매핑 (signal→L1, alo→L1+L2+L3, full→+최적화).
- **Edge Cases:** bridge 예외 시 `labeled=pd.DataFrame()` fallback; tiered 예외 시 Phase D fallback 보존.

## [2026-06-11] L3 CAGR 실측 계산 수정 (C1 결함)

- **Delta:** `run_l3_holdout`의 `vol_proxy=0.01` magic number + Sharpe 역산 제거 → `_run_awf_simulation` 공용 헬퍼 + `_cagr((1+Σr)^(bpy/n)−1)` 실측 계산. `Layer2Result` 인터페이스 불변.
- **Rationale:** 가짜 CAGR이 result.md에 표출되면 [[project_candidate_ml_promoted_2026_06_06]] 허위 양성 재발 — 실측 수익률 없는 metric은 reporting 금지.
- **Edge Cases:** `base≤0` → -1.0 (total loss); `signal_total=0` → 0.0; N2 friction_pass_pct 분모 `len(selected)` 수정.

## [2026-06-11] 3-Layer Tiered Hybrid Architecture 구현 완료

- **Delta:** `tiered_workflow.py`(814L) 신규 — `run_l1_cpcv`/`run_l2_awf`/`run_l3_holdout`/`run_tiered_pipeline`. `walk_forward.py`에 `CPCVFold`+`build_cpcv_folds`. `signal_composer.py`에 HAC t-stat+SymbolSignal adapter. `cs_rank.py`에 BTC-β wiring. `portfolio_constructor.py` friction mask `abs(mu)>=hurdle` (shorts 지원). `workflow.py`에 decoupled Optuna objectives.
- **Rationale:** Phase D(Ensemble B0) pass_ratio=0 병목 — CS 랭킹+Diagonal Kelly parallel seam으로 alpha 재설계 경로 확보. `USE_CS_RANK_ENGINE=False` 기본으로 Phase D 완전 보존.
- **Edge Cases:** Phase D fallback on exception; CPCV degenerate → single fold; REGIME_FLOOR clamp warning; DataStageResult에 aligned_market_data 미주입 → 현재 dormant (설계 의도).

## [2026-06-11] 개별 심볼 동적 시그널 및 글로벌 랭킹 기반 하이브리드 아키텍처 설계

### 1. Context: 현 시스템의 한계 (The "Ensemble B0" Problem)
현재의 `Ensemble B0` (Regime-Conditional Shrinkage) 방식은 다음과 같은 구조적 한계로 인해 복리 자산 증식(Compound Growth)에 제약이 있음:
- **Stationarity Bias:** 과거 특정 국면의 평균 수익률이 미래에도 반복될 것이라는 정적 가정에 의존하여 패러다임 변화에 취약함.
- **Point-wise Evaluation:** 20개 유니버스 종목 간의 상대적 우위를 무시하고 개별 종목의 절대 수익률만 예측하여 시장 노이즈(Market Beta)에 노출됨.
- **Binary Admission:** 베이지안 사후 확률 기반의 컷오프 방식이 포트폴리오의 회전율을 높이고 비연속적인 비중 변화를 유발함.

### 2. Decision: 2계층 하이브리드 아키텍처 도입 (Tiered Hybrid Design)

알파 창출의 유연성과 포트폴리오 리스크 통제의 안정성을 결합하기 위해 시스템을 두 개의 레이어로 분리함:

#### **Layer 1: Bottom-Up Symbol-Specific Dynamic Signals**
- **역할:** 각 심볼의 특이성(Idiosyncrasy)을 포착하여 고유한 시그널 스코어를 생성하고 신뢰성을 보증.
- **메커니즘:** 
    1. **Dynamic Selection:** 심볼별로 가장 적합한 전략을 동적으로 선택/가중치 부여하여 `Raw Score` 도출.
    2. **Reliability Quality Control (QC):** 통계적 유의성(t-stat) 및 최소 샘플 수(min_obs)를 검증하여 노이즈와 유효 신호를 구별.
    3. **Standardized Interface & Safety:** 
        - 모든 출력을 **`[Raw Mu, Volatility]` 튜플 형태**로 규격화.
        - **Zero-Division Guard:** 수식 폭발 방지를 위해 `Volatility`에 최소 하한선(Epsilon, 예: 1e-6)을 강제 적용.
        - **Strict Point-in-Time:** 모든 시그널은 현재 캔들 종가 시점의 가용 정보만을 사용하며, 미래 정보 참조(Look-ahead bias)를 원천 차단.

#### **Layer 2: Top-Down Global Ranking & Capital Allocation**
- **역할:** 개별 심볼들이 제출한 '검증된' 시그널을 통합 관리하고 상대 평가를 통해 자본을 배분.
- **메커니즘:** 
    1. **Breadth-based Risk-Off with Hysteresis:** 
        - 유효 신호 종목 수가 임계치 미만일 경우 현금화하되, 잦은 진출입(Whipsaw) 방지를 위해 비대칭 임계치(예: 진입 20%, 탈출 15%) 또는 최소 유지 기간(Cooldown)을 적용.
    2. **Cross-sectional Ranking with Diversity:** 
        - 제출된 `Raw Mu / Volatility` (Sharpe Ratio)를 기반으로 Z-Scoring 수행.
        - **Sector Cap 필터:** 동일 섹터 내 종목이 Top-K를 독점하지 않도록 순위 가중치 또는 선발 수 제한을 적용하여 기회비용 최소화 및 분산 확보.
    3. **Top-K Selection:** 최종 보정된 랭킹을 기반으로 상위 $K$개 종목 선발 (Selection).

### 3. Rationale (의사결정 근거)
- **Garbage In, Garbage Out 방지:** Layer 1에서 통계적 신뢰성을 선제적으로 검증(QC)하여, 랭킹 시스템이 무작위 노이즈에 속아 자본을 배분하는 위험을 원천 차단함.
- **데이터 무결성:** Interface를 튜플화하여 랭킹(상대평가)에는 위험조정수익률을, 사이징(절대평가)에는 원본 기댓값을 사용하여 논리적 일관성 유지.
- **자본 효율성:** 절대 평가 방식의 '입장 컷'을 완화하여 신뢰할 수 있는 시그널들이 랭킹 경쟁에 참여하게 함으로써 시장 기회 포착 능력 극대화.

### 4. Sizing & Portfolio Simplification (사이징 및 제약 조건 간소화)

#### **제거/완화 대상 (과잉 방어 제거)**
- **Breakeven Hard Gate (절대 수익성 게이트):** 랭킹 단계에서는 제거하고, 최종 사이징 단계의 Friction Filter로 역할 전환.
- **Ledoit-Wolf & Black-Litterman:** 높은 상관관계와 노이즈가 많은 코인 시장에서 행렬 역산에 의한 오차 확대를 방지하기 위해 폐기하고 대각 행렬(Diagonal) 방식으로 전환.

#### **필수 유지 대상 (복리 증식의 수학적 기초)**
- **Top-K Selection:** 최상위 상대적 엣지에만 자본을 집중하여 기회비용 최적화.
- **Magnitude-Aware Sizing with Friction Filter:** 
    - 최종 비중 산출 시에는 Layer 1에서 받은 `Raw Mu`를 복원 적용하여 신호 강도 보존.
    - **Friction Filter:** 복원된 Mu가 마찰 비용(수수료+슬리피지 허들) 미만일 경우 비중을 0으로 강제하여 확정 손실 방지.
- **Diagonal Kelly ($w_i \propto \mu_i / \sigma_i^2$):** 개별 종목의 기댓값과 변동성을 동시에 고려하여 기하 수익률(복리) 극대화.
- **Per-Symbol & Sector Exposure Guard:** 특정 자산 및 클러스터의 블랙스완 리스크 방어를 위한 최종 비중 상한선(Cap) 적용.
- **Gross Cap (Total Leverage):** 거래소 강제 청산 방지를 위한 물리적 안전 퓨즈.

#### **Execution Translation (Binance Mapping)**
이론적 비중과 실제 거래소 체결 사이의 오차를 방지하기 위한 실행 매핑 원칙:
- **Notional Value Based Ordering:** `Target Notional Value = Total Equity * Target Weight` 수식을 사용하여 산출된 코인 수량(Quantity) 기반으로 주문. 레버리지 설정(Set Leverage)을 비중 조절 수단으로 사용하지 않음.
- **Unified Margin Mode:** 바이낸스 **Cross Margin (교차 마진)** 모드를 기본으로 하며, 개별 심볼 레버리지는 주문 전 충분히 높은 값(예: 10x~20x)으로 사전 고정하여 마진 확보 권리만 유지함.

### 5. Consequences (예상 영향)
- **Positive:** 시장 급변 시 상대적으로 강한 종목으로 자본이 즉각 이동함. 포트폴리오의 Drawdown 방어력이 개선되고 복리 성장의 천장이 높아짐.
- **Negative:** 데이터 구조의 튜플화 및 횡단면 정규화 로직 추가로 인해 구현 복잡도가 상승함.

### 6. Ranking Algorithm Strategy (추천 알고리즘)

#### **Phase 1: Cross-Sectional Z-Scoring (즉각적 도입)**
- **목적:** 20개 심볼 봇의 Raw 스코어를 동일한 횡단면 기준으로 평준화.
- **방법:** 특정 타임스탬프의 모든 Sharpe Ratio 집합을 평균 0, 표준편차 1로 정규화하여 '상대적 우위' 추출.

#### **Phase 2: LightGBM Ranker - LambdaMART (고도화)**
- **목적:** 종목 간의 단순 순위를 넘어, 비선형적인 상대적 경쟁 우위를 학습.
- **방법:** 미래 수익률 순위를 Target으로 하여 '어떤 조건에서 어떤 종목이 1등을 할 확률이 높은가'를 학습.

### 7. Optimization Strategy (Decoupled Optuna Tuning)

과적합(Overfitting)을 방지하고 각 계층의 전문성을 극대화하기 위해 최적화(Optuna)의 목적과 대상을 철저히 분리하여 운영함:

#### **Optuna Phase 1: Alpha Optimization (Layer 1)**
- **목적:** 시그널의 예측력(IC) 및 신호 발생 빈도(Breadth) 극대화.
- **검증 기법:** **CPCV (Combinatorial Purged Cross-Validation)**. 시간을 쪼개고 섞어 다양한 시장 국면에서 알파의 강건성(Robustness)을 평가.
- **주요 튜닝 대상:** 룩백 윈도우, 노이즈 필터링 임계치, 모델 하이퍼파라미터 등.

#### **Optuna Phase 2: Allocation Optimization (Layer 2)**
- **목적:** 포트폴리오 위험조정수익률(Sharpe Ratio) 및 복리 성장성 극대화.
- **검증 기법:** **AWF (Anchored Walk-Forward)**. 자본의 연속성과 경로 의존성(Path-dependency)을 보존하기 위해 시간 순차적 전진 검증 수행.
- **OOS Stacking 원칙:** 데이터 누수(Data Leakage) 방지를 위해, Layer 1이 학습하지 않은 구간(Out-of-Sample)의 결과물만을 Layer 2의 입력값으로 사용하여 최적화.
- **주요 튜닝 대상:** Top-K 개수, 켈리 분수, 섹터 노출 한도, 마찰 비용 허들 등.

#### **Layer 3 Validation: Pure Event-driven Backtest**
- **목적:** Phase 1, 2에서 최적화된 파라미터를 고정(Frozen)하여 실제 물리적 거래 환경에서의 생존력 최종 검증.
- **원칙:** 최적화 과정에서 완전히 격리된 **Hold-out 데이터 세트**에서 단 1회의 테스트만 수행. **이 단계에서는 Optuna를 절대 사용하지 않음.**

#### **Data Splitting & Regime Awareness (시계열 데이터 격리 정책)**
코인 시장의 빠른 패러다임 변화(Regime Shift)와 알파 부패를 고려하여 다음과 같은 데이터 윈도우 규칙을 강제함:
- **Regime Filtering:** 마이크로스트럭처가 현재와 판이하게 다른 **2022년 이전(FTX 사태 이전) 데이터는 최적화 및 학습에서 원칙적으로 배제**하여 노이즈 유입을 차단함.
- **Sliding Window 구조:** 고정된 날짜가 아닌 상대적 기간(Duration)을 기준으로 훈련 세트를 구성함.
    1.  **Layer 1 (18 Months):** 최근 시장 구조를 학습하기 위한 18개월간의 CPCV 훈련 구간.
    2.  **Layer 2 (12 Months):** Layer 1의 학습 종료 시점 이후 이어지는 12개월간의 AWF 배분 최적화 구간. (반드시 Layer 1의 OOS 결과물만 사용)
    3.  **Hold-out (Recent 6 Months):** 최적화 완료 후 최신 6개월간의 순수 블라인드 테스트 구간.
- **Cycle Consistency:** 실전 운용 시, 이 윈도우 구성을 유지하며 일정 주기(예: 분기별)로 전체 윈도우를 전진(Sliding)시켜 모델의 최신성을 유지함.

### 8. Validation Criteria (계층별 유효성 검증 및 Pass/Fail 기준)

백테스팅 및 실전 투입 전, 각 계층이 독립적으로 제 역할을 수행하는지 검증하기 위한 정량적 지표를 다음과 같이 정의함:

#### **Layer 1: Signal Quality & Breadth (알파의 신뢰성)**
- **Information Coefficient (IC):** 제출된 `Raw Mu`와 실제 미래 수익률 간의 Spearman Rank Correlation을 측정.
    - **Pass:** Mean IC >= 0.02 및 IC t-stat >= 1.96 달성 시.
- **Signal Breadth (생존율):** Reliability QC를 통과하여 유효 신호를 생성하는 빈도 측정.
    - **Pass:** 개별 심볼 기준 유효 신호 생성 빈도 >= 30%, 전체 타임스텝 중 유효 종목수가 임계치(예: 4개) 이상인 비율 >= 80% 달성 시.

#### **Layer 2: Allocation Efficiency & Risk Defense (배분의 효용성)**
- **Friction Filter Pass Rate:** Top-K로 선정된 종목 중 실제 비중(Sizing > 0)을 할당받는 비율 측정.
    - **Pass:** Top-K 종목 중 Friction Filter를 통과하여 실제 집행되는 비율 >= 40% 달성 시.
- **Portfolio Value-Add:** 동일비중(1/N) 전략 대비 하이브리드 배분 로직의 성과 비교.
    - **Pass:** 동일비중 대비 포트폴리오 Sharpe Ratio >= 20% 향상 및 MDD(최대 낙폭) 유의미하게 축소 시.

### 9. Next Step
- 랭킹 스코어를 켈리 공식의 기대수익률($\mu$)로 매핑하는 `Rank-to-Mu Scaling` 함수 설계.
- `candidate_workflow.py` 내에 `[Raw Mu, Volatility]` 반환 및 횡단면 Z-Score 변환 로직 프로토타입 구현.
- 위 가드레일 및 검증 기준을 반영한 `portfolio_optimizer.py` 제약 조건 절 업데이트.
- Phase ALO 단계에 위 Pass/Fail 검증 로직을 통합하여 백테스트 전 자동 필터링 체계 구축.

### 10. [2026-06-12] Layer1 IC Diagnostics & Statistics Fix (ADR)
- **Delta:** Cross-sectional IC 산출 로직을 단일 Fold OOS 이벤트 기준의 Time-series Rank IC로 재정의하였으며, 인덱스 정렬 버그를 `FoldDiagnostic` 객체 통합으로 완전히 해결함.
- **Rationale:** 횡단면 분산이 극도로 제한적인 조건에서 노이즈가 제거된 정직한 시계열 IC를 확보하고, CPCV 중첩 보정(`n_eff` 도입 및 `ddof=1` 표준편차 반영)으로 통계적 유의성(t-stat)을 복원함.
