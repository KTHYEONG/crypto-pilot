# Futures ML Strategy Architecture

> last_verified: 2026-06-06

## 1. Overview
본 문서는 `my-coin-traider` 프로젝트의 선물(Futures) ML 전략 아키텍처를 기술합니다. 본 아키텍처는 단순 순위 기반(Rank-based) 모델에서 벗어나, 개별 **Candidate Event**를 추출하고 이를 ML로 필터링하여 최종 **Target Weight**를 생성하는 파이프라인을 핵심으로 합니다.

## 2. ML 파이프라인 (Lifecycle)
```text
[Universe Selection] -> [Vectorized Signals] -> [Sparse Event Extraction]
      -> [Triple-Barrier Labeling] -> [Feature Engineering: Identity/Mkt/Symbol]
      -> [Model Training: Calibrated Gate + Shrunk Edge/Downside]
      -> [Inference & Regime Scaling] -> [Cross-Sectional Alpha Selection]
      -> [Top-K Sparsification] -> [Portfolio Sizing]
```

## 3. 핵심 모듈별 상세 역할

### 3.1 Vectorized Signals & Event Extraction (`rule_signals.py`)
- **Vectorized Indicators**: `numba`와 `numpy`를 활용한 고속 벡터화 연산으로 EMA, Rolling Mean/Std, Log Return 등을 심볼별/타임스탬프별로 동시 계산.
- **Rule Signal Panels**: 정의된 전략 로직(Trend, Reversion 등)을 적용하여 밀집(Dense) 시그널 생성.
- **Sparse Event Extraction**: 시그널 문턱값을 넘는 진입 시점만 `Candidate Event`로 추출. 이때 사전에 통계적 유의미성(`KEEP`)이 검증된 전략 변종(Variant)만 필터링하여 노이즈 최소화.

### 3.2 Leak-free Triple-Barrier Labeling (`candidate_labels.py`)
- **Dynamic Barriers**: 진입 시점의 ATR을 기준으로 익절(TP), 손절(SL), 시간 제한(Time-Exit) 장벽을 동적으로 설정.
- **Cost-Aware Edge**: `edge_after_hurdle_bps` 레이블링 시 진입/청산 슬리피지 및 수수료(Hurdle)를 선제적으로 차감하여 실질 수익성 학습 유도.
- **Barrier Logic**: 
  - `raw_barrier_label`: TP 도달 여부 (단순 분류)
  - `edge_bps`: 청산 시점까지의 실질 실현 수익 (회귀)
  - `q10_bps`: 보유 기간 중 발생한 최악의 하방 변동성 (리스크 예측)

### 3.3 Multi-Group Feature Engineering (`candidate_dataset.py`)
학습 데이터셋은 세 가지 핵심 피처 그룹으로 구성됩니다.
- **Identity Features**: 전략 패밀리 및 변종 ID를 원-핫 인코딩하여 개별 로직의 고유 특성 반영.
- **Market State**: BTC 수익률, 추세, 전체 시장 변동성 및 분산, 시장 폭(Breadth) 등 거시 국면 정보.
- **Symbol State**: 개별 코인의 변동성 Z-score, 펀딩비 상태, 수익률 랭크 등 자산 고유 상태 정보.

### 3.4 Model Training & Calibration (`*_gate.py`, `*_edge.py`)
- **Calibrated Gate (Classifier)**: LightGBM을 사용하여 성공 확률(`p_pass`)을 예측. 학습은 `train / early_stop / calibration` 3분할로 수행하고, `CalibratedClassifierCV`는 calibration 구간에만 적합한다.
- **Shrunk Edge (Regressor)**: 
  - **Prior Shrinkage**: 개별 변종의 기대 수익을 글로벌 평균과 가중 결합(Shrinkage)하여 샘플이 적은 전략의 예측 불안정성 해소.
  - **Prior Deviation Clipping**: 변종 prior를 `global_prior ± edge_prior_max_deviation_bps`로 클리핑하여 IS over-confidence를 억제하고 ML residual(`center_pred`)의 상대적 영향력을 보존. (`mu_net = center_pred + prior`이므로 prior 폭주 시 ML 신호가 무력화되는 것을 방지)
  - **Multi-Objective**: 기대 수익(`mu`)뿐만 아니라 하방 리스크(`q10`) 및 상방 잠재력(`mfe`)을 별도의 Quantile Regressor로 학습.

### 3.5 ML Gate & Dynamic Selection (`candidate_portfolio.py`)
시그널이 포트폴리오에 최종 편입되기까지의 **4단계 동적 필터링** 과정입니다.

1.  **Stage 1: Pointwise filtering (Gate & Utility)**
    - 현재 구현은 `selection_scope=per_timestamp` 이며, 각 timestamp 내부에서만 후보를 정렬하고 선택한다.
    - `utility_score`는 production 기준에서 `expected_edge_direct`를 사용하며, OOS fold 전체 분위수로 percentile gate를 만들지 않는다.
    - `cost_floor_bps` 및 `selection_max_events_per_bar`는 절대 제약으로만 작동한다.
2.  **Stage 2: Regime-Aware Scaling (Market Context)**
    - `CRISIS_GAMMA` 지표로 시장 위기 국면을 감지하여 롱/숏 진입 강도를 동적으로 조절.
    - 하락장/위기 시 `DYNAMIC_RA_CRISIS_COEF`를 적용하여 방어적 포지셔닝 수행.
3.  **Stage 3: Cross-Sectional Relative Filter (Alpha Selection)**
    - `CS_Z_SCORE_THRESHOLD`를 사용하여 시장 전체 평균 대비 독보적인 엣지를 가진 자산 선별.
    - 동일 국면 내에서 상대적으로 강한 시그널에 자본 집중.
4.  **Stage 4: Top-K Sparsification & Sizing**
    - `K_RANK` 제한을 통해 가장 점수가 높은 최정예 후보군만 최종 선택.
    - **Variant Concentration Cap**: 단일 `family:variant`가 top-k의 `max_variant_selection_fraction` 이상을 점유하지 못하도록 제한하여 특정 변종 독점(concentration risk) 방지. cap은 fold 전체가 아니라 timestamp 내부에 적용한다.
    - `p_pass` 할인 켈리(Kelly) 베팅과 `q10` 기반의 리스크 분산 제약으로 최종 `Target Weight` 산출.

### 3.6 Universe-to-ML Coupling
- **Metadata Propagation**: 유니버스의 정적 메타데이터(`vol_30d`, `friction_score`)가 ML 학습 피처와 진단 지표로 전달됨.
- **Diagnostics**: Bridge 단계에서 변동성 데실(Decile)별 생존율 등을 로깅하여 모델의 편향성 및 유니버스 적합성 모니터링.

## 4. 설계 원칙 (Design Principles)
- **Point-in-Time Integrity**: 모든 결정은 T시점 이전의 데이터로만 수행 (No Look-ahead).
- **Fail-Closed Selection**: 모델의 확신이 낮거나 시장 리스크가 크면 자본을 투입하지 않음.
- **Risk-Reward Asymmetry**: 단순히 수익만 쫓지 않고, 예측된 하방 변동성(`q10`)을 sizing의 핵심 제약 조건으로 활용.
- **Workflow Status Contract**: bridge는 성공 시에도 `wf_eligible`까지만 출력하고, `deployment_promoted`는 별도 배포 게이트에서만 사용한다.

## 5. 핵심 기술 스택
- **Engine**: `numpy`, `pandas`, `numba` (고속 벡터 연산)
- **ML**: `lightgbm` (Tabular 데이터 최적화), `scikit-learn` (검증 및 보정)
- **Optimization**: `optuna` (전략 파라미터 최적화)
- **Validation**: Purged Walk-forward, Bootstrap, DSR/PBO

---
*참고: 상세 구현 로직은 `src/domain/futures/strategy/` 및 `src/domain/futures/optimization/` 경로의 각 모듈을 참조하십시오.*
