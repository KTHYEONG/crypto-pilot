# Futures ML Strategy Architecture

> last_verified: 2026-06-05

## 1. Overview
본 문서는 `my-coin-traider` 프로젝트의 선물(Futures) ML 전략 아키텍처를 기술합니다. 본 아키텍처는 단순 순위 기반(Rank-based) 모델에서 벗어나, 개별 **Candidate Event**를 추출하고 이를 ML로 필터링하여 최종 **Target Weight**를 생성하는 파이프라인을 핵심으로 합니다.

## 2. ML 파이프라인 (Pipeline)
```text
[Universe Filters] -> [Rule Signal Panels] -> [Candidate Event Extraction]
      -> [Leak-free Labeling] -> [Tabular Dataset Build]
      -> [Gate Model (Classifier)] -> [Edge/Downside Model (Regressor)]
      -> [Utility-based Selection] -> [Fractional Kelly Weights]
      -> [Backtest Engine] -> [Compound Evaluation]
```

## 3. 핵심 모듈별 역할

### 3.1 Signal & Event (`rule_signals.py`)
- **Signal Panel**: OHLCV 데이터를 기반으로 다양한 규칙(Trend, Breakout, Reversion 등)의 밀집(Dense) 시그널 생성.
- **Event Extraction**: 밀집 시그널에서 진입 조건이 충족된 시점을 희소(Sparse) 이벤트 행으로 변환.

### 3.2 Labeling (`candidate_labels.py`)
- **Triple Barrier**: Take-Profit, Stop-Loss, Time-Exit를 조합한 레이블링.
- **Cost-Aware**: `edge_after_hurdle_bps`와 같이 거래 비용(Base RT 7.5bps)을 차감한 실질 기대수익 레이블링.
- **Anti-Leakage**: 결정 시점(T) 이후의 정보 오염 방지 및 T+1 진입 원칙 준수.

### 3.3 Dataset & Models (`candidate_dataset.py`, `*_gate.py`, `*_edge.py`)
- **Feature Groups**: 시그널 강도, 심볼 상태(Vol, Funding), 시장 상태(BTC Trend, Breadth), 실행 비용, 그리고 Stage6 유니버스 메타데이터(`vol_30d`, `friction_score`, `alpha_capacity_score`, `diversification_score`, `tradeable_score`)를 포함합니다.
- **Risk Scale Feature**: `sl_thr_bps`를 ex-ante feature로 포함하여 q10 downside model이 자산별 stop distance scale을 직접 관측합니다.
- **Gate Model**: LightGBM Classifier를 이용한 거래 승인 여부 판정 (Calibrated Probability).
- **Edge Model**: LightGBM Regressor (Huber/Quantile)를 이용한 기대 수익 및 하방 리스크(q10) 추정.

### 3.4 Portfolio & Weights (`candidate_portfolio.py`)
- **Selection**: `p_pass`, `mu_net`, `q10_shortfall` 임계치를 통과한 이벤트 중 유틸리티가 가장 높은 후보 선택.
- **Shortfall Thresholding**: 기본값은 절대 bps 기준(`shortfall_threshold_basis="absolute_bps"`)이며, 필요 시 `stop_relative` 모드로 자산별 `sl_thr_bps` 기반 한계를 사용할 수 있습니다. 현재 기본값은 보수적으로 유지됩니다.
- **Sizing**: Fractional Kelly 기반 사이징 및 심볼/Net/Gross/Beta/Vol 캡 적용.
- **Bridge**: 최종 생성된 `target_weights`를 백테스트 엔진에 주입.

### 3.5 Universe-to-ML Coupling
- **Metadata Propagation**: Stage6 selection 결과의 정적 메타데이터는 `UniverseSnapshot -> data_maps -> AlignedMarketData -> CandidateDataset` 순서로 전달됩니다.
- **Diagnostics**: bridge 단계에서 `vol_30d` decile별 `mu_mean`, `q10_median`, `selected_pass_rate`를 로그로 남겨, 고변동성 유니버스가 downstream selection에서 어떻게 생존/탈락하는지 추적합니다.

## 4. 설계 원칙 (Design Principles)

### 4.1 데이터 무결성 (Integrity)
- **Point-in-Time**: 모든 피처는 결정 시점(T) 이전에 이용 가능한 데이터로만 구성.
- **Purge & Embargo**: Walk-forward 교차 검증 시 레이블 중첩 및 정보 유출 방지.

### 4.2 비용 모델 (Execution Cost)
- **Single Source of Truth**: `execution_cost.py`에서 정의된 비용 모델(Maker Ratio, Fee, Slippage)을 학습과 백테스트에 동일하게 적용.
- **Stress Testing**: 프로모션 게이트 통과를 위해 기본 비용의 1.5배(Stress RT)를 적용하여 검증.

### 4.3 검증 및 승격 (Validation & Promotion)
- **Nested Walk-Forward**: 훈련/검증/OOS(Out-of-Sample)를 분리한 순방향 검증.
- **Compound Gate**: 단순 IC/AUC가 아닌, CAGR, Max Drawdown, Log Growth 등 실제 자본 성장 지표를 기준으로 전략 승격.

## 5. 핵심 기술 스택
- **Base**: `numpy`, `pandas`, `numba` (벡터화 및 고속 연산)
- **ML**: `lightgbm` (표 형식 데이터 최적화), `scikit-learn` (검증 및 메트릭)
- **Validation**: Purged Walk-forward, Bootstrap, DSR/PBO

---
*참고: 상세 구현 로직은 `src/domain/futures/strategy/` 경로의 각 모듈을 참조하십시오.*
