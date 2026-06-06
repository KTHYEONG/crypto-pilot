# Futures ML Strategy Architecture

> last_verified: 2026-06-06 (ablation v2 causal framework)

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
- **Next-Open Exit**: time exit는 `entry_idx + expected_holding_bars`의 open 가격을 사용 (close가 아님). Engine 진입 시점과 동일한 계약을 유지한다.
- **Cost-Aware Net Edge**: `net_event_bps = gross_event_bps - execution_cost_bps - realized_funding_bps - hurdle_bps`. taker 비용 + realized funding 기준.
- **Risk-Unit Normalization**: `z_i = net_event_bps / s_i`, `s_i = max(sl_thr_bps, min_risk_unit_bps)`. center/q10/q90 모델이 동일 stationary target을 학습한다.
- **Barrier Logic**:
  - `net_event_bps`: taker 비용 + funding 차감 후 실현 수익 (회귀 target)
  - `mae_r`: path MAE / s_i (risk-unit 하방 변동성)
  - `mfe_r`: path MFE / s_i (risk-unit 상방 잠재력)
  - `edge_after_hurdle_bps`: `net_event_bps`의 backward-compat alias

### 3.3 Multi-Group Feature Engineering (`candidate_dataset.py`)
학습 데이터셋은 세 가지 핵심 피처 그룹으로 구성됩니다.
- **Identity Features**: 전략 패밀리 및 변종 ID를 원-핫 인코딩하여 개별 로직의 고유 특성 반영.
- **Market State**: BTC 수익률, 추세, 전체 시장 변동성 및 분산, 시장 폭(Breadth) 등 거시 국면 정보.
- **Symbol State**: 개별 코인의 변동성 Z-score, 펀딩비 상태, 수익률 랭크 등 자산 고유 상태 정보.

### 3.4 Model Training & Calibration (`*_gate.py`, `*_edge.py`)
- **Calibrated Gate (Classifier)**: LightGBM으로 성공 확률(`p_pass`)을 예측. calibration-fit/eval 분리 적용. gate는 calibration incremental uplift가 양수일 때만 low-quality event veto로 작동하며, **`p_pass`를 `mu`에 곱하지 않는다** (이중계상 금지).
- **Risk-Unit Edge (Regressor)**:
  - **Prior Shrinkage**: calibration-set 가중 평균으로 variant prior `mu_prior_i = E[z_i]` 추정. global prior와 shrinkage 결합.
  - **Residual Champion**: ML residual feature 모델은 calibration eval에서 incremental LCB > 0일 때만 활성화. 비활성 시 prior-only로 fallback.
  - `mu_i = mu_prior_i + mu_residual_i` (residual champion pass 시). `expected_net_bps_i = mu_i * s_i`는 표시/검사 전용이며 raw model 출력이 아님.
  - **Multi-Objective**: risk-unit center(z), mae_r, mfe_r 별도 Quantile Regressor 학습.

### 3.5 ML Gate & Dynamic Selection (`candidate_portfolio.py`)
시그널이 포트폴리오에 최종 편입되기까지의 **4단계 동적 필터링** 과정입니다.

1.  **Stage 1: Pointwise filtering (Gate & Utility)**
    - `selection_scope=per_timestamp`: timestamp 내부에서만 후보를 정렬하고 선택.
    - `utility_score`는 production 기준에서 `expected_edge_direct`를 사용.
    - `cost_floor_bps` 및 `selection_max_events_per_bar`는 절대 제약으로만 작동.
    - Gate는 incremental uplift 검증 통과 시에만 low-quality veto로 작동.
2.  **Stage 2: Regime-Aware Scaling (Market Context)**
    - `CRISIS_GAMMA` 지표로 시장 위기 국면 감지, 롱/숏 진입 강도 동적 조절.
    - 하락장/위기 시 방어적 포지셔닝 적용.
3.  **Stage 3: Cross-Sectional Relative Filter (Alpha Selection)**
    - `CS_Z_SCORE_THRESHOLD`로 시장 평균 대비 독보적 엣지 보유 자산 선별.
4.  **Stage 4: Sizing**
    - **Stop-Risk Budget (기본)**: `abs(w_i) = min(max_symbol_weight, event_risk_budget / (s_i / 10_000))`. 진입 bar 종가 미사용 (look-ahead 제거).
    - **Calibrated Event Kelly (선택)**: `f_bin = clip(kelly_fraction * E[r|score_bin] / max(E[r²|score_bin], floor), 0, max_symbol_weight)`. `E[r]`, `E[r²]`, score-bin 경계는 calibration-fit에서만 추정하며 OOS에서 recalculate하지 않는다.
    - **`p_pass`는 sizing에 곱하지 않음**: gate와 sizing의 이중계상 금지.
    - **Variant Concentration Cap**: timestamp 내부에서만 적용.

### 3.6 Universe-to-ML Coupling
- **Metadata Propagation**: 유니버스의 정적 메타데이터(`vol_30d`, `friction_score`)가 ML 학습 피처와 진단 지표로 전달됨.
- **Diagnostics**: Bridge 단계에서 변동성 데실(Decile)별 생존율 등을 로깅하여 모델의 편향성 및 유니버스 적합성 모니터링.

## 4. 설계 원칙 (Design Principles)
- **Point-in-Time Integrity**: entry bar 종가 미사용, next-open exit, calibration-only score-bin moment 추정.
- **Fail-Closed Selection**: 모델의 확신이 낮거나 시장 리스크가 크면 자본을 투입하지 않음. threshold 완화로 통과 금지.
- **Single-Unit Contract**: label, ML target, sizing이 동일한 risk-unit(s_i) 기준으로 통일. bps/q10/mfe를 혼용하지 않음.
- **Incremental Uplift Gate**: residual/gate/Kelly 각각이 calibration eval에서 block-bootstrap lower bound > 0을 증명할 때만 활성화.
- **Workflow Status Contract**: bridge는 성공 시에도 `wf_eligible`까지만 출력하고, `deployment_promoted`는 별도 배포 게이트에서만 사용한다.

## 4.1 Causal Ablation Framework (v2)

`run_candidate_ablation()`은 6개의 causal variant로 ML 각 컴포넌트의 증분 가치를 인과적으로 분리한다.

| # | Variant | 설명 |
|---|---|---|
| 1 | `rule_stop_risk` | Rule signal만, stop-risk sizing |
| 2 | `prior_rank_stop_risk` | + variant prior 순위 필터 |
| 3 | `prior_residual_rank_stop_risk` | + ML residual edge 추가 |
| 4 | `edge_plus_validated_gate_stop_risk` | + validated gate veto |
| 5 | `edge_plus_gate_event_kelly` | + calibrated event Kelly (caps bypass) |
| 6 | `full_portfolio_caps` | + 최종 portfolio caps 적용 |

- 모든 variant는 동일 OOS bars, fold boundaries, candidate set, execution contract를 사용.
- `trade_count == len(engine_trades)` (delta weight proxy 미사용).
- `real_edge_bps_p50`: `(pnl - entry_fee) / (entry_price * amount) * 10_000` 의 중앙값.

## 5. 핵심 기술 스택
- **Engine**: `numpy`, `pandas`, `numba` (고속 벡터 연산)
- **ML**: `lightgbm` (Tabular 데이터 최적화), `scikit-learn` (검증 및 보정)
- **Optimization**: `optuna` (전략 파라미터 최적화)
- **Validation**: Purged Walk-forward, Bootstrap, DSR/PBO

---
*참고: 상세 구현 로직은 `src/domain/futures/strategy/` 및 `src/domain/futures/optimization/` 경로의 각 모듈을 참조하십시오.*
