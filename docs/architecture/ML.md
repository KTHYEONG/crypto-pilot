---
title: Futures ML Pipeline Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/candidate_dataset.py
  - src/domain/futures/strategy/candidate_edge.py
  - src/domain/futures/strategy/candidate_labels.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/candidate_workflow.py
change_triggers:
  - src/domain/futures/strategy/candidate_*.py
last_verified: 2026-06-08
---

# 1. Overview

선물(Futures) ML 파이프라인(`ML.md`)은 시그널 모듈에서 생성된 **Candidate Event(이벤트 후보)** 들을 받아, 이들이 실제로 시장에서 알파(Edge)를 창출할 수 있는지 예측(Scoring)하고 검증(Gating)하여 최종 배분 가중치(Target Weight)를 결정하는 역할을 합니다. 순위 기반(Rank-based) 모델이 아닌 개별 이벤트에 대한 **Risk-Unit Edge Regressor**를 중심 축으로 삼아 Fail-Closed 철학을 실현합니다.

---

# 2. Core Components

| Component | 책임 | 파일 |
|-----------|------|------|
| Triple-Barrier Labeling | 진입 시점 대비 변동성(Yang-Zhang)을 활용한 슬리피지/펀딩비용 반영 실현 수익(Net Event Bps) 타겟팅 | `candidate_labels.py` |
| Feature Engineering | 정체성(Identity), 시장(Market), 개별 코인(Symbol), 신호 맥락(Signal Context) 특성 추출 및 Matrix 생성 | `candidate_dataset.py` |
| Edge Model & Calibration | Prior Shrinkage 및 LightGBM을 통한 Multi-Objective(Risk-unit center, MAE, MFE, Q10) Quantile 회귀 학습 | `candidate_edge.py` |
| ML Gate (Validation) | `Rank IC` 검증을 통한 실질 엣지 창출 여부 판별 (미달 시 Prior Fallback 또는 Disabled) | `candidate_edge.py` / `candidate_gate.py` |
| Portfolio Sizing | Pointwise Utility 필터링, Regime Overlay 결합, Kelly/Stop-Risk 기반 최종 Target Weight 산출 | `candidate_portfolio.py` |
| Workflow Orchestrator | Purged Walk-Forward 방식을 통한 학습/검증 데이터 스플릿 및 훈련/추론 오케스트레이션 | `candidate_workflow.py` |

---

# 3. Data Flow

```mermaid
graph TD
    A[Sparse Candidate Events 추출] --> B[candidate_labels: Leak-free Triple Barrier 타겟(y) 산출]
    A --> C[candidate_dataset: Identity, Market, Symbol, Signal Context 피처(X) 병합]
    B --> D[candidate_workflow: Purged Walk-forward Split]
    C --> D
    D --> E[candidate_edge: LightGBM Regressor 모델 훈련 및 Prior/Residual 추정]
    E --> F[candidate_edge/gate: Calibration Set에서 Rank IC 통과 여부 검증 (Hard Gate)]
    F --> G[candidate_portfolio: Regime Overlay, q10 Veto 필터 적용 후 Kelly / Stop-Risk 사이징]
    G --> H[최종 Target Weight 생성 및 Portfolio 반영]
```

---

# 4. Business Rules & Invariants

- **Fail-Closed Selection (보수적 게이팅):** 모델의 예측 확신이 낮거나 Rank IC(Spearman) 평가를 통과하지 못하면 자본을 투입하지 않습니다 (`edge_gate_mode="rank_ic"`).
- **Leak-Free Labeling:** 미래 데이터를 참조하지 않도록(Look-ahead Bias 원천 차단), 진입 바(Bar)의 종가가 아닌 다음 바의 오픈 가격(Next-Open)을 체결 기준으로 삼아 Time-Exit을 처리합니다.
- **Cost-Aware Net Edge:** ML 모델의 회귀 타겟은 단순 가격 차이가 아닌, Taker 비용과 Realized Funding Bps, Hurdle Bps가 모두 차감된 **순 실현 수익(`net_event_bps`)** 입니다.
- **Single-Unit Contract:** Label 생성, ML Target 학습, Sizing 산출이 모두 동일한 위험 단위(`risk-unit, s_i`)를 기준으로 통일되어 있어야 합니다. (Bps/Q10/MFE 혼용 금지)
- **Continuous Market Regime 결합:** Portfolio 사이징 단계에서 `entry_idx - 1` 시점의 Causal Regime Overlay 강도(`overlay_mult`)를 직접 가중치에 곱하여 포지션 스케일을 조정합니다.

---

# 5. Data Schemas

### `CandidateFeatureSchema` (피처 메타데이터)
- **Signal Pre-Qualification:** `block_bootstrap` 및 `concurrency_t` 등을 통해 사전 검증된 Variant만 남김.
- **Feature Groups:** 
  - `Identity`: 전략 및 변종 원-핫 인코딩
  - `Market State`: BTC 수익률, 추세, 변동성 등 거시 정보
  - `Symbol State`: 심볼 변동성, 펀딩비 상태 등 고유 정보
  - `Signal Context`: 발화 시점의 Regime, 전략수렴도(confluence) 등 시점 맥락

### `CandidateEdgeModels` (모델 패키지 컨트랙트)
- `center_model`: `LGBMRegressor` (기대 수익 예측)
- `q10_model`, `q90_model`: `LGBMRegressor` (하방 위험, 상방 잠재력 예측 - Quantile)
- `variant_prior_r` / `global_prior_r`: Calibration Set 가중 평균에 따른 Shrinkage Prior 값

### `EdgeModelValidation` (모델 통과 기준)
- `rank_ic_cal_eval`: Calibration Set에서 실측 엣지와 예측 엣지 간의 Spearman 상관계수(IC).
- `accepted`: Rank IC T-stat이 허용 임계치를 넘었는지 여부 (bool).

---

# 6. Theory (수식 근거)

- **Risk-Unit Normalization (Z-변환):** `z_i = net_event_bps / s_i` (여기서 `s_i`는 `max(sl_thr_bps, min_risk_unit_bps)`). 이를 통해 모든 전략/코인이 동일한 정상성(Stationary) 타겟을 학습하게 됩니다.
- **Prior Shrinkage (사전 확률 수축):** 데이터 부족으로 인한 ML 모델의 과적합(Overfitting)을 방지하기 위해, Variant별 평균 모멘트(`mu_prior_i = E[z_i]`)를 베이지안 축소(Shrinkage) 방식으로 결합(`mu_i = mu_prior_i + mu_residual_i`)하여 사용합니다.
- **Calibrated Event Kelly (켈리 사이징):** `f_bin = clip(kelly_fraction * E[r] / max(E[r²], floor), 0, max_symbol_weight)`. 단, OOS(아웃오브샘플)에서는 모멘트를 재추정하지 않고 Calibration Set에서만 추정된 값을 사용합니다.
- **Catastrophic Veto:** LightGBM q10 모델 예측값이 일정 한도를 초과하는 큰 손실(`q10_net_bps < -catastrophic_shortfall_bps`)을 지시할 경우, Pointwise Filtering 단계에서 해당 이벤트를 즉각 차단합니다.

---

# 7. Known Limitations

- **Shadow Profile (그림자 포트폴리오):** 현재 Shadow Selection Profile은 예측(Prediction) 시점의 단순 이벤트 수 요약 진단 기능만 하며, OOS 실현 결과에 기반해 베스트 섀도우를 승격시키는 동적 조정은 미지원합니다.
- **Fallback Dependency:** Edge Regressor가 게이트(Rank IC)를 통과하지 못할 때 `edge_prior_enabled=True`에 의해 단순 Variant Prior로 사이징되는 Fallback이 동작하는데, 이 경로에서는 ML Residual Edge의 기여가 0이 되어 복합 피처들의 이점을 상실합니다.