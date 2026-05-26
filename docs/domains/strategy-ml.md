---
title: Binance Futures ML Strategy
domain: futures-strategy-ml
type: domain-spec
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/
  - src/domain/futures/optimization/objectives.py
change_triggers:
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/ranker.py
  - src/domain/futures/optimization/objectives.py
last_verified: 2026-05-26
---

# Binance Futures ML Strategy

## 1. Overview
`opt_main_futures.py`의 strategy stage에 비용을 초과 극복하는 expected return alpha를 공급하는 ML 기반 전략입니다. 주문, 비중, 레버리지를 직접 제어하지 않고 오직 `alpha_panel` (Bps)만을 생성하는 순수 Alpha Supplier 역할을 수행합니다.

**관련 ADR:** [`docs/decisions/fix-logic.md`](../decisions/fix-logic.md) — Phase 0-2 신뢰도 복구 (leakage 제거, purge/embargo 실효화, gate 단일화, quantile 단조성)

---

## 2. Core Components (3-Layer ML Architecture)

| Layer | Component | Algorithm | Responsibility |
|---|---|---|---|
| **L1: Estimation** | `ranker.py` | LGBM GBDT | CS-Demeaned Regression을 통한 상대적 Alpha Rank 추론 |
| **L2: Calibration** | `calibrator.py` | Quantile Reg. | $q_{10}, q_{50}, q_{90}$ 예측 및 불확실성 반영 보수적 EV 산출 |
| **L3: Verification** | `diagnostics.py` | Statistical Filter | IC Quality Gate 및 Cost-Wall (Alpha vs Cost) 최종 검증 |
| **Infrastructure** | `ml_builder.py` | AWF Logic | Refit 오케스트레이션, Feature/Label 생성 및 캐싱 |
| **Data Engine** | `features.py` / `labels.py` | PIT Engine | 50+ Pillars 피처 생성 및 Beta-Residualized 라벨 생성 |

---

## 3. Data Flow

```text
[Data Maps] -> [Feature/Label Generation] -> [L1: CS-Demeaned Ranker] 
  -> [L2: Multi-Quantile Calibrator] -> [Uncertainty/Tail Penalty] 
  -> [L3: Cost-Wall & IC Gate] -> [Final Alpha Panel]
```

---

## 4. Business Rules

### Must Follow
- **Pure Alpha Supplier:** Target weight나 leverage를 직접 계산하지 말고 순수 기대수익률(Bps)만 산출할 것.
- **Single-Order Weighting:** 가중치는 `labels.py`에서 1회만 계산 ($1 + 2|y_{ev}|$). `dataset.py`에서 재곱 금지.
- **Target Separation:** Ranker는 `signed_net_ret`(Demeaned)를, Calibrator는 `exec_net_ret`(Gross)를 타깃으로 사용.
- **PIT Integrity:** 모든 정규화(Robust Scaler) 및 결측치 처리(Imputer)는 Train split 기반으로만 수행 (Leakage 방지).
- **Dual-Side Learning:** Long과 Short을 별도의 모델로 학습하여 비대칭적 시장 특성 반영.
- **Embargo Invariant:** `embargo_bars >= label_horizon_bars` 조건을 반드시 준수.

### Must Not Do
- **Portfolio Control:** ML 내부에서 Risk-cap projection이나 Portfolio optimization 수행 금지.
- **Look-Ahead Leakage:** 미래 시점의 Cross-sectional median/mean을 참조하여 과거 시점 정규화 금지.
- **Repeated Centering:** Calibrator 출력 이후 추가적인 Group-centering을 수행하여 절대 EV 크기를 소거하지 말 것.
- **Realized-Label Gate:** Test 구간에서 `cost_clearance_target` (미래 실현 수익)을 gate 조건으로 사용 금지. ex-ante `_ev_gate_2d = (cost + hurdle)/1e4` 만 허용.
- **Dead Purge/Embargo:** `FoldSpec` 생성 시 purge/embargo를 인덱스에 직접 반영하지 않으면 dead param이 됨. `_purged_valid_start`, `_embargoed_test_start`로 오프셋 적용 필수.

---

## 5. Detailed Specifications

### 5.1 L1: Alpha Ranker (Estimation)
- **Model Family:** `lightgbm_regression` (Default) 및 `lgbm_huber` 지원.
- **CS-Demeaning:** 각 타임스텝(Group) 내에서 수익률 평균을 차감하여 Market-beta를 제거. 모델이 방향성이 아닌 '상대적 우위'를 학습하도록 강제.
- **Fallback Logic:** 데이터 그룹이 부족할 경우 `lambdarank (NDCG)`에서 `pointwise regression`으로 자동 다운그레이드하여 안정성 확보.

### 5.2 L2: EV Calibrator (Magnitude & Confidence)
- **Quantile EV:** $q_{10}, q_{50}, q_{90}$ 3개 분위수 예측을 통해 기대값의 분포를 추정.
- **Conservative EV:** `ev = q50 * (1 - lambda * penalty_ratio)`. 불확실성($q_{90}-q_{10}$)이 클수록 Sign-symmetric 페널티를 부과하여 보수적으로 산출.
- **Confidence Score:** $|q_{50}| / (q_{90} - q_{10})$ 지표를 통해 예측의 신뢰도를 계량화.
- **Monotonic Sort:** `predict_ev_quantiles` 반환 직후 per-row `np.sort(axis=1)` 적용으로 quantile crossing 방지 (`q10 ≤ q50 ≤ q90` 보장).

### 5.3 L3: Quality Gates (Verification)
- **IC Gate:** `mean_ic >= 0.01`, `t_stat >= 1.5`, `hit_ratio >= 0.45` 등 통계적 유의성 검증.
- **Cost-Wall (B4):** `alpha_p95 > friction + hurdle`. 개별 신호 단위에서 `ev > dynamic_cost + hurdle` 미충족 시 $0.0$으로 소거(Gating).
- **Preservation Ratio:** Gating 이후에도 유의미한 수의 신호가 살아남는지 (`xs_long_preservation_ratio`) 체크하여 모델 사멸 방지.
- **AWF Gate:** AWF 경로도 동일한 quality/IC gate를 적용하되, Optuna 최적화 중단 방지를 위해 warn-only 모드로 실행.
- **Gate 단일화:** ml_builder 내 ex-ante cost gate 제거. 단일 gate는 `signal_composer`의 `mu = beta*alpha - friction; where(mu >= hurdle, mu, 0)` 로 일원화.

### 5.4 Feature Engineering
- **Robust Scaler:** Train 구간의 99.5% 분위수를 기준으로 Clip하여 이상치 영향 최소화.
- **Imputer:** Train 구간의 Median값을 사용하여 실시간 추론 시의 데이터 결측(Missing) 대응.
- **Feature Groups:** Carry, Volatility, Momentum, CS-Sharpe 등 도메인 지식 기반 50+ Pillars 구성.

### 5.5 Split-Consistent Alpha Artifact Contract (2026-05-26)
- **Single Build-Merge Path:** Final evaluation의 IS/HO 평가는 OOS와 동일하게 `build_strategy_alpha -> merge_ml_output_into_data_maps` 경로를 사용해야 한다.
- **Metadata Consistency Gate:** Split 간 `strategy_name`, `config_hash`, `selected_horizon`, `model_family`, `cost_source`가 불일치하면 즉시 fail-fast 한다.
- **Cost Source Inference:** `execution_cost_bps` 컬럼 존재 여부를 기반으로 split cost source를 `per_symbol`/`fallback_global`로 표준화해 비교한다.

### 5.6 Calibrator Weight Propagation Contract (2026-05-26)
- **Weight Forwarding:** `LongMatrixDataset.sample_weight`는 quantile calibrator 학습 시 `model.fit(sample_weight=...)`에 반드시 전달한다.
- **Validation Weighting:** early-stopping validation set에도 `eval_sample_weight`를 전달해 train/valid weighting policy를 일치시킨다.

### 5.7 Canonical EV Hurdle Default (2026-05-26)
- **Single Default Resolver:** `default_ev_hurdle_bps()`를 통해 `FUTURES_DEFAULT_EV_HURDLE_BPS`의 기본값 해석을 단일화한다.
- **No Local Fallback Drift:** 모듈별 `5/10/40` 하드코딩 fallback을 금지하고 동일 helper를 재사용한다.

### 5.8 Beta Fallback Contract (2026-05-26)
- **Causal BTC-Beta:** 소스 `beta` 컬럼이 없으면 trailing return window 기반 `cov(symbol, btc)/var(btc)`로 `beta_2d`를 계산한다.
- **No Look-Ahead:** 시점 `t`의 beta는 `[t-lookback+1, t]` 과거 구간만 사용한다.
- **BTC Dependency:** BTC 기준 심볼이 없으면 fallback beta 계산을 수행하지 않는다.

---

## 6. Examples
- **Input:** $q_{50}=15\text{bps}$, $\text{friction}=14\text{bps}$, $\text{hurdle}=10\text{bps}$
- **Output:** $0\text{bps}$ (Cost Barrier $24\text{bps}$ 미달로 인한 소거)
- **Input:** $q_{50}=30\text{bps}$, $\text{friction}=14\text{bps}$, $\text{hurdle}=10\text{bps}$, $\text{uncertainty penalty}=5\text{bps}$
- **Output:** $11\text{bps}$ (페널티 차감 후 허들 통과)

---

## 7. Testing Expectations
- **Spearman Rank IC:** OOS 구간에서 3-fold 연속 음수 발생 시 파이프라인 중단.
- **Directional Viability:** `alpha_long/short`의 비영(non-zero) 비율이 0일 경우 Execution Stage 진입 차단.
- **Memory/Time Efficiency:** AWF leg refit 시 `hidden_overhead`가 전체 실행 시간의 20% 이내인지 확인.
- **Leakage Regression:** `test_build_ml_strategy_alpha_anchored_test_alpha_independent_of_future_labels` — future label perturbation 시 test alpha 불변 단언.
- **Quantile Monotonicity:** `predict_ev_quantiles` 반환값에서 `q10 ≤ q50 ≤ q90` per-row 조건 항상 성립.
