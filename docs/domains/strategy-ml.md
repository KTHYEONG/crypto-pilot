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

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `ml_builder.py` | AWF(Anchored Walk-Forward) 기반 학습-추론 오케스트레이션 및 패널 캐싱 |
| `features.py` | PIT-safe 피처 텐서 (CS-Sharpe, Volatility, Carry 등 50+ Pillars) 생성 |
| `labels.py` | Beta-Residualized + CS-Demeaned 수익률 라벨 생성 (Fee 제외, Funding 반영) |
| `ranker.py` | LightGBM 기반 상대 스코어(Rank) 추론 (group_ndcg 또는 regression) |
| `calibrator.py` | Quantile Loss 기반 EV 교정 및 Uncertainty Penalty 적용 |
| `diagnostics.py` | IC Quality Gate 및 Cost-Wall(Friction + Hurdle) 유효성 검증 |

---

## 3. Data Flow

```text
[Data Maps] -> [Feature/Label Generation] -> [Single-Weighted Dataset] 
  -> [CS-Demeaned Training] -> [Quantile Calibration (Gross Bps)] 
  -> [Dynamic Cost Barrier Gating] -> [Alpha Panel (v3 Metadata)]
```

---

## 4. Business Rules

### Must Follow
- **Pure Alpha Supplier:** Target weight나 leverage를 직접 계산하지 말고 순수 기대수익률(Bps)만 산출할 것.
- **Single-Order Weighting:** 가중치는 `labels.py`에서 1회만 계산 ($1 + 2|y_{ev}|$). `dataset.py`에서 재곱 금지.
- **Target Separation:** Ranker는 `signed_net_ret`(Demeaned)를, Calibrator는 `exec_net_ret`(Gross)를 타깃으로 사용.
- **PIT Integrity:** 모든 정규화(Robust Scaler) 및 결측치 처리(Imputer)는 Train split 기반으로만 수행 (Leakage 방지).
- **Embargo Invariant:** `embargo_bars >= label_horizon_bars` 조건을 반드시 준수.

### Must Not Do
- **Portfolio Control:** ML 내부에서 Risk-cap projection이나 Portfolio optimization 수행 금지.
- **Look-Ahead Leakage:** 미래 시점의 Cross-sectional median/mean을 참조하여 과거 시점 정규화 금지.
- **Repeated Centering:** Calibrator 출력 이후 추가적인 Group-centering을 수행하여 절대 EV 크기를 소거하지 말 것.

---

## 5. Detailed Specifications

### 5.1 Training & Labels
- **Beta-Residualization:** 120-bar 롤링 OLS를 통해 시장 베타 제거.
- **CS-Demean:** 라벨 생성 시점(B1/B2)에서 1회 적용하여 `short_net ≈ -long_net` 대칭성 보존.
- **Model Family:** `lgbm_regression` (Default) 및 `lgbm_huber` 지원. `ranking_mode="group_ndcg"` 자동 Fallback 지원.

### 5.2 Calibration & EV (v2)
- **Quantile EV:** $q_{10}, q_{50}, q_{90}$ 예측을 통한 보수적 EV 산출.
- **Tail Penalty:** 불확실성($q_{90}-q_{10}$)에 비례한 Sign-symmetric 페널티 부과.
- **Dynamic Cost Gate:** `ev > (dynamic_cost + hurdle)` 조건 미충족 시 신호를 $0.0$으로 소거.

### 5.3 Quality Gates (B4)
- **IC Gate:** `mean_ic >= 0.01`, `t_stat >= 1.5`, `hit_ratio >= 0.45` (완화된 초기값, 점진적 강화).
- **Cost-Wall:** `alpha_p95 > friction + hurdle`. 기본 `hurdle_bps=10.0`, 탐색 범위 `[3.0, 20.0]`.

### 5.4 Performance & Caching
- **Panel Cache:** AWF leg 간 중복되는 Feature/Label 생성을 `precompute_anchored_ml_panels`로 최적화.
- **Ensemble Cache:** 동일 파라미터 멤버에 대한 중복 Alpha build 및 백테스트 생략.

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
