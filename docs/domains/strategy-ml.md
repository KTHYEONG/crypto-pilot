---
title: Binance Futures ML Strategy
domain: futures-strategy-ml
type: domain-spec
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/
change_triggers:
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/ranker.py
last_verified: 2026-05-24
---

# Binance Futures ML Strategy

## 1. Overview
`opt_main_futures.py`의 strategy stage에 비용을 초과 극복하는 expected return alpha를 공급하는 ML 기반 전략입니다. 주문, 비중, 레버리지를 직접 제어하지 않고 오직 `alpha_panel`만을 생성합니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `ml_builder.py` | ML feature-label-train-infer 오케스트레이션 |
| `features.py` | PIT-safe 피처 텐서(CS-Sharpe 등) 생성 |
| `labels.py` | T+1 체결 기준 마찰 비용이 제외된 Gross Alpha 레이블 생성 |
| `ranker.py` | CS-demeaned GBT Regressor 학습 및 스코어 추론 |
| `calibrator.py` | 분위수 기반 동적 불확실성 조정 및 EV 보정 |

---

## 3. Data Flow

```text
[Data Maps] -> [Feature/Label Generation] -> [Double-Weighted Dataset] 
  -> [CS-Demeaned Training] -> [Quantile Calibration] 
  -> [Cost Barrier Gating] -> [Alpha Panel]
```

---

## 4. Business Rules

### Must Follow
- **Alpha Supplier Only:** pure expected return alpha(Bps)만 산출할 것.
- **Double-Weighting:** 실측 리턴 절대값에 비례하여 sample_weight 가중.
- **Dynamic Cost Barrier:** 거래 비용보다 작은 노이즈성 시그널은 `0.0`으로 소거.

### Must Not Do
- **Portfolio Control:** ML이 target weight, order, leverage를 직접 계산 금지.
- **Look-Ahead Leakage:** 미래 시점 데이터를 참조하여 Scaler/Imputer 피팅 금지.

---

## 5. Detailed Specifications

### 5.1 Feature Schema (50 Pillars)
- **Reversal/Momentum:** `ret_1` ~ `ret_36` 및 횡단면 랭크 팩터.
- **Volatility:** Realized Vol, Downside Vol, ATR Ratio.
- **Carry/Liquidity:** Funding Z-score, Volume Z-score, ADV Rank.
- **CS-Sharpe (High-Performance):** 개별 변동성 대비 기대수익률 강도를 크로스섹션 랭크화 한 `cs_sharpe_6` 및 `cs_sharpe_18`.

### 5.2 Double-Weighting System
무작위 노이즈 신호 배제를 위해 리턴 절대값($|y_{ev}|$)에 비례하여 샘플 가중치를 동적으로 부여합니다.
$$\text{sample\_weight} = \text{original\_weight} \times (1.0 + 2.0 \times |y_{ev}|)$$

### 5.3 Quantile EV Calibration
- **Quantile Loss:** `q10`, `q50`, `q90` 분위수 예측기를 동시 학습.
- **Uncertainty Adjustment:** 예측 불확실성 폭($q_{90} - q_{10}$)에 따라 알파 강도를 조절하여 꼬리 위험 방어.

### 5.4 Output Contract (`alpha_panel`)
- **Index:** `MultiIndex(datetime, symbol)`
- **Columns:** `alpha_long` (Bps), `alpha_short` (Bps)
- **Zero-filling:** 미매칭 구간은 반드시 `0.0`으로 치환하여 계좌 오염 방지.

---

## 6. Examples
- **Input:** Predicted EV 10bps, Round-trip Cost 14bps
- **Output:** Gated EV 0bps (Cost Barrier 적용으로 노이즈 소거)

---

## 7. Testing Expectations
- **Spearman IC Test:** 3 fold 연속 음수 기록 시 학습 하드 페일 판정.
- **Inference Integrity:** 추론 결과에 NaN이 포함되지 않았는지, 롱/숏 양방향 신호가 존재하는지 확인.
- **PIT Test:** 피처 연산 시 미래 데이터 참조(Look-ahead)가 없는지 검증.
