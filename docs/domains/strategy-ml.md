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

### 5.5 B2 - Beta-Residualized Labels
`src/domain/futures/strategy/labels.py`의 `build_label_panel()`은 수익률에서 시장 베타 성분을 제거하여 특정 종목 고유의 알파를 분리합니다.

**Core Algorithm:**
- **_compute_trailing_beta():** 각 종목별로 120-bar 롤링 OLS 회귀를 통해 등가중 크로스섹션 시장 수익률에 대한 베타 추정
  - 윈도우: `_BETA_WINDOW = 120` bars
  - 최소 관측값: `_BETA_MIN_PERIODS = 20`
  - Look-Ahead Free: 시점 t의 베타는 [t-120, t) 범위 과거 데이터만 사용하여 미래 정보 누설 방지
  - 루프 정책: N개 종목(~20-50) 단위 루프, T 길이 루프 없음 (Zero-Loop Policy 준수)
  
- **Residualization Formula:**
  ```
  long_net[t,i] = gross_long[t,i] - beta[t,i] * market_fwd_ret[t] - cost - funding[t,i]
  short_net[t,i] = gross_short[t,i] + beta[t,i] * market_fwd_ret[t] - cost + funding[t,i]
  ```
  
**Benefit:** 시장 공통 인수를 제거하여 순수 특정 위험 알파만 모델에 노출, 신호 강도 향상

### 5.6 Track A - Diagnostic Logging
`src/domain/futures/strategy/ml_builder.py` 함수 `build_ml_strategy_alpha()` 내에서 다음 구조화된 로그를 출력하여 신호 품질을 실시간 모니터링합니다.

**[ML-ALPHA-IC]**: 크로스섹션 IC(Information Coefficient) 기반 신호 유효성 판정
```
[ML-ALPHA-IC] mean_ic=0.0112 icir=0.42 t_stat=1.02 hit_ratio=0.55 n_obs=180
```
- `mean_ic`: 평균 Spearman Rank IC (이상적: ≥0.02)
- `icir`: IC Ratio = mean_ic / std(ic) (이상적: ≥0.5)
- `t_stat`: t-통계량 = mean_ic / (std_ic / √n_obs) (이상적: ≥2.0, p<0.05)
- `hit_ratio`: IC > 0인 기간 비율 (이상적: ≥0.45)
- `n_obs`: 유효 크로스섹션 시점 개수

**[ML-COST-WALL]**: 거래 비용 대비 예상 수익의 유효성 진단
```
[ML-COST-WALL] alpha_p95=10.04bps friction=14bps hurdle_bps=40.0 floor=54.0 status=WARN
```
- `alpha_p95`: 모델 예측 알파의 95분위수(단위: bps)
- `friction`: 편도 거래 비용(수수료+슬리피지, 기본값: ~14bps)
- `hurdle_bps`: EV_HURDLE 임계값(기본값: 40bps, 설정 범위: [5.0, 100.0])
- `floor`: 유효 비용 벽 = friction + hurdle_bps
- `status`: 신호 통과 여부
  - `OK`: alpha_p95 > floor
  - `WARN`: floor/2 < alpha_p95 ≤ floor (한계 신호, 주의 필요)
  - `FAIL`: alpha_p95 ≤ floor/2 (실패, 개선 필요)

### 5.7 B4 - IC Quality Gate
신호 품질 게이트: `src/domain/futures/strategy/diagnostics.py` 함수 `passes_ic_gate()` 및 `ml_builder.py` 적용점

**Gate Thresholds (현재: Warning-Only, 향후 Exception 전환 예정):**
- `mean_ic ≥ 0.02` (절대 신호 강도)
- `t_stat ≥ 2.0` (통계적 유의성, p<0.05 equiv.)
- `hit_ratio ≥ 0.45` (신호 방향성 일관성)

**Behavior:**
- Gate 미만족 시 `[ML-IC-GATE] WARN: reason=...` 로그 출력 (현재 경고 수준)
- 향후 Phase: EV_HURDLE 자동 상향 또는 모델 완전 차단으로 전환 가능

---

## 6. Examples
- **Input:** Predicted EV 10bps, Round-trip Cost 14bps
- **Output:** Gated EV 0bps (Cost Barrier 적용으로 노이즈 소거)

---

## 7. Testing Expectations
- **Spearman IC Test:** 3 fold 연속 음수 기록 시 학습 하드 페일 판정.
- **Inference Integrity:** 추론 결과에 NaN이 포함되지 않았는지, 롱/숏 양방향 신호가 존재하는지 확인.
- **PIT Test:** 피처 연산 시 미래 데이터 참조(Look-ahead)가 없는지 검증.
