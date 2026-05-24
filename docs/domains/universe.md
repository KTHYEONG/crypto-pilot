---
title: Futures Universe Management
domain: futures-universe
type: domain-spec
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/universe/
  - src/application/futures/optimization/universe_service.py
change_triggers:
  - src/domain/futures/universe/**
last_verified: 2026-05-24
---

# Futures Universe Management

## 1. Overview
거래 가능한 심볼 리스트를 필터링하고, 데이터 충족성 및 상장 기간 등을 고려하여 동적인 트레이딩 유니버스를 관리합니다. PIT(Point-In-Time)를 준수하여 생존/상폐 편향을 원천 차단합니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `universe_service.py` | 유니버스 생성 및 필터링 오케스트레이션 |
| `candidate_selector.py` | 거래량, 시총 등 기준 후보 심볼 선택 |
| `data_readiness.py` | 심볼별 백테스트/트레이딩 가용 데이터 검증 |
| `pipeline.py` | 7단계 Funnel(Stage 0~6) 순차 실행 |

---

## 3. Data Flow

```text
[Exchange Symbols] -> [Eligibility (Stage 0)] -> [Structure (Stage 1)] 
  -> [Data Quality (Stage 2)] -> [Liquidity (Stage 3)] 
  -> [Execution Cost (Stage 4)] -> [Risk Events (Stage 5)] 
  -> [Selection/Ranking (Stage 6)] -> [Universe Snapshot]
```

---

## 4. Business Rules

### Must Follow
- **Strict Listing Period:** 최소 상장 기간 90일 미달 심볼 제외 (`listing_age_days >= 90`).
- **Volume Block:** 30일 Median ADV가 25M USDT 미만인 심볼 차단.
- **PIT Integrity:** `knowledge_date <= as_of` 조건으로 미래 정보 참조 원천 차단.

### Must Not Do
- **Survivor Bias:** 상장 폐지된 심볼의 과거 데이터를 누락하여 수익률을 왜곡하지 말 것.
- **Price Predictive Delisting:** `deliveryDate` 등으로 상폐 시점을 미리 예측하여 청산하지 말 것.

---

## 5. Detailed Specifications

### 5.1 7-Stage Funnel Thresholds
- **Stage 2 (Quality):** `min_is_coverage >= 0.80`, `min_coverage_60d >= 0.95`.
- **Stage 3 (Liquidity):** `adv_usdt_median >= 25M`, `max_amihud_30d <= 1.63e-9`.
- **Stage 4 (Cost):** `execution_cost_bps <= 50.0 bps`.
- **Stage 5 (Risk):** `vol_30d` in `[0.05, 4.0]`, `|funding_zscore| <= 2.5`.

### 5.2 Execution Cost Function
$$\text{cost\_bps} = 2 \cdot \text{taker} + 2 \cdot \text{half\_spread} + \text{impact} + \text{tick\_cost}$$
- **half_spread (Post-2020):** `bookDepth` 실측 중앙값.
- **half_spread (Pre-2020):** `Corwin-Schultz` OHLC 변형 모델 Fallback.

### 5.3 Snapshot Quality Score
$$Score_{universe} = fill\_rate \times \log_{10}\!\left(\frac{\text{median\_adv\_usdt}}{10^6}\right) \times \frac{1}{\text{mAEC\_bps}}$$
- **Excellent:** Median Cost < 18.0 bps, Median ADV > 100M USDT.

---

## 6. Examples
- **Input:** Symbol 'ABC' listed for 30 days, Min requirement 90 days
- **Output:** Excluded from Universe (Stage 5 Listing Age Gate)

---

## 7. Testing Expectations
- **Forced Dropout Rate:** 90일 Dwell Time 미달 자산의 비정상 퇴출률이 10% 미만인지 확인.
- **Universe Consistency Test:** 백테스트와 라이브 트레이딩의 유니버스 선정 결과가 동일 시점에서 일치하는지 확인.
