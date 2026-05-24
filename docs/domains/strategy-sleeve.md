---
title: Binance Futures Strategy Sleeve
domain: futures-strategy-sleeve
type: domain-spec
status: active
priority: medium
ai_read_policy: when_related
related_paths:
  - src/domain/futures/portfolio/signal_composer.py
change_triggers:
  - src/domain/futures/portfolio/signal_composer.py
last_verified: 2026-05-24
---

# Binance Futures Strategy Sleeve

## 1. Overview
여러 개의 개별 전략(Sleeve)으로부터 오는 신호를 결합하고 상충되는 시그널을 조정하여 최종 포트폴리오 가중치를 결정하는 레이어입니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `signal_composer.py` | 다중 슬리브 신호 결합 및 중화(Neutralization) |
| `portfolio_constructor.py` | 최종 가중치 스케일링 및 레버리지 적용 |
| `builder.py` | `build_strategy_alpha()` - 다중 슬리브 오케스트레이션 |

---

## 3. Data Flow

```text
[Sleeve Alphas] -> [Dynamic Blending (IC-weighted)] 
  -> [Grinold Calibration] -> [Combined Alpha Panel] 
  -> [Signal Composer (Friction/Hurdle)] -> [Target Weights]
```

---

## 4. Business Rules

### Must Follow
- **Neutralization:** 전체 포트폴리오의 넷 익스포저(Net Exposure)가 목표 범위 내에 있도록 조정.
- **Dynamic Weighting:** 각 슬리브의 롤링 Spearman IC 및 t-stat를 바탕으로 성과가 검증된 슬리브에만 가중치 배분.
- **Grinold Calibration:** 블렌딩된 z-score를 변동성과 IC 강도를 고려하여 expected return($\alpha_{hat}$) 단위로 변환.

### Must Not Do
- **Over-leveraging:** 개별 슬리브 비중의 합이 시스템 전체 레버리지 한도를 초과 금지.
- **Legacy Dependency:** `legacy/*` 또는 `alpha_factory/*` 모듈을 참조하지 말 것.

---

## 5. Detailed Specifications

### 5.1 Multi-Sleeve List
1. **XS Reversal (Core):** 최근 24h 수익률 하위 심볼 매수, 상위 심볼 매도.
2. **Carry:** 8h 원천 펀딩비 롤링 평균 기반 고펀딩 수혜 포지션 확보.
3. **TS Momentum:** 심볼별 시간축 모멘텀 평가 (기본 비활성, t-stat 통과 시 활성).

### 5.2 Dynamic Blending Logic
- **IC Window:** 180 bars 롤링 통계 사용.
- **Hard Gates:** `min_t_stat >= 2.0`, `min_hit_ratio >= 0.45` 통과 시에만 가중치 부여.
- **Fallback:** 모든 슬리브 미통과 시 절대값이 가장 높은 `mean_ic` 슬리브를 강제 적용.

### 5.3 Grinold Calibration Formula
$$\alpha_{hat} = \text{score} \times \sigma_{fwd} \times \text{IC}_{lagged}$$
- **score:** Blended & Winsorized CS z-score.
- **$\sigma_{fwd}$:** 30일 롤링 변동성 기반 예상 선행 변동성.
- **$\text{IC}_{lagged}$:** 롤링 IC 강도.

---

## 6. Examples
- **Input:** Sleeve A (Long 0.5, t=2.5), Sleeve B (Short 0.3, t=1.2)
- **Output:** Combined Signal dominated by Sleeve A (Sleeve B filtered by t-stat gate).

---

## 7. Testing Expectations
- **Neutralization Test:** 시장 중립 전략 설정 시 넷 익스포저가 0에 수렴하는지 확인.
- **Sleeve IC Monitoring:** 각 슬리브별 개별 IC 및 결합 IC 성과 보고서 생성 확인.
