---
title: MHS Architecture - 04. Portfolio Assembly & Regime Control
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/regime.py
  - src/mhs/scaling.py
  - src/mhs/books.py
  - src/mhs/pipeline/stages/assemble.py
  - src/mhs/params.py
change_triggers:
  - src/mhs/regime.py
  - src/mhs/scaling.py
  - src/mhs/pipeline/stages/assemble.py
last_verified: 2026-09-03
---

# 04. 포트폴리오 조립 및 2중 레짐 방어 제어

## 1. 개요 (Overview)
횡단면 모멘텀 전략(Cross-Sectional Momentum)은 평상시 탁월한 시장 초과 수익을 내지만, 시장 전체가 패닉 셀링에 빠지거나 급반등할 때 자산 간 상관관계가 1로 수렴하고 숏 포지션에서 숏스퀴즈가 폭발하며 자본이 붕괴하는 **모멘텀 크래시(Momentum Crash)**에 매우 취약합니다.

MHS는 포트폴리오 조립 단계에서 추적 오차 기반 리밸런싱을 적용하고, **시장 베타 직교화와 2중 레짐 방어 체계(BTC 틸트 + 전략 P&L 변동성 타겟팅)**를 통해 어떠한 극단적 시장 충격에도 자본을 생존시키는 견고한 리스크 엔진을 탑재하고 있습니다.

---

## 2. 북 블렌드 및 포트폴리오 조립 (Portfolio Assembly)

### 1) Phase 1 북 블렌드 비중
디스커버리 및 퀄리피케이션 게이트 실측 결과에 따라 확정된 배분 비율은 다음과 같습니다 ([`BOOK_BLEND_WEIGHTS`](file:///home/kth/crypto-pilot/src/mhs/params.py)):
- **Fast Reversal Book**: **0.0 (0%)**
- **Slow Momentum Ensemble Book**: **1.0 (100%)**

### 2) 포트폴리오 리밸런스 트리거 (RC-1, Portfolio Rebalance Trigger)
일반적인 자산별 개별 데드밴드(Deadband) 방식은 특정 코인만 주문이 나가고 다른 코인은 대기하면서 포트폴리오 전체의 **달러 중립성(Dollar Neutrality)이 붕괴**되는 치명적인 결함을 유발합니다.

MHS는 이를 해결하기 위해 포트폴리오 전체 레벨에서 리밸런스를 판단합니다 ([`portfolio_rebalance_trigger`](file:///home/kth/crypto-pilot/src/mhs/books.py)):
- **추적 오차(Tracking Error) 측정**:
  $$\text{Tracking Error}_t = \sum_{i} \left| w_{i, t}^{\text{target}} - w_{i, t}^{\text{current}} \right|$$
- **임계값 기반 일괄 업데이트**:
  - 추적 오차가 임계값(`0.20`, 20%) 미만일 때는 이전 포지션을 그대로 유지하여 마이크로 턴오버 및 틱 수수료를 억제합니다.
  - 추적 오차가 **20% 이상 벌어질 때만** 전체 포트폴리오의 타겟 수량을 한 번에 일괄 갱신합니다.
  - 이를 통해 달러 중립성과 현금 균형을 완벽히 보존하면서 회전율을 최적화합니다.

---

## 3. 시장 베타 직교화 (RC-4, Market Beta Neutralization)

포트폴리오가 암호화폐 전체 시장의 단순 상승/하락 방향성(Market Beta)에 편승하여 수익이나 손실을 내는 것을 막고, 순수한 횡단면 상대 우위(Alpha)만을 취하기 위해 직교화(Orthogonalization)를 수행합니다 ([`beta_neutralize_weights`](file:///home/kth/crypto-pilot/src/mhs/regime.py)).

1. **인과적 롤링 베타 산출 ([`causal_market_beta`](file:///home/kth/crypto-pilot/src/mhs/regime.py))**:
   - 최근 720개 바(30일)의 1시간봉 수익률을 바탕으로, 시장 대표 자산(`BTCUSDT`) 대비 각 심볼의 롤링 OLS 베타 $\beta_i$를 인과적으로 산출합니다.
2. **베타 제로 투영**:
   - 포트폴리오의 총 시장 노출도 $\sum w_i \beta_i = 0$이 되도록 가중치를 선형 투영(Projection)하여 시장의 거시적 급변으로부터 포트폴리오를 격리합니다.

---

## 4. 2중 레짐 방어 아키텍처 (Dual-Axis Regime Defense)

MHS는 횡단면 모멘텀의 꼬리 위험(Tail Risk)을 방어하기 위해 완전히 독립된 2개의 레짐 제어 축을 동시에 가동합니다.

```text
                     [시장 충격 및 모멘텀 크래시 발생]
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
[1축: BTC 참조 자산 크래시 틸트]                     [2축: 전략 P&L 실현 변동성 타겟팅]
• BTC 추세 Z-score 급락 감지                        • 전략 자체의 최근 21일 P&L 변동성 추적
• 달러 중립 북에 하락 틸트 혼합                       • 변동성 급등 시 Gross Exposure 인과적 축소
• 숏스퀴즈 및 하락장 꼬리 손실 상쇄                  • 자본 완전 파괴(CAPITAL_INVARIANT_BREACH) 원천 차단
```

### 1축: 참조 자산 기반 크래시 레짐 틸트 (`crash_regime_tilt_weights`)
- **고정 참조 자산**: 유니버스 변경에 따른 착시를 배제하기 위해 시장 대표성이 절대적인 `BTCUSDT`를 단일 참조 지표로 고정합니다.
- **방향성 틸트 혼합**:
  - BTC의 인과적 추세 Z-score가 크래시 임계치 밑으로 떨어지면, 달러 중립 북에 **하락 방향성 틸트(`alpha` 비율)**를 혼합합니다.
  - 이는 폭락장에서 알트코인 숏 포지션의 부담을 완화하고, 하락 꼬리 위험을 자연스럽게 방어(Hedging Offset)합니다.

### 2축: 전략 자체 P&L 실현 변동성 타겟팅 (`Strategy P&L Volatility Targeting`)
- **전략 레벨 변동성 측정**:
  - 개별 코인의 변동성이 아니라, **"MHS 전략 포트폴리오 자체의 최근 21일 일별 P&L 실현 변동성"**을 추적합니다 ([`PNL_TARGET_ANNUAL_VOL = 0.20`](file:///home/kth/crypto-pilot/src/mhs/params.py)).
- **Two-Pass Replay 기반 동적 노출 축소**:
  - 전략의 실현 변동성이 장기 중앙값(Median)보다 급격히 튀어 오르면 모멘텀 붕괴 레짐으로 판정합니다.
  - 인과적 비율(`변동성 중앙값 / 최근 실현 변동성`)을 적용하여 전체 익스포저(Gross Exposure)를 즉각 축소합니다 (최대 0.2까지 방어적 축소).
- **실측 검증된 생존 필수선**:
  - 실제 백테스트 진단에서 P&L 변동성 타겟팅을 비활성화(`--no-pnl-vol-target`)한 결과, **2025-07-20 극단적 크래시 시점에 `CAPITAL_INVARIANT_BREACH`(자본 완전 증발)가 발생**하여 파산했습니다.
  - 현재 유지되는 평균 53% 수준의 총 익스포저는 인위적 제약이 아닌, **전략 생존을 담보하는 필수불가결한 꼬리 위험 방어선**임이 실측으로 입증되었습니다.

---

## 5. 자본 스케일링 및 단일 오버레이 계약 (I-SCALE-IS-DEPLOYED-OVERLAY)

MHS의 리스크 스케일링([`src/mhs/scaling.py`](file:///home/kth/crypto-pilot/src/mhs/scaling.py))은 다음 원칙을 준수합니다:
- **배포 오버레이 재정의**: `exposure_scale`은 검증 대상 알파 신호가 아니며, 최상위 블렌드가 배치 확정한 리스크 관리 오버레이입니다.
- **Walk-Forward 폴드 일치성**: 각 검증 폴드는 상위 블렌드가 결정한 `exposure_scale`을 구간별로 슬라이스하여 reindex 후 동일하게 적용함으로써, 신호 생성의 독립성과 배포 리스크 통제의 일관성을 동시에 유지합니다.

---

## 6. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 클래스 |
|---|---|---|
| 포트폴리오 리밸런스 트리거 | [src/mhs/books.py](file:///home/kth/crypto-pilot/src/mhs/books.py) | `portfolio_rebalance_trigger` |
| 시장 베타 및 크래시 틸트 | [src/mhs/regime.py](file:///home/kth/crypto-pilot/src/mhs/regime.py) | `causal_market_beta`, `beta_neutralize_weights`, `crash_regime_tilt_weights` |
| P&L 변동성 타겟팅 및 스케일 | [src/mhs/scaling.py](file:///home/kth/crypto-pilot/src/mhs/scaling.py) | `scale_pnl_by_vol_target`, `apply_exposure_scale` |
| 포트폴리오 조립 파이프라인 | [src/mhs/pipeline/stages/assemble.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/assemble.py) | `assemble_report` |
