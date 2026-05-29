---
title: Master Architecture — ML Alpha & Universe 3-Cohort Redesign
domain: strategy-ml
type: domain-spec
status: active
priority: critical
ai_read_policy: always
supersedes:
  - docs/specs/ml_alpha_gate_recovery.md
  - docs/specs/ml_alpha_specs_consolidated.md
  - docs/specs/universe_three_cohort_redesign.md
last_verified: 2026-05-29
---

# Master Architecture: ML Alpha & Universe 통합 명세서

## Executive Summary
본 문서는 파편화되어 있던 ML Alpha 추출 모델링, OOS 평가 체계, Universe 3-Cohort 재설계, 그리고 최근의 신호 복구(Gate Recovery) 전략을 하나의 마스터 사양서로 통합·압축한 것입니다.

추론(Inference)과 집행(Execution)의 목적을 직교 분리하여 **통계적 검정력(Breadth)**과 **실제 거래 가능성**을 동시에 극대화하며, Dynamic Cost Gate와 Regime Gate를 통해 최종 OOS 수익성을 방어합니다.

---

## 1. Universe Architecture: 3-Cohort 분리 모델

유니버스를 단일 타임라인에 결합할 경우 발생하는 Breadth 축소(N=18) 및 생존자 편향(Boundary Noise) 문제를 해결하기 위해 3-Cohort 모델을 도입합니다.

| Cohort | 역할 | 규모 | 도출 기준 | 우선 최적화 목표 |
|--------|------|------|------------------|-----------------|
| **C1 Inference** | ML 학습 및 IC 측정 | 80~150 | 분기별 Stage5 통과 종목의 시계열 Union | Breadth (√N) 극대화 |
| **C2 Live** | OOS 알파 신호 발산 | 50~80 | 현재 분기 Stage5 통과 종목 | 예측 Coverage 최대화 |
| **C3 Execution**| 실제 포지션 진입 집행 | 18~24 | 현재 분기 Stage6 최종 선발 종목 | Net IC × Capacity × 분산 |

### 1.1 Stage6 3축 다목표 점수 (Multi-Objective Selection)
과거 체결 마찰(Friction) 100% 기반의 선발을 개선하여, 자산증식 기여도를 반영합니다.
- **Friction (50%)**: 유동성(40%) + 역비용(30%) + 데이터품질(20%) + 안정성(10%)
- **Alpha Capacity (30%)**: 사전 변동성(40%) + 시장 대비 베타 분산(30%) + 레짐 독립성(30%)
- **Diversification (20%)**: 클러스터 크기의 역수(`1/√size`) 적용 및 클러스터당 최대 3종목(Cluster Cap) 제한

---

## 2. ML Alpha 모델링 및 신호 생성

### 2.1 코어 모델 아키텍처
- **Low-Capacity GBT**: 과적합 방지를 위해 구조를 단순화하고, signed target을 직접 예측합니다 (Ranker 의존성 탈피).
- **복잡도 증가 영구 금지**: 2-stage 분류기/회귀 파이프라인이나 무분별한 앙상블은 역선택을 유발하므로 배제합니다.
- **Target 정합성**: 시장 성분 누출을 막기 위해 타깃 수익률은 Beta-residualize 처리하여 학습 및 평가에 사용합니다.

### 2.2 가중치 3중 보정 (Sample Weighting)
시계열적 비정상성(Non-stationarity)과 유니버스 확장(C1)으로 인한 노이즈를 제어하기 위해 3중 가중치를 적용합니다.
1. **Time-Decay**: 최근 데이터에 더 높은 가중치 부여 (`halflife=1080` 4h bars, 약 6개월).
2. **Quality Clip**: `coverage_60d` 등 품질 하위 종목의 영향력 제한 (min=0.5).
3. **Cluster Balance**: 과대 대표된 군집(예: ETH-correlated)의 가중치 축소 (`1/√cluster_size`).

### 2.3 Horizon 및 Calibration 설정 (최신 Phase 3-B 기준)
- **Label Horizon**: 단기 Signal/Noise 비율이 높은 `label_horizon_bars = 6` (24h)을 최우선으로 채택합니다 (과거 12/18 시도는 회전율 저하 효과가 있었으나 신호 강도 강화를 위해 6으로 롤백/조정).
- **Calibrator Target**: 향후 EV 클리핑 문제 해소를 위해 `calibrator_target = "gross"` 적용을 테스트하여 비용 장벽 통과 종목 수를 확대합니다.

---

## 3. Alpha Evaluation & Execution Gates

### 3.1 Dual-Panel 평가 프레임워크
- **Acceptance 판정 (C1)**: 모델의 통계적 유의성은 반드시 **추론 패널(C1, N=70+)**을 기준으로 평가합니다. (`net_ic ≥ 0.012`, `t-stat ≥ 2.0`, `breadth ≥ 8.0`)
- **실제 집행 (C3)**: 비용 장벽(Cost Wall) 및 거래 필터는 **집행 패널(C3, N=20)**에만 적용하여 거래 현실성을 유지합니다.

### 3.2 비용 장벽 및 Gate 체계
- **Dynamic Cost Floor**: 24bps 고정 비용을 탈피하여 종목별 슬리피지, 스프레드, 펀딩비를 반영한 동적 허들 적용.
- **비대칭 숏 게이트 (Asymmetric Gate)**: 크립토 롱-바이어스를 감안하여, 진입 문턱값을 비대칭으로 설정 (Long `1.0%`, Short `0.5%`).
- **Regime Gate**: 
  - 하락장(Bear): `0.0` 노출 (전면 차단)
  - 상승장(Bull) / 횡보장(Chop): `1.0` 노출 (전면 진입)

---

## 4. 진행 현황 및 Action Items (2026-05-29 기준)

### 4.1 Gate Recovery 최근 실적
SCOREBOARD 측정 로직의 Target Mismatch(P-A, P-B, P-C) 진단 및 복구 결과:
- **타깃 Beta-residualize 적용 완료**: 시장 베타 노이즈 제거 후 `net_ic`가 `-0.0023`에서 `+0.0025`로 **부호 양수 전환**.
- **Time-decay(1080) 적용 완료**: 전 Regime(Bull/Bear/Chop) IC 양수 전환 달성.

### 4.2 잔존 Blocker 및 다음 단계 (Phase 3-B)
현재 모델의 내부 IC는 양호하나, EV 클리핑 후 C3에서의 **실질 Breadth가 1.01**에 불과해 최종 성과로 이어지지 않는 문제가 확인되었습니다 (목표치 ≥ 8.0). 이를 해결하기 위해 다음 조치를 순차 실행합니다.

1. **단기 Horizon 적용**: `config.py:label_horizon_bars = 6` 적용 (우선순위 1위, 신호 강도 회복).
2. **Target Gross 전환**: `calibrator_target = "gross"` A/B 테스트 (EV 과소평가 방지).
3. **정규 Fold IC 분리 측정**: ML_EVAL 리포트에서 Virtual refit 기여분을 분리하여 순수 OOS IC 신뢰성을 확보합니다.