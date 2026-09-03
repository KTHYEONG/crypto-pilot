---
title: MHS Architecture - 02. Alpha Signals & Horizons
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/discovery.py
  - src/mhs/books.py
  - src/mhs/horizons.py
  - src/mhs/features.py
  - src/mhs/funding.py
  - src/mhs/trend_sleeve.py
  - src/mhs/pipeline/stages/book.py
  - src/mhs/params.py
change_triggers:
  - src/mhs/discovery.py
  - src/mhs/books.py
  - src/mhs/pipeline/stages/book.py
last_verified: 2026-09-03
---

# 02. 알파 신호 및 멀티 호라이즌 아키텍처

## 1. 개요 (Overview)
금융 시계열은 단일한 시간 척도(Time Scale)로 움직이지 않습니다. 단기(수십 시간)에서는 유동성 충격과 오버슈팅에 의한 반전(Mean-Reversion)이 발생하며, 중장기(수백 시간)에서는 자금 유입과 구조적 트렌드에 따른 모멘텀(Momentum)이 시장을 지배합니다.

MHS는 단기 Reversal과 중장기 Momentum이라는 두 상반된 경제적 현상을 분리된 북(Book)으로 모델링하고, 통계적 엄밀성을 갖춘 디스커버리 게이트와 호라이즌 앙상블 기법을 통해 견고한 알파를 추출합니다.

---

## 2. Fast Reversal vs Slow Momentum 북 구조

| 구분 | Fast Reversal Book | Slow Momentum Book |
|---|---|---|
| **측정 기간 (Horizon)** | 48시간 (`48h`) | 72시간 ~ 504시간 (`72h ~ 504h`) |
| **방향 부호 (Sign)** | **-1** (오버슈팅 시 매도, 과매도 시 매수) | **+1** (상승 자산 매수, 하락 자산 매도) |
| **알파 원천** | 단기 유동성 흡수 및 일시적 가격 왜곡 수렴 | 중장기 자금 흐름(Flow) 및 트렌드 지속성 |
| **가중치 부여 방식** | 횡단면 랭크 기반 선형 가중치 | 횡단면 랭크 기반 선형 가중치 |
| **Phase 1 자본 비중** | **0.0 (0%)** (게이트 미달) | **1.0 (100%)** (게이트 통과) |

---

## 3. 디스커버리 및 퀄리피케이션 게이트 (Discovery & Qualification Gate)

백테스트 전체 구간에서 단 한 해만 운 좋게 특출난 수익을 거둔 호라이즌이 평균값(Mean)을 왜곡하여 선택되는 과적합(Overfitting)을 차단하기 위해 **Worst-Year Robustness 검증**([`select_horizon_by_discovery_qualification`](file:///home/kth/crypto-pilot/src/mhs/discovery.py))을 거칩니다.

```text
호라이즌 후보군 (Reversal: 24h~168h / Momentum: 72h~504h)
       │
       ▼ [1단계] Discovery 구간 (2021~2022년)
연도별 oriented net t-stat 계산 ──► Worst-Year(최솟값) |t| >= 2.0 통과 후보 선별
       │
       ▼ [2단계] Qualification 구간 (2023년 Out-of-Sample)
완전 분리된 2023년 데이터에서 동일한 부호 방향 및 |t| >= 2.0 재검증
       │
       ▼ [최종 판정]
두 구간 모두 통과한 호라이즌만 자본 투입 승인
```

- **Discovery 단계 (2021~2022년)**:
  - 후보 호라이즌 각각에 대해 2021년과 2022년의 순비용 차감 후 t-stat(oriented net t-stat)을 산출합니다.
  - 최악의 해(Worst-year)의 통계적 유의성이 임계값($|t| \ge 2.0$)을 충족하지 못하면 탈락합니다.
- **Qualification 단계 (2023년)**:
  - Discovery를 통과한 후보들을 완전히 봉인되었던 2023년 OOS 데이터에서 동일한 방식으로 검증합니다.
  - 신호의 부호(방향)가 일관되고 $|t| \ge 2.0$을 유지하는 호라이즌만 최종 통과합니다.
- **실측 검증 결과**:
  - **Fast Reversal**: 단기 틱 수수료와 슬리피지 장벽으로 인해 Qualification 게이트를 통과하지 못하여 Phase 1 자본 배분 비중이 **0%**로 고정되었습니다.
  - **Slow Momentum**: 대다수 호라이즌이 높은 t-stat으로 두 게이트를 모두 견고하게 통과했습니다.

---

## 4. 호라이즌 동일가중 앙상블 (RC-2, Horizon Ensemble)

과거 데이터에서 가장 성과가 좋았던 단 하나의 호라이즌(예: 딱 168시간)만 선택하는 Argmax 방식은, 시장의 주기가 조금만 바뀌어도 예측력이 급격히 저하되는 **고분산(High Variance) 취약점**을 갖습니다.

MHS는 이를 해결하기 위해 Slow 모멘텀에 **동일가중 호라이즌 앙상블([`equal_weight_book_ensemble`](file:///home/kth/crypto-pilot/src/mhs/books.py))**을 적용합니다:

```text
72h, 96h, 120h, 144h, ..., 480h, 504h (24시간 간격 총 19개 호라이즌)
  ├─ Horizon 72h Book  (Target Weights w_72)
  ├─ Horizon 96h Book  (Target Weights w_96)
  ├─ ...
  └─ Horizon 504h Book (Target Weights w_504)
        │
        ▼
  동일가중 평균: w_ensemble = (1 / 19) * Σ w_h
```

### 앙상블의 경제적 효과
1. **신호 확신도에 비례하는 포지션 크기**:
   - 19개 호라이즌의 의견이 일치할 때(단기/중기/장기 모멘텀이 모두 강세일 때)만 해당 자산의 포지션 크기가 최대로 확대됩니다.
   - 호라이즌 간 의견이 엇갈리면 자연스럽게 상쇄되어 포지션이 축소되므로 노이즈 장세에서 손실을 방어합니다.
2. **파라미터 민감도 제거**:
   - 특정 윈도우 길이에 대한 과적합 위험이 사라지고, 시간 횡단면 전체에 걸친 견고한 알파를 확보합니다.

---

## 5. 보조 슬리브 아키텍처 (Secondary Sleeves)

### 1) 트렌드 슬리브 (Trend Sleeve)
- **소스**: [`src/mhs/trend_sleeve.py`](file:///home/kth/crypto-pilot/src/mhs/trend_sleeve.py)
- **개념**: 336시간부터 1440시간까지의 장기 시계열 모멘텀을 측정하여, 횡단면 롱숏 포트폴리오에 절대 모멘텀(Time-Series Momentum) 트렌드 필터를 오버레이합니다.
- **역할**: 시장 전반의 장기 추세 방향과 일치하는 포지션에 가중치를 부여하고, 추세 역행 포지션의 익스포저를 억제합니다.

### 2) 펀딩 캐리 슬리브 (Funding Carry Sleeve)
- **소스**: [`src/mhs/funding.py`](file:///home/kth/crypto-pilot/src/mhs/funding.py)의 [`funding_carry_execution_book`](file:///home/kth/crypto-pilot/src/mhs/funding.py)
- **개념**: 최근 168시간 누적 펀딩비가 극단적으로 양수인 자산(선물 프리미엄 과열)을 숏하고, 음수인 자산(선물 할인 과열)을 롱하여 펀딩비 수익(Funding Yield)을 수취합니다.
- **배분 비중**: [`FUNDING_CARRY_SLEEVE_WEIGHT = 0.30`](file:///home/kth/crypto-pilot/src/mhs/params.py) (설정 시 모멘텀 북과 결합).

---

## 6. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 클래스 |
|---|---|---|
| 호라이즌 디스커버리 | [src/mhs/discovery.py](file:///home/kth/crypto-pilot/src/mhs/discovery.py) | `select_horizon_by_discovery_qualification`, `yearly_net_t_diagnostic` |
| 북 생성 및 앙상블 | [src/mhs/books.py](file:///home/kth/crypto-pilot/src/mhs/books.py) | `rank_weight_book`, `equal_weight_book_ensemble` |
| 북 빌드 파이프라인 스테이지 | [src/mhs/pipeline/stages/book.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/book.py) | `build_books` |
| 트렌드 슬리브 | [src/mhs/trend_sleeve.py](file:///home/kth/crypto-pilot/src/mhs/trend_sleeve.py) | `build_trend_sleeve_weights` |
| 펀딩 캐리 | [src/mhs/funding.py](file:///home/kth/crypto-pilot/src/mhs/funding.py) | `funding_carry_execution_book` |
