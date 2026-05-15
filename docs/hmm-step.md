# HMM 고도화 로드맵: Phase 3.8 ~ 4.0 (Tail-Risk Specialization)

이 문서는 v10.7.0 (Student-t HMM) 이후, 꼬리 방어 커버리지(Tail-Capture)와 위기 탐지 정밀도(Crisis-Prec)의 임계 돌파를 위한 고도화 방안을 정의한다.

## 1. 현황 및 병목 분석 (v10.7.0 기준)
- **달성**: JAX 기반 Student-t 도입으로 `Avg-Duration` 22.3 bars 달성 (안정성 확보).
- **병목**: Student-t의 대칭적 특성으로 인해 하방 꼬리(Negative Tail) 감지가 지연됨.
  - `Tail-Capture`: 75.4% (목표 > 85%)
  - `Crisis-Prec`: 9.7% (목표 > 20%)
- **원인**: EM 학습이 전체 우도(Likelihood) 최적화에 집중되어, 빈도가 낮은 극단적 BEAR/CRISIS 상태의 사전 확률(Prior)이 과소 평가됨.

---

## 2. [Phase 3.8] 목적 함수 기반 EM 최적화 (Outcome-Weighted NLL)
Baum-Welch 학습 단계에서 단순 우도가 아닌, 리스크 관리 목적함수를 직접 반영한다.

- **핵심 로직**:
  - NLL(Negative Log-Likelihood) Loss 함수에 **Outcome-Penalty Term** 추가.
  - 실제 과거 데이터 중 하위 5% 수익률 구간에서 HMM이 BEAR/CRISIS 상태를 점유하지 못할 경우 지수적 패널티 부여.
- **기대 효과**:
  - HMM 자체가 "수익을 지키는 방향"으로 파라미터(Means, Covs, Transitions)를 최적화.
  - 별도의 Bias 조절 없이도 BEAR 상태의 민감도(Recall) 자연 상승.

## 3. [Phase 3.9] 경로 의존적 메타 라벨링 (Triple-Barrier Meta-Labeling)
Supervised Layer의 라벨링 방식을 고정 시간 윈도우에서 가격 경로 중심으로 전환한다.

- **핵심 로직**:
  - **Triple-Barrier**: 1) ATR 기반 하단 손절선, 2) 상단 익절선, 3) 시간 제한.
  - 하단 손절선에 먼저 닿는 경우만 '진정한 꼬리 위험'으로 라벨링하여 학습.
- **기대 효과**:
  - 변동성 군집(Vol Clustering)을 반영한 정교한 라벨링.
  - `Crisis-Prec` (정밀도) 20% 돌파를 위한 노이즈 제거.

## 4. [Phase 4.0] 비대칭 분포 모델 교체 (Skewed-t HMM)
분포 모델 자체를 코인 시장의 비대칭성에 맞게 교체한다.

- **핵심 로직**:
  - 기존 Student-t 방출 모델(Emission)을 **Skewed-t (비대칭 스튜던트 T)**로 확장.
  - 각 상태별 비대칭 파라미터($\lambda$)를 JAX EM 과정에서 함께 추정.
- **기대 효과**:
  - 하방 쏠림이 강한 코인 폭락장의 특성을 수학적으로 완벽히 수용.
  - 적은 데이터 변화로도 즉각적인 체제 전환(Early Warning) 가능.

---

## 5. 단계별 실행 계획 및 KPI

| 단계 | 주요 작업 | 목표 지표 | 우선순위 |
| :--- | :--- | :--- | :--- |
| **Phase 3.8** | Outcome-Weighted NLL 구현 | Tail-Capture > 85% | 1순위 |
| **Phase 3.9** | Triple-Barrier Meta-Labeling 도입 | Crisis-Prec > 20% | 2순위 |
| **Phase 4.0** | Skewed-t HMM 아키텍처 전환 | All-Pass (Audit v20) | 3순위 |

---

## 6. 기술적 참고 사항 (Technical Notes)
- 모든 구현은 `JAX`의 CPU-only 컴파일 환경을 유지해야 함.
- EM 수렴 속도 저하를 방지하기 위해 가중치 패널티(Penalty Weight)는 초기 0.1에서 시작하여 점진적으로 튜닝.
- `Downside_Vol_Z`와 `OI_Delta_Z`를 Skewed-t의 핵심 비대칭 피처로 활용.
