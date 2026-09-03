---
title: MHS Architecture - 06. Purged Validation & Research GO
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/evidence.py
  - src/mhs/research_go.py
  - src/mhs/statistics.py
  - src/mhs/pipeline/stages/fold.py
  - src/mhs/pipeline/stages/diagnostic.py
  - src/mhs/params.py
change_triggers:
  - src/mhs/evidence.py
  - src/mhs/research_go.py
  - src/mhs/pipeline/stages/fold.py
last_verified: 2026-09-03
---

# 06. 교차 검증 및 Research GO 평가 체계

## 1. 개요 (Overview)
금융 전략 검증에서 단순 In-Sample 맞춤이나 셔플(K-fold) 방식은 시계열 자기상관으로 인한 치명적인 정보 누출(Data leakage)을 일으킵니다.

MHS는 마르코프 연쇄와 모멘텀 지평에 대응하는 **168시간(1주) 엠바고/퍼징(Purged & Embargoed)이 적용된 3-Fold Walk-Forward 교차 검증**과 **9대 합성 스트레스 시나리오**, 그리고 **블록 부트스트랩 배포 준비도 평가**를 거쳐 전략의 실전 배포 가능성(Research GO)을 판정합니다.

---

## 2. 3-Fold Level 2 Anchored Purged Walk-Forward 검증 체계

검증은 [`src/mhs/evidence.py`](file:///home/kth/crypto-pilot/src/mhs/evidence.py)의 [`phase_1_anchored_purged_folds`](file:///home/kth/crypto-pilot/src/mhs/evidence.py) 및 [`src/mhs/pipeline/stages/fold.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/fold.py)에 의해 엄격히 집행됩니다:

```text
Fold 0: [2021 ~ 2022 Train] ──(168h Purge/Embargo)──► [2023 Validation OOS]
Fold 1: [2021 ~ 2023 Train] ──(168h Purge/Embargo)──► [2024 Validation OOS]
Fold 2: [2021 ~ 2024 Train] ──(168h Purge/Embargo)──► [2025 Validation OOS]
```

- **Anchored (누적 확장)**: 실제 시장 운용 환경과 동일하게 과거 데이터가 계속 누적되며 Train 구간이 확장됩니다.
- **168시간 Purge & Embargo**:
  - 최대 168시간(7일) 모멘텀 신호의 잔여 자기상관이 Train 종료 시점과 Validation 시작 시점 사이에 영향을 미치지 않도록 168시간 윈도우를 강제 삭제(Purge) 및 격리(Embargo)합니다.
- **폴드별 독립 시뮬레이션**:
  - 각 폴드는 해당 시점까지의 Train 데이터만으로 적합된 파라미터(또는 상위 확정 배포 오버레이)를 사용해 미래 OOS 구간을 100% 독립적으로 검증합니다.

---

## 3. 9대 합성 스트레스 시나리오 (Synthetic Stress Scenarios)

단순한 역사적 재현을 넘어, MHS는 포트폴리오를 파괴할 수 있는 9가지 극단적 가상 스트레스 상황을 인위적으로 주입하여 회복력을 검증합니다 ([`synthetic_stress_scenarios`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/assemble.py)):

1. **`BTC_DOWN_10`**: 비트코인 10% 순간 급락 충격.
2. **`BTC_DOWN_20`**: 비트코인 20% 메가 크래시 충격.
3. **`ALT_BETA_UP`**: 알트코인 전반의 시장 베타 급증.
4. **`XS_CORRELATION_ONE`**: 자산 간 횡단면 상관관계가 1.0으로 수렴 (롱숏 분산 효과 상실).
5. **`SPREAD_AND_COST_X3`**: 호가 스프레드 및 체결 수수료 3배 폭등.
6. **`PASSIVE_FILL_DEGRADATION`**: 지정가 주문 체결률 50% 급감.
7. **`FUNDING_EXTREME`**: 선물 펀딩비가 극단적 양수/음수로 폭등.
8. **`LIQUIDITY_DETERIORATION_50PCT`**: 유니버스 내 50% 심볼의 유동성 급감.
9. **`VENUE_API_OUTAGE_30M`**: 30분간 거래소 API 다운 및 주문 불가 상태.

---

## 4. 꼬리 위험 및 부트스트랩 배포 준비도 (Deployment Readiness)

### 1) 꼬리 민감도 및 윈저 곡선 (Winsorization Curve)
- 극소수의 이상치 대형 이익이 전체 샤프 지수를 왜곡했는지 감시합니다.
- 상위 수익률을 50%, 30%, 20%, 10%로 잘라낸 윈저화 손익 곡선과 최악 이벤트 1건 제외 샤프([`leave_worst_event_out_sharpe`](file:///home/kth/crypto-pilot/src/mhs/evidence.py))를 측정하여 소수 대박 코인에 의존하지 않는 견고함을 입증합니다.

### 2) Stationary Block Bootstrap (2,000 Paths)
- 168시간(1주일) 블록 크기로 2,000회의 역사적 손익 경로를 재표본화하여 배포 리스크를 정량화합니다:
  - **MDD 20% 초과 확률 (`probability_mdd_over_20pct`)**
  - **MDD 30% 초과 확률**
  - **파산 확률 (`ruin_probability`, 40% 이상 영구 자본 손실)**

---

## 5. 최신 실측 성능 및 Research GO 게이트 판정 결과

아래는 **위원회 레짐 적응형 트랜치(기본값, 2026-08-17 실측)** 기준 공식 진단 결과입니다:

| 평가지표 (Metric) | 이전 17차 (고정 평활) | **위원회 레짐 적응형 (최신)** | Research GO 승인 기준 | 판정 결과 |
| :--- | :---: | :---: | :---: | :---: |
| **`primary_autocorr_sharpe`** | 0.5257 | **1.0792** | $\ge 0.60$ | **통과 (기준 대폭 상회)** |
| **`primary_geometric_cagr`** | 7.84% | **19.23%** | $> 0.0\%$ | **통과 (수익률 2.5배)** |
| **`primary_max_drawdown`** | -22.69% | **-17.05%** | $\le 25.0\%$ | **통과 (MDD 5.6%p 개선)** |
| **`deployment_readiness.calmar`**| 0.35 | **1.128** | $\ge 0.50$ | **통과** |
| **`stress_naive_sharpe` (x3 Cost)**| +0.1420 | **+0.8334** | $> 0.0$ | **통과 (비용 내구성 압도)** |
| **`research_go.folds_passed`** | 2 / 3 | **3 / 3 (100%)** | 3 / 3 전원 통과 | **통과** |
| **`research_go.reason_codes`** | 알파 미달 다수 | **`['UNSPECIFIED_POLICY']` 단일** | 무차단 (`[]`) | **절차 이슈만 잔여** |

### 세부 폴드별 OOS 실측치
- **Fold 0 (2023년 OOS)**: Autocorr Sharpe **1.158** (통과)
- **Fold 1 (2024년 OOS)**: Autocorr Sharpe **0.853** (통과)
- **Fold 2 (2025년 OOS)**: Autocorr Sharpe **3.387** (통과)
- **3개 폴드 전원 0.60 게이트 플로어를 여유 있게 상회**.

### 최종 상태 요약
알파의 통계적 우위, 비용 내구성, 자본 보존, Walk-Forward 일관성은 **완전히 입증**되었습니다. 현재 Research GO의 유일한 차단 사유는 `MHS_REGISTERED_POLICY_THRESHOLDS` 미등록에 따른 `UNSPECIFIED_POLICY`뿐이며, 통계적 게이트가 아닌 정책 임계값 등록 절차에 해당합니다.

---

## 6. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 클래스 |
|---|---|---|
| 교차 검증 및 지표 계산 | [src/mhs/evidence.py](file:///home/kth/crypto-pilot/src/mhs/evidence.py) | `phase_1_anchored_purged_folds`, `autocorrelation_adjusted_sharpe`, `compute_deployment_readiness` |
| Research GO 최종 게이트 | [src/mhs/research_go.py](file:///home/kth/crypto-pilot/src/mhs/research_go.py) | `evaluate_research_go`, `_drawdown_budget_reasons` |
| 통계 유의성 계산 | [src/mhs/statistics.py](file:///home/kth/crypto-pilot/src/mhs/statistics.py) | `deflated_sharpe_ratio`, `probabilistic_sharpe_ratio` |
| 폴드 파이프라인 스테이지 | [src/mhs/pipeline/stages/fold.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/fold.py) | `run_folds` |
| 진단 리포트 조립 | [src/mhs/pipeline/stages/diagnostic.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/diagnostic.py) | `assemble_report` |
