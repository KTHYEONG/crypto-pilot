# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-10 (5차 — 체결 사다리 + discovery/qualification 게이트 실측)
- **Registered ADRs**:
  - `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION` (1차: 실행 roster 마스킹 후 dollar-neutral/unit-gross 재정규화)
  - `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT` (2차: roster 진입/이탈 히스테리시스 + causal inverse-vol tilt)
  - `ADR_20260810_MHS_BOOK_ADMISSION_VOL_MASK` (3차: book admission 동결 + regime vol_mean roster 마스킹)
  - `ADR_20260810_MHS_TOUCH_PROXY_FILL_MEASUREMENT` (strict-vs-touch 교차 판정 가설 반증, 30분 타임아웃발 비용 폭증 확인)
  - `ADR_20260810_MHS_BLEND_GRID_COUPLING_FIX` (4차: blend 실행 격자/spec을 admission 가중치 기반으로 재결합)
  - `ADR_20260810_MHS_EXECUTION_LADDER_AND_DISCOVERY_GATE` (5차: 에스컬레이팅 체결 사다리 + discovery/qualification 게이트 구현·실측, 이번 갱신)
- **이번 실측 근거**: `docs/specs/mhs_execution_ladder_and_discovery_gate.md`
  §1(체결 사다리, opt-in `--ladder-diagnostic`) + §2(discovery/qualification
  horizon 선정 게이트, `src/mhs/discovery.py`)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact tier), [`docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json) (`--ladder-diagnostic --output-tier full`)
- **Execution Status**: `COMPLETE`
- **Run Metadata**: 2021-01-01~2025-12-31, `execution_universe_size=30`, `execution_timeframe=5m`, `eligible_symbols=446`

---

## 0. 기본 실행 경로 (변화 없음, 4차 수정 유지 확인)

이번 5차 작업은 둘 다 **opt-in 진단**으로 설계됐다(§1.6/§2.4 거버넌스: 측정
전 승격 금지). 기본 CLI 실행 경로는 4차 수정 이후와 수치가 완전히 동일하다.

| Book | step_hours | fills | Autocorr Sharpe | MDD | Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **slow_momentum** | 24 | 118,557 | -1.8373 | -99.2360% | 5.6211 |
| **blend** | 24 | 118,557 | -1.8373 | -99.2360% | 5.6211 |

**Research GO: 여전히 FALSE** (`folds_passed=0/3`, reason codes:
`CAPITAL_INVARIANT_BREACH`, `INCOMPLETE_ANCHORED_FOLD`,
`PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`,
`UNSPECIFIED_POLICY`).

---

## 1. Part B 실측 — discovery/qualification horizon 선정 게이트

`select_horizon_by_discovery_qualification`(discovery 2021-2023 worst-year-robust
선택 → qualification 2024-2025 단일 재확인, 재탐색 없음)을 두 sign 계열에
실제 데이터로 실행했다.

| Family | Discovery 최고 점수 (worst-year net_t, 2.64bps) | 최고 후보 | Selected | Qualification | Admitted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| reversal (sign=-1) | **-1.812** | 168h | `None` | 미평가 | `False` |
| momentum (sign=+1) | **-0.736** | 336h | `None` | 미평가 | `False` |

전체 discovery 점수(모든 후보, worst-year net_t @2.64bps):

- reversal: 24h=-3.932, 48h=-3.599, 72h=-3.441, 96h=-2.082, 120h=-2.024, **168h=-1.812**
- momentum: 72h=-1.077, 120h=-1.693, 168h=-0.999, 240h=-1.399, **336h=-0.736**, 504h=-1.031

**핵심 발견**: 이전(제거된) 단일구간 12점 스윕에서 유일하게 |t|>=2.0 문턱을
넘었던 "168h reversal"(전체 5년 aggregate t=-2.16)이, **연도별 최소값
기준으로는 -1.81로 문턱 미달**이다 — 특정 연도(단일 구간)에 성과가 몰려 있었을
뿐 discovery 3개 연도 전체에서 강건하지 않았다는 뜻. §2.4에서 명시한 대로
이 결과로 어떤 후보도 채택하지 않는다(fail-closed, qualification 자체가
평가되지 않음). **현재 방법론(단순 rank-weight momentum/reversal,
tranche_count=1)으로는 discovery 구간에서 강건한 edge가 어느 horizon에서도
재현되지 않는다** — fast/slow 투 밴드 아키텍처의 파라미터 문제가 아니라
접근 방식 자체를 재검토할 근거로 봐야 한다.

---

## 2. Part A 실측 — 에스컬레이팅 체결 사다리 (`--ladder-diagnostic`, K=4)

`OHLCV_LADDERED_PROXY`(주문을 4개 tranche로 분할, 실패 시 선형 리프라이스,
마지막 tranche만 시장가 폴백)를 strict proxy와 나란히 5년 전체 재생했다
(blend/slow_momentum, 두 북이 격자 결합 수정 이후 동일하므로 하나로 보고).

| 지표 | strict (K=1, 기존) | ladder (K=4) | 변화 |
| :--- | :--- | :--- | :--- |
| intent shortfall (bps) | 1000.29 | **875.43** | **-12.5%** |
| fill_count | 81,018 | 293,898 | tranche 분할로 3.6배 (예상된 동작) |
| unfilled_count(최종 시장가 폴백) | 37,539 | 36,836 | 거의 동일 |
| naive Sharpe | -0.6982 | **-0.9093** | **악화** |
| forced_exit_count | 0 | 27 | 미미 |

**해석 — 단순 개선이 아니라 트레이드오프**: shortfall(bps, aggregate 평균
지표)은 의도대로 줄었으나, risk-adjusted 지표(naive Sharpe)는 오히려
나빠졌다. tranche 2~K가 시장 쪽으로 리프라이스된 가격에서 체결되면서, 이전엔
"결국 decision_price에 전량 체결됐을" 다수의 평범한 주문에 작은 비용이
새로 추가된 것으로 보인다 — 최악의 꼬리 이벤트(단발 대형 손실)는 줄었지만
평균적으로는 더 많은 주문이 소액 손실을 보게 돼 분산 대비 평균(Sharpe)이
악화됐다. `unfilled_count`가 거의 그대로인 것도 시사적이다: 대다수 주문이
여전히 마지막 tranche까지 가서 시장가로 떨어진다 — "추세 지속 구간에서
가격이 안 돌아온다"는 근본 원인 자체는 안 풀렸고, 그 손실을 4등분해서 나눠
맞은 것에 가깝다.

**결론**: 현재 파라미터(K=4, 선형 리프라이스)로는 primary 승격을 권하지
않는다(§1.6 거버넌스 유지). 비선형 리프라이스(초반 느리게·후반 빠르게)나
다른 K 값 스윕이 다음 후보이나, 이번 spec 범위 밖의 별도 실측 과제로 남긴다.

---

## 3. 종합 — 다음 단계

두 해법 모두 "쉬운 승리"가 아니었다는 것이 이번 실측의 핵심 결론이다:

- **Part B**: 이전 스윕에서 유일하게 보였던 "탈출구"(168h reversal)가
  discovery worst-year-robust 기준에서 사라졌다 — horizon 재선정으로는
  현재 접근 방식의 근본 한계를 넘지 못한다는 증거.
- **Part A**: 체결 사다리는 꼬리 위험은 줄이지만 평균 비용이 늘어 Sharpe
  기준으로는 개선이 아니다 — 30분 타임아웃 자체보다 "추세가 지속되면
  가격이 안 돌아온다"는 시장 구조 자체가 더 근본적인 제약일 가능성.
- 두 결과를 함께 보면, 남은 레버는 파라미터 재조정(§1/§2 스윕)보다
  **신호·실행 접근 방식 자체의 재설계**(예: 다른 신호 축, 다른 체결
  스케줄 형태)일 가능성이 높다 — 다음 spec 사이클의 핵심 질문으로 남긴다.
