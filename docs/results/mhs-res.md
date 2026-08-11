# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-11 (7차 — Research-GO primary를 realistic immediate-taker로 교체 실측)
- **Registered ADRs**:
  - `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION` (1차: 실행 roster 마스킹 후 dollar-neutral/unit-gross 재정규화)
  - `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT` (2차: roster 진입/이탈 히스테리시스 + causal inverse-vol tilt)
  - `ADR_20260810_MHS_BOOK_ADMISSION_VOL_MASK` (3차: book admission 동결 + regime vol_mean roster 마스킹)
  - `ADR_20260810_MHS_TOUCH_PROXY_FILL_MEASUREMENT` (strict-vs-touch 교차 판정 가설 반증, 30분 타임아웃발 비용 폭증 확인)
  - `ADR_20260810_MHS_BLEND_GRID_COUPLING_FIX` (4차: blend 실행 격자/spec을 admission 가중치 기반으로 재결합)
  - `ADR_20260810_MHS_EXECUTION_LADDER_AND_DISCOVERY_GATE` (5차: 에스컬레이팅 체결 사다리 + discovery/qualification 게이트 구현·실측)
  - `ADR_20260811_MHS_DISCOVERY_2021_GAP_AND_DENSE_GRID` (6차: 2021 결손 데이터-커버리지 원인 규명, discovery.py 가중치 재사용 리팩터링, horizon 격자 조밀화 + tranche_count 배선 수정 실측)
  - `ADR_20260811_MHS_REALISTIC_EXECUTION_PRIMARY_SWAP` (7차: Research-GO primary를 `OHLCV_STRICT_PROXY`(patient 30분 대기)에서 `OHLCV_IMMEDIATE_TAKER`로 교체, stress를 ×3 cost bound로 교체, 이번 갱신)
- **이번 실측 근거**: `docs/specs/mhs_realistic_execution_primary_swap.md`
  (30분 타임아웃발 동조화 taker 폭주가 성과 파괴 원인이라는 실측 규명 §0,
  primary/stress/reference 교체 설계 §1)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact tier), [`docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json) (`--ladder-diagnostic --output-tier full`)
- **Execution Status**: `COMPLETE`
- **Run Metadata**: 2021-01-01~2025-12-31, `execution_universe_size=30`, `execution_timeframe=5m`, `eligible_symbols=446`

---

## 0. 기본 실행 경로 (7차: Research-GO primary 교체 실측)

5차(사다리)·6차(discovery 조밀화)는 opt-in 진단이었지만, 이번 7차는 **기본
primary를 바꾼다**: 5년 전체 재실행에서 strict-proxy(30분 대기)가 주문 후
정확히 30분 뒤에 로스터 전 종목이 동시에 taker로 폴백해 단발 꼬리 손실
(-41.6%/5분봉 등)이 성과를 파괴했음을 실측으로 규명했다. 참여율이 분당 거래량의
1e-9 수준이라 footprint 회피용 대기의 경제적 근거가 없으므로, primary를
`OHLCV_IMMEDIATE_TAKER`로 교체하고 stress는 동일 체결에 비용만 ×3
(`SPREAD_AND_COST_X3`, taker 15bp + slippage 9bp) 가정으로 교체했다. 기존
strict-proxy는 `patient_reference` 진단 필드로만 보존된다(Research GO 미게이팅).

| Metric | primary (`OHLCV_IMMEDIATE_TAKER`) | stress (×3 cost) | patient_reference (구 strict-proxy) |
| :--- | :--- | :--- | :--- |
| Daily autocorr-adjusted Sharpe | **+0.4317** | — | — |
| Naive Sharpe | +0.0991 | **+0.0250** | **-0.6913** |
| Final equity (5y, from 1.0) | **1.2549** | 0.7198 | 0.0081 |
| Max drawdown | **-45.74%** | -58.43% | -99.22% |

(slow_momentum/blend 두 북이 격자 결합 수정 이후 동일하므로 blend 기준으로 보고.
`fast_reversal`은 immediate-taker에서도 여전히 `CAPITAL_INVARIANT_BREACH` —
이는 실행 아티팩트가 아니라 reversal 신호 자체의 실제 문제다.)

**Research GO: 여전히 FALSE** (`folds_passed=1/3`, reason codes:
`CAPITAL_INVARIANT_BREACH`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`,
`STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY`). fold별로는 fold0
(autocorr -0.789, stress -0.374), fold1 (autocorr -0.517, stress -0.334)이
기각, fold2 (autocorr +1.604, stress +0.332)만 통과했다.

**정직한 해석**: +0.432는 -1.837보다 ~2.3 Sharpe만큼 큰 개선이지만, 이는
"실행 모델링 오류 제거"의 결과일 뿐이다. 최종 자산 1.255(연 복리로 양수)로
전환됐고 꼬리 손실 메커니즘은 사라졌지만, **Research GO floor(0.6)에 아직
미달** — 이번 수정은 GO 달성이 아니라 선행 전제 조건 수정이며, 상위 레버는
여전히 신호 자체다.

---

## 1. Part B 실측 — discovery/qualification horizon 선정 게이트

`select_horizon_by_discovery_qualification`(discovery 2021-2023 worst-year-robust
선택 → qualification 2024-2025 단일 재확인, 재탐색 없음)을 두 sign 계열에
실제 데이터로 실행했다.

**2026-08-11 조밀 격자 재실행**: `discovery.py`를 `_horizon_weights`/
`_score_masked_net_t`로 리팩터링(horizon당 가중치 1회 재사용, byte-identical —
기존 6개 unit 시나리오 무변경 통과로 증명)하고, 그 절약분으로 horizon 후보를
24h 균등 격자로 조밀화했다(reversal 6→7개, momentum 6→19개). 동시에
`evaluation.py`의 `--discovery-gate` CLI 배선에 빠져 있던 `tranche_count=8`
(production 관례)도 함께 고쳤다. 실측은 실제 446 심볼 5개년 패널
(`data/futures/ohlcv/1h`)에 대해 discovery-gate 단계만 별도 실행했으며
59.5초가 걸렸다(전체 패널 로드 포함 66.1초).

| Family | Discovery 최고 점수 (worst-year net_t, 2.64bps, tranche_count=8) | 최고 후보 | Selected | Qualification | Admitted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| reversal (sign=-1) | **-1.175** | 24h | `None` | 미평가 | `False` |
| momentum (sign=+1) | **+0.493** | 360h | `None` | 미평가 | `False` |

전체 discovery 점수(모든 후보, 올바른 worst-year net_t @2.64bps,
tranche_count=8, 조밀 격자):

- reversal (7개 후보): **24h=-1.175**, 48h=-1.107, 72h=-0.712, 96h=-0.752, 120h=-0.283, 144h=-0.707, 168h=-1.057
- momentum (19개 후보): 72h=-0.098, 96h=-0.117, 120h=-0.357, 144h=+0.113, 168h=+0.107, 192h=-0.393, 216h=-0.359, 240h=-0.168, 264h=-0.229, 288h=+0.138, 312h=+0.140, 336h=+0.220, **360h=+0.493**, 384h=+0.354, 408h=+0.121, 432h=-0.142, 456h=-0.120, 480h=-0.128, 504h=-0.191

**조밀화의 실제 효과**: 6개 후보로는 336h가 최고(+0.220)였지만, 24h 간격으로
채워보니 진짜 국지 최댓값은 **360h(+0.493)**에 있었다 — 336h는 그 근처의
어깨였을 뿐 정점이 아니었다. 144h~408h 구간에서 양의 점수가 몰려 있고
192h~264h는 국지적으로 음전환하는 이봉(bimodal) 패턴도 처음 드러났다. 그래도
+0.493은 여전히 `admission_t=2.0`에 한참 못 미쳐 **결론(admitted=False)은
바뀌지 않는다** — 격자가 성겨서 edge를 놓치고 있었던 것은 아니라는 뜻이다.
reversal은 새로 채운 144h(-0.707)를 포함해도 최고 후보가 여전히 24h(-1.175)
그대로다.

**핵심 발견 (수정 후에도 결론 유지)**: 이전에 보고한 "reversal 최고 후보
168h, worst-year=-1.812"는 sign-unaware `min()` 버그로 계산된 틀린 수치다.
올바른 sign-aware worst-year 기준 reversal 최고 후보는 **24h(-1.175)** 이고,
여전히 |t| < 2.0이라 어느 후보도 채택되지 않는다(fail-closed, qualification
미평가). 즉 결론(admitted=False) 자체는 우연히 바뀌지 않았지만, 이전에는 그
결론이 "알고리즘이 reversal을 원천적으로 통과 못 시키게 돼 있어서" 나온
것이었고 이제는 "데이터가 실제로 약해서" 나온 것임이 확인됐다. **현재
방법론(단순 rank-weight momentum/reversal, tranche_count=8)으로는 discovery
구간에서 강건한 edge가 어느 horizon에서도 재현되지 않는다** — fast/slow 투
밴드 아키텍처의 파라미터 문제가 아니라 접근 방식 자체를 재검토할 근거로 봐야
한다.

(참고) discovery 2021년은 모든 horizon에서 `net_t`가 non-finite(계산 불가)이며,
원인을 실측으로 규명해 종결했다(`docs/specs/mhs_discovery_2021_gap_and_dense_grid.md`
§0). liquid_half_eligibility(720시간 lookback)의 eligible 심볼 수가 2021년
전체~2022년 1분기까지 평균 3개(min_symbols=8 미만)로 고정돼 있다가 2022년
2분기부터 18개로 늘어난다 -- 446개 유니버스 대부분이 2022년 중반 이후 상장된
실제 데이터 커버리지 한계이며 파이프라인 버그가 아니다. MHS_DISCOVERY_START를
2021-01-01에서 2022-04-01로 옮겨도 실측 결과 discovery 점수는 소수점 6자리까지
동일했다(연 단위 버킷팅 + non-finite 연도 제외가 이미 이 차이를 흡수하기 때문) --
discovery 구간 시작점은 유효한 레버가 아니다.

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
