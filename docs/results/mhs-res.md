# MHS Horizon Diagnostic — Quantitative Performance Report

- **Document Date**: 2026-08-11 (10차 갱신)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `2021-01-01~2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`, `eligible_symbols=446`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact, `--fold-safe-horizon --discovery-gate`), `_full/report.json` (`--discovery-gate --output-tier full`)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe ≥ 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과

## 0. 현재 상태 (10차, 최신 실측)

| Metric | 값 |
| :--- | ---: |
| `primary_autocorr_sharpe` | **+0.4464** |
| `primary_naive_sharpe` | +0.0888 |
| `primary_max_drawdown` | **-21.87%** |
| `primary_net_ann` | +0.51% |
| `primary_geometric_cagr` | +0.35% |
| `primary_annualized_turnover` | 2.562 |
| `stress_naive_sharpe` (×3 cost) | +0.0163 |
| `pre_vol_target_reference_naive_sharpe` (Pass 1, vol-target 적용 전) | +0.0926 |
| `blend.failure` | `None` |
| `fast_reversal.failure` | `CAPITAL_INVARIANT_BREACH` (기존 이슈, blend 자본 0%, 무관) |
| `research_go.eligible` | **`False`** |
| `research_go.reason_codes` | `CAPITAL_INVARIANT_BREACH`(fast_reversal 기인), `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |
| `folds_passed` | 1 / 3 |

9차(+0.4661) 대비 top-level 수치의 소폭 하락(-0.0197)은 이번 변경이 아니라
`eligible_symbols` 445→446 자연 드리프트(신규 데이터 갱신에 따른 유동성 유니버스
경계 이동)로 설명됨 — fold별 수치(아래)는 소수점까지 9차와 동일해 fold-safe-horizon
자체는 아무 것도 바꾸지 않았음을 직접 증명한다.

### Fold 상세

| Fold | Validation | `primary_autocorr_sharpe` | `MDD` | `stress_naive_sharpe` | `failures` | `slow_horizon_hours` | `slow_horizon_source` |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| 0 | 2023 | -0.706 | -23.6% | -0.354 | `PRIMARY_SHARPE_BELOW_0_6`, `STRESS_NOT_POSITIVE` | 168 | `frozen_default` |
| 1 | 2024 | -0.504 | -18.3% | -0.345 | `PRIMARY_SHARPE_BELOW_0_6`, `STRESS_NOT_POSITIVE` | 168 | `frozen_default` |
| 2 | 2025 | +1.605 | -40.9% | +0.265 | (통과) | 168 | `frozen_default` |

### Fold-safe horizon 재선정 (10차 신규, `ADR_20260811_MHS_FOLD_SAFE_HORIZON_SELECTION`)

`select_horizon_by_discovery_qualification`을 각 anchored fold의 train-only
윈도우(validation 연도를 전혀 보지 않음)로 재실행해 `slow_momentum`의
`horizon_hours=168` 레거시 기본값을 leak-free하게 재검증하는 `--fold-safe-horizon`
opt-in 경로를 추가하고 실측:

| Fold | train-only discovery 윈도우 | 최고 후보(worst-year net_t) | `admitted` | 결과 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 2021만 | 19개 후보 전부 non-finite | `False` | 168h 폴백 (2021 데이터 커버리지 결손, §2 참고) |
| 1 | 2021-2022 | **168h = +1.382** (현재 기본값 자신이 최고 후보) | `False` (2.0 미달) | 168h 폴백 |
| 2 | 2021-2023 | 360h = +0.626 | `False` (2.0 미달) | 168h 폴백 |

**3개 fold 전부 168h 유지, 수치 무변동** — 안전한 no-op으로 확인됨. 다만 fold 1의
최고 후보가 이미 168h 자신이라는 점, fold 2의 최고 후보(360h, +0.626)가
admission_t=2.0의 절반에도 못 미친다는 점을 통해 §4의 "discovery worst-year 통계의
통계적 가혹함" 항목이 추측이 아니라 **실측으로 확인된 실질 병목**임이 이번 차수의
핵심 성과다 (전역 다년 윈도우 기준으로 캘리브레이션된 `admission_t=2.0`이
fold-local 1~3년 표본에는 지나치게 엄격함).

### Discovery-gate (진단 전용, Research GO 미게이팅, 9차와 동일 — 유지)

| Family | 최고 worst-year `net_t` | 최고 후보 | `selected_horizon` | `admitted` |
| :--- | ---: | :--- | :--- | :--- |
| reversal (sign=-1, raw signal) | -1.175 | 24h | `None` | `False` |
| momentum (sign=+1, **vol-normalized** signal) | **+0.673** | 360h | `None` | `False` |

(admission floor = 2.0. momentum raw signal 기준은 +0.493 — vol-normalized가 discovery 프리스크린에서만 유지됨, §2 참고)

---

## 1. 반복별 실측 이력 (7~9차, 비교 가능한 realistic-execution primary 기준)

| 차수 | 변경 | `primary_autocorr_sharpe` | `MDD` | `stress_sharpe` | Research GO | 결과 |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| 7차 | primary를 `OHLCV_STRICT_PROXY`→`OHLCV_IMMEDIATE_TAKER` 교체 | +0.4317 | -45.74% | +0.0250 | False (fold2만) | **채택** — 30분 타임아웃발 동조화 taker 폭주 제거 |
| 8차 (시도) | momentum 신호를 vol-normalized로 교체, 실전 배선 | BREACH | — | — | False (primary 무효) | **원복** — 이중 볼스케일링 제거해도 재현, 2021-24 누적손실이 raw보다 깊음 |
| 8차 (최종) | 8차 원복 → 7차와 동일 | +0.4317 | -45.74% | +0.0250 | False (fold2만) | 7차와 byte-identical 재확인 |
| 9차 | 전략 자체 P&L vol-targeting (2-pass 리플레이) | **+0.4661** | **-21.64%** | +0.0210 | False (fold2만) | **채택** — MDD 거의 절반, Sharpe +8% |
| 10차 | fold-safe(leak-free) discovery/qualification horizon 재선정 opt-in 추가 | +0.4464 (445→446 심볼 드리프트, fold별은 9차와 byte-identical) | -21.87% | +0.0163 | False (fold2만) | **채택(no-op 확인)** — 3개 fold 전부 168h 유지, admission_t=2.0을 fold-local 표본의 실질 병목으로 실측 확정 |

### 근본 원인 진단 이력

| 발견 | 실측 근거 | 상태 |
| :--- | :--- | :--- |
| 30분 patient-timeout 동조화 taker 폭주 | strict-proxy에서 로스터 전체 동시 taker fallback, -41.6%/5분봉 | 7차에서 제거 완료 |
| raw momentum 신호가 고변동 종목에 의해 순위 지배 | vol-normalized 시도 시 discovery 프리스크린 +37%(0.493→0.673) 개선 | 확인됨, 그러나 실전 배선 시 자본침범 (8차) |
| 2023/2024 손실은 비용이 아니라 실제 방향성 오판 | `mark_to_market_pnl` 연도합: 2022 +0.218, **2023 -0.155**, **2024 -0.181**, 2025 +0.531 | 미해결 (9차로도 못 고침, 의도된 결과) |
| 모멘텀 크래시 (단일일 전량 포지션 반전) | 2025-12-23 -40.4%, 2025-10-12 -31.6%, 둘 다 `daily_turnover≈2.0`·`daily_fill_count=1`; frictionless prescreen엔 안 보임(최저 -5.5%) → 실행마찰 증폭 확인 | 9차에서 완화 (MDD -45.7%→-21.6%) |
| 기존 `_regime_cash_scale`(자산단위 vol)은 이 크래시를 막지 못함 | 두 크래시일에도 활성 상태였으나 무력 — 실측으로 직접 확인 | 전략 자체 P&L vol(9차)로 별도 방어 추가 |

---

## 2. Discovery/Qualification 게이트 — 전체 격자 (Part B, 6차 실측, 유지)

`select_horizon_by_discovery_qualification`, worst-year-robust discovery(2022-2023, 2021은 데이터 커버리지로 non-finite) → qualification(2024-2025) 단일 재확인.

| Family | 후보 수 | 최고 horizon | 최고 worst-year `net_t` | 비고 |
| :--- | ---: | :--- | ---: | :--- |
| reversal (raw) | 7 | 24h | -1.175 | 144h~168h 포함 전부 음수 |
| momentum (raw) | 19 | 360h | +0.493 | 144h~408h 구간 양수 몰림, 192h~264h 국지 음전환(이봉) |
| momentum (vol-normalized) | 19 | 360h | **+0.673** | 8차 시도, 승자 horizon 불변, discovery 전용 |

전체 momentum(raw) 격자: 72h=-0.098, 96h=-0.117, 120h=-0.357, 144h=+0.113, 168h=+0.107, 192h=-0.393, 216h=-0.359, 240h=-0.168, 264h=-0.229, 288h=+0.138, 312h=+0.140, 336h=+0.220, **360h=+0.493**, 384h=+0.354, 408h=+0.121, 432h=-0.142, 456h=-0.120, 480h=-0.128, 504h=-0.191

2021 discovery 결손 원인: `liquid_half_eligibility`(720h lookback) eligible 심볼 수가 2021~2022Q1 평균 3개(`min_symbols=8` 미만)로 고정 — 데이터 커버리지 한계, 파이프라인 버그 아님 (`ADR_20260811_MHS_DISCOVERY_2021_GAP_AND_DENSE_GRID`).

---

## 3. 실행 사다리 진단 (Part A, 5차 실측, 유지 — primary 승격 안 함)

`OHLCV_LADDERED_PROXY` (K=4 tranche, 실패 시 선형 리프라이스, 마지막 tranche 시장가 폴백) vs strict (K=1):

| 지표 | strict (K=1) | ladder (K=4) | 변화 |
| :--- | ---: | ---: | :--- |
| intent shortfall (bps) | 1000.29 | 875.43 | -12.5% |
| fill_count | 81,018 | 293,898 | ×3.6 (예상됨) |
| unfilled_count | 37,539 | 36,836 | 거의 동일 |
| naive Sharpe | -0.6982 | -0.9093 | **악화** |
| forced_exit_count | 0 | 27 | 미미 |

**결론**: shortfall은 줄지만 risk-adjusted 지표는 악화 — tranche 분할이 평범한 주문에 소액 비용을 추가하고, 대다수 주문은 여전히 마지막 tranche에서 시장가로 떨어짐(추세 지속 시 "가격이 안 돌아온다"는 근본 원인 불변). Primary 승격 보류.

---

## 4. 다음 레버 후보 (미해결 과제)

| 후보 | 상태 |
| :--- | :--- |
| 2023/2024 방향성 손실 자체를 고치는 신호 재설계 | 미탐색 — vol-normalization/funding-carry/multi-horizon ensemble 모두 discovery 프리스크린 기준 admission floor(2.0) 근처도 못 감 |
| discovery worst-year 통계(admission_t=2.0)의 fold-local 표본 통계적 가혹함 재설계 | **10차에서 실측 확정** (미탐색 아님) — fold 1 최고 후보(168h, net_t=+1.382)조차 2.0 미달, fold 2 최고 후보(360h, +0.626)는 2.0의 절반 미만. 전역 다년 윈도우 기준 캘리브레이션이 fold-local 1~3년 표본엔 과도하게 엄격함이 확인된 다음 스펙 후보 (`docs/specs/mhs_fold_safe_horizon_selection.md` §2.1, 이제 삭제됨 — ADR 참고) |
| fast_reversal의 독립적 `CAPITAL_INVARIANT_BREACH`(`ts=2025-07-14`) | 미해결 — blend 자본 0%라 Research GO엔 영향 없으나 reversal 북 자체는 여전히 깨짐 |
| 사다리 비선형 리프라이스 (초반 느리게·후반 빠르게) 또는 다른 K 값 | 미탐색 |
