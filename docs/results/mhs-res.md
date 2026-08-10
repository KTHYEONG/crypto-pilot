# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-10 (4차 수정 재실행 — blend 실행 격자 결합 결함 수정)
- **Registered ADRs**:
  - `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION` (1차: 실행 roster 마스킹 후 dollar-neutral/unit-gross 재정규화)
  - `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT` (2차: roster 진입/이탈 히스테리시스 + causal inverse-vol tilt)
  - `ADR_20260810_MHS_BOOK_ADMISSION_VOL_MASK` (3차: book admission 동결 + regime vol_mean roster 마스킹)
  - `ADR_20260810_MHS_TOUCH_PROXY_FILL_MEASUREMENT` (strict-vs-touch 교차 판정 가설 반증, 30분 타임아웃발 비용 폭증 확인)
- **이번 수정 근거**: `docs/specs/mhs_root_cause_reassessment.md` §3.2 — blend의 실행
  결정 격자/BookSpec이 `PHASE_1_BOOK_BLEND_WEIGHTS`와 무관하게 `fast_reversal`
  (6h/1h)에 하드코딩돼 있던 구조적 결함 수정
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact tier)
- **Execution Status**: `COMPLETE`
- **Run Metadata**: 2021-01-01~2025-12-31, `execution_universe_size=30`, `execution_timeframe=5m`, `eligible_symbols=446`

---

## 0. 이번 수정과 결과

`PHASE_1_BOOK_BLEND_WEIGHTS`가 `{fast_reversal: 0.0, slow_momentum: 1.0}`로 개정된
뒤(3차 수정)에도, `blend`의 실행 결정 시각(decision cadence)과 `BookSpec`은 3개
호출부(`_run_books_concurrent`, `_build_fold_target_weights`, `phase_blend`)에서
여전히 `fast_reversal`의 6h/1h 격자에 하드코딩돼 있었다. 신규 순수 함수
`_active_blend_book_and_grid`가 `PHASE_1_BOOK_BLEND_WEIGHTS`를 유일한 진실 소스로
격자/spec을 파생하도록 배선했다.

**결과**: `blend`이 `slow_momentum`과 모든 지표에서 완전히 동일해짐 (`step_hours`,
`fills`, `autocorr Sharpe`, `MDD`, `turnover` 전부 일치) — 이전 재실행 대비 fills가
473,613 → **118,557**로 감소(4배 인위적 팽창 해소).

| Book | step_hours | fills | Autocorr Sharpe | MDD | Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **slow_momentum** | 24 | 118,557 | -1.8373 | -99.2360% | 5.6211 |
| **blend** (수정 후) | **24** (이전: 6) | **118,557** (이전: 473,613) | **-1.8373** (이전: -2.1289) | **-99.2360%** (이전: -99.9806%) | **5.6211** (이전: 7.5340) |

**Research GO는 여전히 FALSE** (`folds_passed=0/3`, reason codes:
`CAPITAL_INVARIANT_BREACH`, `INCOMPLETE_ANCHORED_FOLD`,
`PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`,
`UNSPECIFIED_POLICY`) — 격자 잡음 제거는 blend를 slow_momentum의 순수한 결과로
정리했을 뿐, 근본적인 이론-실행 격차(prescreen t≈+1.63 vs 실제 autocorr Sharpe
-1.84)는 해소하지 못했다.

## 1. 남은 근본 원인 (다음 우선순위, `mhs_root_cause_reassessment.md` §3.3/§4)

- **strict proxy 30분 타임아웃발 비용 폭증** (이미 측정됨,
  `ADR_20260810_MHS_TOUCH_PROXY_FILL_MEASUREMENT`): primary intent shortfall
  428~563bps vs 가정 비용 tier 2.64~6.07bps (~100배). strict-vs-touch 교차
  판정 가설은 반증됐고, 실제 원인은 추세 확인 구간에서 30분 내 passive 지정가로
  가격이 되돌아오지 않아 taker fallback이 발생하는 것. 이번 격자 수정으로
  잡음이 제거됐으니, 다음 재측정에서 순수한 타임아웃 비용 크기를 정확히
  볼 수 있다 — 별도 spec 필요(timeout 스윕 또는 부분 체결 아키텍처).
- **fast(48h)/slow(168h) horizon 선택 근거 문서 부재**: 코드가 인용하는
  "spec §2.1/§2.2/§2.4"는 git 히스토리 전체에 존재한 적 없음 — 판단 보류,
  별도 사전등록 discovery spec 필요.
- **execution_universe_size=30** 등 기존에 동결된 미측정 상수들
  (`docs/specs/mhs_fundamental_redesign.md` §2.B/D/E)은 이번 수정과 무관하게
  그대로 후속 과제로 남아 있다.
