# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-10 (3차 근본 개선 재실행 — book admission + regime vol_mean 수정)
- **Registered ADRs**:
  - `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION` (1차: 실행 roster 마스킹 후 dollar-neutral/unit-gross 재정규화)
  - `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT` (2차: roster 진입/이탈 히스테리시스 + causal inverse-vol tilt)
- **이번 수정 근거**: `docs/specs/mhs_fundamental_redesign.md` (§2.A blend 가중치 개정 + §2.C `_regime_cash_scale` vol_mean 마스킹)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact tier)
- **Execution Status**: `COMPLETE` — 3북 중 `fast_reversal`과 anchored fold 2만 여전히 `CAPITAL_INVARIANT_BREACH`로 중도 실패, `blend`는 이번에 완주 전환
- **Run Metadata**: 2021-01-01~2025-12-31, `execution_universe_size=30`, `execution_timeframe=5m`, `eligible_symbols=446`, `run_elapsed_seconds≈254s`

---

## 0. 이 리포트를 읽는 법 — 왜 숫자가 이전과 다른가

이번 재실행은 신호 유효성과 사이징 결함 두 갈래를 함께 고쳤다:

1. **북 admission (§2.A)**: `fast_reversal`은 446종목 전체·비용 이전 prescreen에서
   t≈-0.15로 잡음과 통계적으로 구분되지 않아(부호조차 불안정) 자본 배분에서
   제외됐다. `PHASE_1_BOOK_BLEND_WEIGHTS`가 `{fast_reversal: 0.0,
   slow_momentum: 1.0}`으로 개정돼, Research GO의 primary 증거인 `blend`가
   죽은 신호의 50%를 더 이상 안고 가지 않게 됐다.
2. **regime vol_mean 마스킹 (§2.C)**: `_regime_cash_scale()`의 입력 변동성이
   eligible 전체(수백 종목) 평균이 아니라 **실제 거래되는 30종목 execution
   roster 자체**의 평균으로 좁혀졌다. 기존에는 "엉뚱한 온도계로 실제 화상
   위험을 재던" 상태였다.

**결과 요약**: `blend`는 `fast_reversal` 자본잠식에서 **완주로 전환**됐다. 그러나
`blend`는 slow_momentum과 **수치적으로 동일하지 않다** — blend 북은 여전히
6h 결정 그리드(fast grid) 위에서 slow 신호를 ffill로 실행하므로(autocorr
Sharpe -2.13, turnover 7.53) 24h 그리드에서 도는 `slow_momentum` 단독
(autocorr -1.84, turnover 5.62)보다 열악하다. "blend = slow_momentum"은
가중치 구성상 맞지만, 실행 그리드 차이는 남는다.

또한 이번 재실행에서 prescreen t-stat가 소폭 이동했다(`fast_reversal`
-0.218→-0.150, `slow_momentum` +1.679→+1.634, eligible 445→446). 이는 코드
변경이 아니라 **자금조달(funding) 커버리지 데이터 드리프트**로 유니버스가 한
종목 늘어난 영향이며, admission 판정(dead/weak)에는 영향을 주지 않는다.

이 리포트가 보여주는 핵심은 여전히 **"성과가 좋아졌다"가 아니라 "숨겨져 있던
진짜 문제가 더 정직하게 드러나고 있다"**는 것이다. 두 차례 구조 수정 + 이번
admission/cash-scale 수정을 거쳐도, 살아남은 유일한 신호(slow_momentum)의
이론적 edge(t≈1.63)와 실행 결과(autocorr Sharpe -1.84, MDD -99.2%)의 격차는
극단적으로 남아 있다.

---

## 1. Full-Period Primary Replay — 세 북의 현재 상태

| Book | 완주 여부 | Autocorr Sharpe | Naive Sharpe | Stress Naive Sharpe | MDD | Annualized Turnover | Research GO Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **fast_reversal** | ❌ **`CAPITAL_INVARIANT_BREACH`** ("pre-trade equity must be positive and finite") | — | — | — | — | — | ≥ +0.60 |
| **slow_momentum** | ✅ 완주 | **-1.8373** | -0.6982 | **+0.0926** | **-99.2360%** | 5.6211 | ≥ +0.60 |
| **blend** | ✅ 완주 (이번에 전환) | **-2.1289** | -0.9882 | **+0.0858** | **-99.9806%** | 7.5340 | ≥ +0.60 |

- `slow_momentum`과 `blend` 둘 다 5년 전체를 파산 없이 완주했지만, MDD가 각각
  -99.24%/-99.98%로 사실상 초기 자본 전멸이다. Autocorr Sharpe도 크게
  음수이며, 이는 **신호가 비용·변동성 노출을 이겨내지 못한다**는 뜻이다.
- `blend`의 이번 완주 전환은 §2.A(dead book 배분 제외)의 직접 결과다.
  다만 blend는 slow 신호를 **6h 그리드**에서 실행하는 구조라(기존 아키텍처,
  이번 scope 밖) turnover가 slow 단독(24h)보다 높고 MDD도 더 깊다 —
  "blend = slow_momentum"은 구성상 정확하지만 수치상 동일하지 않다.
- `fast_reversal`은 여전히 리플레이 도중 자본이 0 이하로 떨어져 원장이
  fail-closed로 중단된다 — prescreen 단계부터 edge가 없는 신호(§3)가 실행
  단계에서도 자본잠식한다는 일관된 진단이다.
- `blend_cash_fraction=0.438`, `blend_target_gross=0.562` — §2.C로 roster
  변동성 기반의 cash scale이 적용돼 평균 노출이 100%에서 56%로 내려갔다.
- Intent Shortfall/Fill·Unfilled count는 `--output-tier full` 재실행 시에만
  `_full/report.json`에 기록된다.

---

## 2. Anchored Folds Quantitative Performance

| Metric | Fold 0 (2023) | Fold 1 (2024) | Fold 2 (2025) | Research GO Target |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Validity** | `True` (완주) | `True` (완주) | **`False`** — `CAPITAL_INVARIANT_BREACH` | All `True` |
| **Autocorrelation Sharpe** | **-3.6898** | **-4.3971** | — | ≥ +0.60 |
| **Naive Sharpe** | -1.0378 | -1.2403 | — | > 0.00 |
| **Stress Naive Sharpe** | -0.1668 | -0.1582 | — | > 0.00 |
| **Maximum Drawdown (MDD)** | **-52.5304%** | **-57.6267%** | — | Low MDD |
| **Failures** | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` | 동일 | `CAPITAL_INVARIANT_BREACH` | — |

Fold 0/1은 완주했지만 Autocorr Sharpe가 각각 -3.69/-4.40으로 여전히 깊은
음수다(2차 수정 대비 -4.63/-7.17에서 소폭 개선 — §2.C의 roster 기반
cash-scale이 고변동 국면 노출을 낮춘 영향으로 해석). Fold 2는 2025년 구간
리플레이 중 자본잠식으로 완주하지 못했다.

---

## 3. Books Prescreen & Tail Analysis (이론적 전체 유니버스 기준, 마스킹 이전)

Prescreen은 실제 roster 마스킹·실행비용을 적용하지 않고 eligible 유니버스
전체·flat 8bps 비용 가정으로 계산한 **이론적 상한**이다. 이 표의 t-stat는
2.64bps(optimistic tier) 기준이다. Research GO 판정에는 쓰이지 않지만, 신호
자체의 알파 존재 여부를 가늠하는 참고 지표다.

| Book | Prescreen Net Sharpe (2.64bps) | Prescreen t-stat | Phase Ensemble Sharpe | Tail Base Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| **fast_reversal** | **-0.0671** | -0.150 | -2.3458 | -0.5113 |
| **slow_momentum** | **+0.7303** | **+1.634** | -0.0544 (`degenerate=True`) | +0.3086 |
| **blend** | **+0.7951** | **+1.778** | -2.3458 | +0.2828 |

- `fast_reversal`은 이론적 전체 유니버스·비용 이전 단계에서도 edge가 없다
  (t≈-0.15, 부호 불안정). 이번 §2.A에서 blend 가중치를 0.0으로 내린 판정의
  근거다.
- `slow_momentum`은 이론상 약하지만 실재하는 edge(t≈+1.63, preregistered
  모멘텀 부호와 일치)를 보이는데, §1의 실제 리플레이(autocorr -1.84, MDD
  -99.2%)와의 격차가 여전히 극단적이다 — 실행/사이징 레이어가 이 이론적
  edge를 심각하게 파괴하고 있다.
- `slow_momentum`의 `phase.degenerate=True`(phase_spread 0.154 >
  |mean_phase_ann| 0.009)는 이 이론적 edge조차 특정 clock-offset에 의존적일
  수 있다는 경고로, 아직 해소되지 않았다.
- `blend`의 Phase Ensemble(-2.3458)은 blend가 여전히 `fast` spec으로
  phase 진단을 계산하는 기존 아키텍처 반영값이다.

Cross-sectional 진단(마스킹과 무관, 신호 자체의 통계적 유의성):
`xs_rank_ic`: mean IC = 0.0939, t = 68.4 (n=43,776일) — 신호와 미래 수익률 간
순위 상관은 통계적으로 매우 유의하다. 문제는 신호의 존재 여부가 아니라, 그
신호를 **실행 가능한 형태로 자본화**하는 과정(roster 선정·사이징·비용)에 있다.

---

## 4. Research GO Evaluation

- **Research GO Final Status**: **`eligible: false`**
- **Reason Codes**: `CAPITAL_INVARIANT_BREACH`, `INCOMPLETE_ANCHORED_FOLD`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` (evaluated_folds=3, folds_passed=0)
- 사유 코드 목록은 2차 수정과 동일하지만, `blend`가 완주로 전환되며
  `INCOMPLETE_ANCHORED_FOLD`의 원인이 fold 2 단독으로 좁혀졌다. Research GO는
  여전히 false — §2.B/D/E(roster N·상수 재측정)가 아직 실행되지 않았고,
  실행 결과 자체가 이론 edge를 파괴하는 문제가 남아 있기 때문이다.

---

## 5. 진단 요약과 다음 단계

### 5.1 지금까지 확인된 근본 원인 (수정 완료)

1. **실행 roster 마스킹 후 미재정규화** (1차 수정, `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION`):
   gross가 몰래 ~7%로 붕괴했던 버그. 수정 완료.
2. **roster 유출입의 강제 전량매매 + 변동성 무시 균등가중** (2차 수정,
   `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT`): 히스테리시스+inverse-vol
   tilt로 부분 완화. 수정 완료.
3. **죽은 신호의 50% 고정 배분** (3차, `docs/specs/mhs_fundamental_redesign.md`
   §2.A): prescreen 통계로 유효성 없는 `fast_reversal`의 blend 가중치를 0.0으로.
   `blend` 완주 전환으로 구조적 결함이 해소됨. 수정 완료.
4. **regime cash scale의 엉뚱한 변동성 입력** (3차, §2.C): vol_mean 분모를
   eligible 전체에서 실제 execution roster로 마스킹. 순수 버그 수정 완료.

### 5.2 남은 문제 (미해결, 후속 과제)

- 100% gross 대비 노출이 cash-scale로 56%까지 내려갔음에도 autocorr Sharpe가
  여전히 -1.8~-2.1인 것은, 사이징·비용 레이어가 이론 edge를 파괴하는 문제가
  아직 해소되지 않았음을 의미한다. 다음 순서로 재검토 필요:
  1. `execution_universe_size=30`은 미측정 기본값 — §2.B 절차(기존 prescreen을
     N 후보별 재계산)로 검증된 상수로 승격 필요.
  2. `MHS_REGIME_CASH_SCALE_FLOOR`(0.5)·`MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER`
     (2.0)·`passive_timeout_minutes`(30) 재측정 — §2.C 완료 후의 전제에서만
     의미 있음.
  3. base gross target 자체를 하향하는 사이징 정책 변경 — 별도 spec 대상.
- `fast_reversal`은 prescreen부터 edge가 없다(§3). 신호 재설계가 필요하며,
  코드/리포트는 유지해 재측정 시 재승격 가능한 상태로 남겨둔다.
- `slow_momentum`의 `phase.degenerate=True` 경고(clock-offset 의존성)가
  미해결 상태로 남아 있다.
