# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-13 (23차, `mhs_carry_and_fast_fair_evaluation` P0 실측 완료 — `ADR_20260813_MHS_CARRY_AND_FAST_FAIR_EVALUATION`. 22차 이전 이력은 git 이력으로 복구 가능)
- **Domain**: Research / MHS (Multi-Horizon Market State)

---

## 22차 (직전 기록) — breadth 포화 확정 + 펀딩비 캐리 leak-free 탈락

- **Run**: `start=2021-01-01 end=2025-12-31 execution_timeframe=1m execution_universe_size=30 eligible_symbols=446 run_elapsed_seconds=728.98`
- **CLI**: `--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full`
- **`effective_breadth` 실측**(신규 계측): `slow_momentum` 19-호라이즌 → $n_{\text{eff}}$=**1.41**(7.4%), `fast_reversal` 7-호라이즌 → $n_{\text{eff}}$=**1.57**(22.4%) — 19개 호라이즌을 동시 운용해도 유효 독립 베팅은 1.4~1.6개뿐임을 확정.
- **펀딩비 캐리 전 구간 예비측정**(fold 미분리): 168h lookback net_t=**+4.02**, net_ann=**+27.4%/년**, momentum과의 일간수익 상관 **+0.22**.
- **펀딩비 캐리 fold-train-only 재검증(3-fold, sign=±1)**: 3개 fold 전부 `funding_carry_lookback_hours=null` → `frozen_default` 폴백, admission 실패. 자본 배분 근거 없음.
- **회귀 불변식**: `slow_momentum.primary_autocorr_sharpe=0.525673922813482`, `blend.primary_autocorr_sharpe=0.5196163403815739`, `realized_execution_roster_size=41.93` — 21차와 바이트 동일.
- **Research-GO**: `eligible=False`, `reason_codes=[PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY]`, `folds_passed=2/3`, `deflated_sharpe_ratio=0.5321328197543407`(`trials_attempted=20`).
- **미해결로 남긴 질문**: admission floor(|t|≥2.0)가 fold-local(1년 남짓) 표본엔 과엄격해 momentum(168h)조차 fold-train discovery에서 매번 `frozen_default`로 폴백 — "펀딩비 캐리에 edge가 없다"가 아니라 "이 게이트로는 아직 못 살렸다"가 정확한 결론.

---

## 23차 (이번 기록) — 게이트 자체 재심사 + fast_reversal에 momentum의 구제책 적용

- **Run**: `start=2021-01-01 end=2025-12-31 execution_timeframe=5m execution_universe_size=30 eligible_symbols=446 run_elapsed_seconds=871.12 realized_execution_roster_size=41.93`
- **CLI**: `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full`
- **Source**: [`report.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json)
- **목적**: 22차가 momentum에게 준 유일한 구제책(단일 호라이즌 argmax → 호라이즌 앙상블)을 `fast_reversal`에도 처음 적용하고, `funding_carry`를 momentum/reversal과 동일한 top-level discovery 경로(2021-2023)에 배선해 3개 후보를 같은 계측기·같은 창에서 나란히 비교.

### 3.1 top-level discovery/qualification 게이트 (2021-2023, admission_t=2.0 불변)

| 후보 | admitted | selected_horizon |
| :--- | :--- | :--- |
| `momentum` | **False** | null |
| `reversal` | **False** | null |
| `funding_carry_long` | **False** | null |
| `funding_carry_short` | **False** | null |

- momentum(168h) discovery-window 연도별 net_t: 2021 **+2.09**(admission floor 겨우 통과) / 2022 **-0.03** / 2023 **-0.11** — 22차 §1.1과 동일 패턴 재확인.

### 3.2 `yearly_net_t_diagnostic` — 5년 전체(2021-2025) 보고용 (admission 미입력)

| 연도 | slow_momentum | fast_reversal(ensemble 적용) | funding_carry |
| :--- | ---: | ---: | ---: |
| 2021 | -0.145 | -1.329 | -4.249 |
| 2022 | +0.169 | +0.171 | -1.813 |
| 2023 | -0.568 | -0.797 | -2.176 |
| 2024 | +0.249 | -0.649 | -2.163 |
| 2025 | **+1.690** | +0.116 | -1.016 |

- `fast_reversal`은 momentum과 동일한 호라이즌 앙상블 구제책을 적용해도 5년 중 admission_t=2.0을 넘는 해가 **단 한 해도 없음** — 22차 §1.2/23차 스펙 §2.3의 "구제할 좋은 해가 애초에 없다"는 예측이 실측으로 확정됨.
- `funding_carry`는 5년 전 구간 **모두 강한 음수**(-4.25~-1.02) — discovery window 3년만 봤던 22차보다 근거가 강해짐: fold-local 표본 부족(게이트 검정력 문제)이 아니라 **실제로 edge가 없음**이 5년 전체 관측으로 확인됨.

### 3.3 `year_restricted_correlation` — momentum 최악의 해(2023)에서의 분산 효과

- `funding_carry_worst_year_corr` = **-0.2657396963823415** (momentum이 가장 약했던 2023년에 한정한 funding_carry-momentum 일간수익 상관).
- 표준 상관(22차, +0.13~+0.23, 전 구간)보다 최악의 해 상관이 오히려 더 음수 — 분산 효과의 방향 자체는 있음. 다만 funding_carry가 매년 손실이므로 이 분산가치를 자본으로 실현할 근거는 없음(§3.2와 결합 판단).

### 3.4 회귀 불변식 확인

- `slow_momentum.primary_autocorr_sharpe=0.525673922813482`, `blend.primary_autocorr_sharpe=0.5196163403815739` — 22차와 바이트 동일.
- `fast_reversal.primary_autocorr_sharpe=-0.8401773720292106` — `fast_book_mode=horizon_ensemble` opt-in 적용 시 값(22차의 `single_horizon` 집행책 -1.669와 다름, 예상된 변화; 기본값은 여전히 `single_horizon`이라 프로덕션 경로는 무변경).
- `research_go.eligible=False`, `reason_codes=[PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY]`, `folds_passed=2/3`, `deflated_sharpe_ratio=0.5321328197543407`(`trials_attempted=20`) — 무변화.

### 3.5 23차 요약

| 항목 | 상태 |
| :--- | :--- |
| `yearly_net_t_diagnostic`(5년 전체, 순수 보고용) | ✅ 완료, 3개 후보 전부 실측 |
| `year_restricted_correlation`(최악의 해 상관) | ✅ 완료, -0.266 |
| `fast_book_mode` 플래그 + `_horizon_ensemble_execution_weights` 개명·배선 | ✅ 완료, opt-in 기본값 `single_horizon` 무회귀 |
| funding_carry top-level discovery 배선 | ✅ 완료, momentum/reversal과 동일 창에서 비교 가능 |
| `fast_reversal` 구제 시도 | ❌ 5년 전체에서도 admission 가능한 해 없음 — "쓸모없음"이 공정하게 검증됨 |
| `funding_carry` 최종 판정 | ❌ 5년 전체 일관된 손실 — 게이트 검정력 문제가 아니라 edge 부재로 결론 강화 |
| `admission_t=2.0` 및 자본 배분(`PHASE_1_BOOK_SPECS`/`PHASE_1_BOOK_BLEND_WEIGHTS`) | 무변경 (스펙 §4 금지사항 준수) |

## 4. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| momentum 3년 discovery window의 admission floor 검정력 재검토 (fold-local 표본 민감도, 22차부터 이어지는 미해결 질문) | 미착수, 사용자 판단 대기 |
| 신규 수익원 후보 탐색 — OI, 청산, 현물-선물 베이시스 | 미착수 |
| `MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`) 등록 여부 | 미착수, 성과 무관 정책 결정 필요 |
| `pnl_vol_target` 기본값 전환 여부 (사전등록 fold-train-only 기준, `mhs_execution_friction_and_exposure_layers.md` §6.1) | 미착수 |
| `fast_book_mode`/`funding_carry` 자본 배분 여부 최종 판정 (본 계약의 실측 결과가 선행조건) | 미착수, 사용자 승인 필요 |
