# MHS Horizon Diagnostic — Latest Result

> **FORMAT POLICY**: 이 문서는 AI 분석용 데이터 로그다. 서술식 문장·설명·배경 스토리를 추가하지 말 것. 신규 실측은 표/키:값/코드 인용으로만 기록한다. 해석이 필요하면 `interpretation` 컬럼 또는 태그(`root_cause=`, `verdict=`)로 한 줄 이내로만 압축. 산문 문단(2문장 이상 서술)은 항상 리라이트 대상.

## META

| key | value |
| :--- | :--- |
| latest_run_seq | 27 |
| latest_run_date | 2026-08-14 |
| latest_adr | ADR_20260814_MHS_DIRECTIONAL_TREND_SLEEVE |
| domain | Research / MHS (Multi-Horizon Market State) |
| history | 23차 이전은 git 이력으로 복구 |

## RUN LOG

| seq | date | scope | cli/flags | book_mode | comparable_to | notes |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 24 | 2026-08-13 | Full 5y production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full` | horizon_ensemble | §1,§2,§3,§6 baseline | `eligible_symbols=445 realized_execution_roster_size=41.934179584940985 run_elapsed_seconds=765.0 peak_rss=16.8GB`; ADR_20260813_MHS_EXECUTION_PERFORMANCE_OPTIMIZATION |
| 25 | 2026-08-13 | discovery-gate only | `--discovery-gate --discovery-gate-adjusted-net-t` | single_horizon (default, non-production) | §3.1 only | top-level book/fold/research_go NOT comparable (book_mode mismatch); `run_elapsed_seconds=490.8 status=COMPLETE` |
| 26 | 2026-08-13 | discovery-gate only | `--discovery-gate --discovery-gate-regime-scaled-net-t` | single_horizon (default, non-production) | §3.2 only | same caveat as 25; `run_elapsed_seconds=501.3 status=COMPLETE` |
| 27 | 2026-08-14 | trend sleeve production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --trend-sleeve --trend-sleeve-gross 0.3 --output-tier full` | horizon_ensemble | §7 (books/folds/research_go bit-identical to 24) | ADR_20260814_MHS_DIRECTIONAL_TREND_SLEEVE; `docs/specs/mhs_directional_trend_sleeve.md` |

> ⚠️ `eligible_symbols` drift: 24차 445 vs 23차 446(Parquet 재수집). 동일 데이터 A/B는 bit-identical(§0). 23차 대비 수치차는 코드 변경 아님, 데이터 drift.

---

## 0. 성능 최적화 A/B (동일 데이터, 2026-08-13)

| 지표 | 원본 코드 | 최적화 코드 (C1-C4) | Δ |
| :--- | ---: | ---: | ---: |
| Full 5y wall (기본 5m, gate off) | 684.9 s | 309.1 s | −55 % |
| Full 5y peak RSS (기본 5m, gate off) | 5.40 GB | 3.15 GB | −42 % |
| `slow_momentum.primary_autocorr_sharpe` | −0.549471229370105 | −0.549471229370105 | bit-identical |
| `blend.primary_autocorr_sharpe` | 0.452119924579761 | 0.452119924579761 | bit-identical |
| `realized_execution_roster_size` | 41.934179584940985 | 41.934179584940985 | bit-identical |
| `test_mhs_replay_resources` checksum | `b7a7ffba…` | `b7a7ffba…` | bit-identical |
| 6mo book worker | 31.7 s | 7.1 s | −78 % |
| 6mo fold worker | 61.5 s | 17.5 s | −72 % |

optimization_map: C1=mark 프레임 캐시, C2=window materialize+pass 재사용, C3=window 단위 parquet 로드(preload 폐기), C4=fold-safe discovery 병렬화.

peak_rss_16.8GB_source: discovery 후보 웨이트 행렬(~13GB, slow 19 + fast 7 + funding 2×8 호라이즌 full-panel). fold-safe/top-level gate 중복 빌드 — dedupe 여지(§9).

---

## 1. Top-level book 성과 (2021-2025, 5m 집행, horizon_ensemble, 24차)

| Book | Horizon | Autocorr Sharpe | Naive Sharpe | Net Ann | CAGR | MaxDD | Turnover(x/yr) | Stress Sharpe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slow_momentum | 168h | +0.536 | +0.472 | +9.95 % | +8.08 % | −22.7 % | 42.8 | +0.152 |
| blend | 168h | +0.536 | +0.484 | +9.50 % | +7.91 % | −20.2 % | 42.0 | +0.146 |
| fast_reversal | 48h | −0.840 | −0.789 | −7.26 % | −7.40 % | −32.2 % | 38.6 | −1.137 |

| key | value |
| :--- | :--- |
| blend.target_gross | 0.7507 |
| blend.cash_fraction | 0.2493 |
| deflated_sharpe_ratio | 0.5470 |
| termination_counts.MISSING_DATA | 62 |
| termination_counts.UNKNOWN_TERMINATION | 0 |
| fast_reversal.verdict | autocorr/naive/stress 전부 음수, capital_allocation=0 |

## 2. Anchored folds (strict primary, horizon `frozen_default` 168h/48h, 24차)

| Fold | Autocorr Sharpe | Naive Sharpe | Stress Sharpe | Net Ann | Failures |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | +0.805 | +0.812 | +0.211 | +9.66 % | — |
| 1 | −0.267 | −0.308 | −0.820 | −4.19 % | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | +1.505 | +1.134 | +0.931 | +48.2 % | — |

| key | value |
| :--- | :--- |
| research_go.eligible | False |
| research_go.reason_codes | PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY |
| research_go.folds_passed | 2/3 |

## 3. Discovery gate & breadth (2021-2023, admission_t=2.0, 24차)

| 후보 | admitted | selected_horizon |
| :--- | :--- | :--- |
| momentum | False | null |
| reversal | False | null |
| funding_carry_long | False | null |
| funding_carry_short | False | null |

| key | value |
| :--- | :--- |
| effective_breadth.slow_horizon | 1.4120/19 (7.4%) |
| effective_breadth.fast_horizon | 1.5720/7 (22.5%) |

### 3.1 Bartlett/HAC 자기상관 보정 진단 (25차, opt-in `--discovery-gate-adjusted-net-t`)

spec: `docs/specs/mhs_discovery_admission_autocorr_robustness.md` (raw admitted/selected_horizon 필드 bit-identical, 신규 진단 필드만 아래)

window_correction: discovery=2021-2023(3yr, worst-of-3), qualification=2024-2025(2yr). source=`trend_screen_catalog.py:DISCOVERY_END/QUALIFICATION_END`. `docs/architecture/mhs-explain.md §4.2`(discovery 2021-2022/qualification 2023) is outdated — code is source of truth.

| 후보 | best raw worst-year net_t (worst-of-3) | best adjusted(HAC) worst-year net_t | admission_t |
| :--- | ---: | ---: | ---: |
| momentum (168h) | −0.0287 (2022) | −0.0294 (2022) | 2.0 |
| reversal (96h) | +0.252 (2021) | +0.270 (2021) | 2.0 |
| funding_carry_long (504h) | −3.322 (2021) | −3.151 (2021) | 2.0 |
| funding_carry_short (72h) | +4.213 (2022) | +4.099 (2022) | 2.0 |

verdict=HAC_HYPOTHESIS_REJECTED (raw vs adjusted delta ±5-10%, no admission-direction shift)

| 연도 | momentum 19-horizon raw net_t range |
| :--- | :--- |
| 2021 | +1.1 ~ +2.6 (all positive) |
| 2022 | −0.03 ~ −2.0 (all negative, no exception) |
| 2023 | −0.46 ~ +0.63 (mixed) |

root_cause = 2022 단일 연도가 19개 후보 전원을 admission_t 미만으로 끌어내림(표본 부족 아님, 구조적 레짐 패턴, 2022 crypto crash와 시기 일치).

| gate scoring vs production defense | value |
| :--- | :--- |
| discovery.py candidate weights | regime 방어 없음(raw signal) |
| production defaults | `_regime_cash_scale`(`evaluation.py:3481`), `pnl_vol_target_scale`(`pnl_vol_target=True`, `evaluation.py:2078`) 적용 |
| `crash_regime_tilt_weights` | default `alpha=None`, 24차 미적용; `ADR_20260813_MHS_CAPITAL_FLOOR_AND_OVERLAY_VALIDATION`에서 stress Sharpe 악화 확인되어 파리티 대상 제외 |

`run_elapsed_seconds=490.8 status=COMPLETE`

### 3.2 Regime-scale 파리티 진단 (26차, opt-in `--discovery-gate-regime-scaled-net-t`)

spec: `docs/specs/mhs_discovery_admission_regime_scale_parity.md` (raw admitted/selected_horizon bit-identical). regime_scale approximation: `realized_vol(log_close,48).where(eligible).mean(axis=1)` (PIT execution_mask 미사용).

| 후보 | best raw worst-year net_t | best regime-scaled worst-year net_t | admission_t | admitted |
| :--- | ---: | ---: | ---: | :--- |
| momentum (168h) | −0.0287 | +0.1798 | 2.0 | False |
| reversal (96h) | +0.252 | −0.142 | 2.0 | False |
| funding_carry_long | −3.32~−4.67 | 소폭 악화 | 2.0 | False |
| funding_carry_short | +2.79~+4.21 | 소폭 개선 | 2.0 | False (부호 규약상 미달) |

| momentum horizon | raw | regime-scaled |
| :--- | ---: | ---: |
| 168h | −0.029 | +0.180 |
| 336h | −0.158 | +0.162 |
| 384h | −0.304 | +0.027 |

verdict = REGIME_SCALE_HYPOTHESIS_PARTIALLY_CONFIRMED (direction correct, magnitude insufficient vs admission_t=2.0). reversal/funding_carry: no improvement or worse.

root_cause_updated = regime 방어 파리티는 방향만 맞음, admission 반전엔 부족. 잔여 지배 요인 = discovery window worst-of-3(2021-2023) 설계, 특히 2022.

`run_elapsed_seconds=501.3 status=COMPLETE`

## 4. `yearly_net_t_diagnostic` — 5년 전체 (admission 미입력, report-only, 24차)

| 연도 | slow_momentum | fast_reversal | funding_carry |
| :--- | ---: | ---: | ---: |
| 2021 | −0.145 | −1.329 | −4.249 |
| 2022 | +0.169 | +0.171 | −1.813 |
| 2023 | −0.568 | −0.797 | −2.176 |
| 2024 | +0.249 | −0.649 | −2.163 |
| 2025 | +1.775 | −0.012 | −1.142 |

| key | value |
| :--- | :--- |
| funding_carry_worst_year_corr (2023) | −0.2657 |
| fast_reversal admission_t=2.0 상회 해 | 0/5 |
| funding_carry 5년 net_t | 전 해 음수 |

## 5. 통계 진단 (전 구간, 24차)

| 계측 | 값 |
| :--- | :--- |
| `xs_rank_ic.n_dates` | 43,704 |
| `xs_rank_ic.mean_ic` | −0.04086 |
| `xs_rank_ic.t` | −46.02 (fwd=48) |
| `date_clustered_regression.n` | 8,257,895 |
| `date_clustered_regression.n_dates` | 1,826 |
| `date_clustered_regression.beta` | −0.01779 |
| `date_clustered_regression.t` | −1.32 |
| `horizon_diagnostics.realized_vol_48h` | 0.09112 |
| `horizon_diagnostics.efficiency_ratio_48h` | 0.14636 |
| `bootstrap_ci` (net_1h) | [−6.02e-6, +2.80e-5] |
| deployment.CAGR | +7.91 % |
| deployment.MaxDD | −20.15 % |
| deployment.Calmar | 0.3926 |
| deployment.P(최종재산<초기) | 16.25 % |

verdict: xs_rank_ic 유의 음수(역행), date_clustered t=−1.32 유의하지 않음(48h 전방가격 예측력 없음).

## 6. 회귀 불변식 & 요약 (24차, drift 데이터)

| 항목 | 값 |
| :--- | :--- |
| slow_momentum autocorr | 0.5360654316354135 |
| blend autocorr | 0.5359443875092911 |
| fast_reversal autocorr | −0.840177372029221 |
| deflated_sharpe | 0.546963269158657 |
| 23차 대비 diff cause | `eligible_symbols 446→445` 데이터 drift, 코드 무관(§0 A/B로 증명) |

capital_conclusion: momentum(168h ensemble)만 유일 생존 후보(5yr CAGR +8%, fold2 +48%). fast_reversal/funding_carry 자본 근거 없음(5yr 전체). Research-GO 미달. `PHASE_1_BOOK_SPECS`/`PHASE_1_BOOK_BLEND_WEIGHTS` 무변경.

## 7. 가산 시계열 추세 슬리브 — 27차 프로덕션 실측 (2026-08-14)

trigger: 사용자 질의 — "1개의 해가 나빠도 futures 숏으로 커버 가능하지 않나, 전략이 coin/futures 환경을 제대로 활용하나".

spec: `docs/specs/mhs_directional_trend_sleeve.md`, ADR: `ADR_20260814_MHS_DIRECTIONAL_TREND_SLEEVE`, module: `src/mhs/trend_sleeve.py`.

| 사전 검증 항목 | 값 |
| :--- | :--- |
| `rank_weight_book` dollar-neutral 강제 여부 | True (롱/숏 정확히 50:50, 코드 확인) |
| momentum 구조적 방향성 노출 가능 여부 | False (unit-gross dollar-neutral, 시장 방향 베팅 불가) |
| root_cause_2022 | 숏 부재 아님 — 전략 구조상 방향성 노출 자체가 불가능 |
| sleeve_default(gross_budget=0.0) vs baseline | bit-identical |

CLI: `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --trend-sleeve --trend-sleeve-gross 0.3 --output-tier full`

### 7.1 슬리브 단독 성과 (gross_budget=0.3)

| 비용 tier | net Sharpe |
| :--- | ---: |
| optimistic | +0.096 |
| base | +0.085 |
| stress | +0.072 |

| 연도 | net_t |
| :--- | ---: |
| 2021 | −0.132 |
| 2022 | +0.193 |
| 2023 | +0.239 |
| 2024 | −0.125 |
| 2025 | +0.309 |

| key | value |
| :--- | :--- |
| slow_momentum_pnl_corr | 0.275 |
| 2022 momentum net_t (§4) | +0.169 |
| 2022 sleeve net_t | +0.193 |

### 7.2 결합 성과 (momentum + 슬리브, risk-budget 방식)

| 비용 tier | net Sharpe |
| :--- | ---: |
| optimistic | +0.274 |
| base | +0.235 |
| stress | +0.188 |

| key | value |
| :--- | :--- |
| worst_year_net_t | −0.173 |
| fold1 stress 실패(§2) 상쇄 여부 | 부분적(완전 반전 아님) |

### 7.3 회귀 확인

| 항목 | 24차 대비 |
| :--- | :--- |
| books (§1) | bit-identical |
| folds (§2) | bit-identical |
| research_go (§2) | bit-identical |

## 8. 전략 요약 (분류 데이터)

| 전략 | 방향성 | 자본 배분 | 5yr CAGR/Sharpe | 상태 |
| :--- | :--- | ---: | :--- | :--- |
| slow_momentum | market-neutral (횡단면 랭크, dollar-neutral) | 100% | CAGR +8.08%, autocorr +0.536 | capital_active |
| blend | market-neutral (=momentum, cash_fraction 25%) | 100%(=momentum) | CAGR +7.91%, MaxDD −20.2%(momentum −22.7%) | capital_active |
| fast_reversal | market-neutral | 0% | autocorr −0.840, 5yr 전 지표 음 | rejected (§1,§4) |
| funding_carry | market-neutral(펀딩비 캐리) | 0% | 5yr 전 해 net_t 음 | rejected (§4) |
| discovery_gate | N/A (검증 절차, 비거래) | N/A | momentum 3/3 candidates rejected 2021-2023 | root_cause=2022 crash dominance(§3.1-3.2) |
| trend_sleeve | directional (시계열, non-dollar-neutral) | 0%(진단 단계, gross_budget default=0.0) | net Sharpe +0.07~+0.10, corr_to_momentum=0.275 | diagnostic_only, 자본 미승인 |

## 9. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| 시계열 추세 슬리브 `gross_budget` 스윕(0.3 초과) 및 discovery-gate 결합 실측 (§7) | 신규, 미착수 |
| momentum discovery window worst-of-3(2021-2023) 설계 재검토 | 원인 규명 2건 소진(§3.1-§3.2); window 구조 변경안은 별도 스펙+사용자 승인 필요; §7 슬리브가 우선 경로 |
| `_pnl_vol_target_scale`(Two-Pass) discovery gate 파리티 진단 추가 | 신규, 미착수 |
| discovery 후보 웨이트 중복 빌드 dedupe (fold-safe ↔ top-level gate) | 신규, 미착수 |
| `MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`) 등록 여부 | 미착수, 정책 결정 필요 |
| `pnl_vol_target` 기본값 전환 여부 (`mhs_execution_friction_and_exposure_layers.md` §6.1) | 미착수 |
| `fast_book_mode`/`funding_carry` 자본 배분 최종 판정 | 미착수, 사용자 승인 필요 |
| trend sleeve 자본 배분(`gross_budget`) 확정 | 미착수, §7 실측 확장 후 사용자 승인 필요 |
