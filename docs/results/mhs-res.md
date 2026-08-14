# MHS Horizon Diagnostic — Latest Result

> **FORMAT POLICY**: 이 문서는 AI 분석용 데이터 로그다. 서술식 문장·설명·배경 스토리를 추가하지 말 것. 신규 실측은 표/키:값/코드 인용으로만 기록한다. 해석이 필요하면 `interpretation` 컬럼 또는 태그(`root_cause=`, `verdict=`)로 한 줄 이내로만 압축. 산문 문단(2문장 이상 서술)은 항상 리라이트 대상.

## META

| key | value |
| :--- | :--- |
| latest_run_seq | 29 |
| latest_run_date | 2026-08-14 |
| latest_adr | ADR_20260814_MHS_COMMITTEE_DESIGN_AND_WEALTH_OBJECTIVE |
| domain | Research / MHS (Multi-Horizon Market State) |
| history | 23차 이전은 git 이력으로 복구 |

## RUN LOG

| seq | date | scope | cli/flags | book_mode | comparable_to | notes |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 24 | 2026-08-13 | Full 5y production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full` | horizon_ensemble | §1,§2,§3,§6 baseline | `eligible_symbols=445 realized_execution_roster_size=41.934179584940985 run_elapsed_seconds=765.0 peak_rss=16.8GB`; ADR_20260813_MHS_EXECUTION_PERFORMANCE_OPTIMIZATION |
| 25 | 2026-08-13 | discovery-gate only | `--discovery-gate --discovery-gate-adjusted-net-t` | single_horizon (default, non-production) | §3.1 only | top-level book/fold/research_go NOT comparable (book_mode mismatch); `run_elapsed_seconds=490.8 status=COMPLETE` |
| 26 | 2026-08-13 | discovery-gate only | `--discovery-gate --discovery-gate-regime-scaled-net-t` | single_horizon (default, non-production) | §3.2 only | same caveat as 25; `run_elapsed_seconds=501.3 status=COMPLETE` |
| 27 | 2026-08-14 | trend sleeve production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --trend-sleeve --trend-sleeve-gross 0.3 --output-tier full` | horizon_ensemble | §7 (books/folds/research_go bit-identical to 24) | ADR_20260814_MHS_DIRECTIONAL_TREND_SLEEVE; `docs/specs/mhs_directional_trend_sleeve.md` |
| 28 | 2026-08-14 | multi-feature diagnostic production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --multi-feature-book --output-tier full` | horizon_ensemble | §8 (books/folds/research_go bit-identical to 24) | ADR_20260814_MHS_MULTI_FEATURE_ALPHA_ARCHITECTURE; `docs/specs/mhs_multi_feature_alpha_architecture.md`; sandbox: `scratch/test_breadth_and_feature_axis.py`, `test_feature_books_net_of_cost.py`, `test_feature_oos_persistence.py` |
| 29 | 2026-08-14 | committee (k=6) production | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --committee-book --output-tier full` | horizon_ensemble | §9 (books/folds/research_go bit-identical to 24) | ADR_20260814_MHS_COMMITTEE_DESIGN_AND_WEALTH_OBJECTIVE; `docs/specs/mhs_committee_design_and_wealth_objective.md`; sandbox: `scratch/build_committee_panel.py`, `analyze_committee_design_v2.py`, `analyze_committee_final.py` |

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

peak_rss_16.8GB_source: discovery 후보 웨이트 행렬(~13GB, slow 19 + fast 7 + funding 2×8 호라이즌 full-panel). fold-safe/top-level gate 중복 빌드 — dedupe 여지(§11).

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

## 8. 다중 피처 알파 아키텍처 — 28차 실측 (2026-08-14)

trigger: 사용자 요청 — 전략 구성/평가 로직 구조 재검토, ML 도입 고려.

ADR: `ADR_20260814_MHS_MULTI_FEATURE_ALPHA_ARCHITECTURE`, module: `src/mhs/features.py`, `src/mhs/stability.py`.

### 8.1 실효 breadth (participation ratio, `effective_breadth`)

| 축 | 후보 수 | n_eff | mean_corr |
| :--- | ---: | ---: | ---: |
| 호라이즌(프로덕션 slow 그리드) | 19 | 1.413 | 0.824 |
| 피처(IC 시계열) | 10 | 5.031 | 0.154 |
| 피처(북 net PnL) | 10 | 4.015 | 0.115 |

### 8.2 피처 vs IC 괴리 (net, 전 구간 2021-2025, top30 로스터, fwd=168h)

| 피처 | base net Sharpe | IC (t) |
| :--- | ---: | ---: |
| mom_168h(프로덕션) | +0.020 | −0.0175 (−15.0) |
| mom_336h | +0.688 | −0.0137 (−11.8) |
| taker_imb_168h | +0.990 | +0.0189 (+18.1) |
| lowvol_168h | −0.038 | +0.1475 (+104.8) |
| hl_range_168h | −0.050 | +0.1530 (+105.9) |

verdict = IC_NOT_ALPHA_PROXY (IC 최고 피처 2종이 실제 북은 음수; IC 작은 피처가 북 성과 최고). 168h 오버랩 독립표본 ~261개(43,824/168), t 과대 계수 ~√168≈13x.

### 8.3 레짐 지속성 (net, base tier)

| 피처 | 2021-2023 | 2024-2025 | sign_consistent |
| :--- | ---: | ---: | :--- |
| mom_168h(프로덕션) | +0.615 | **−0.690** | False |
| mom_336h | +0.651 | +0.742 | True |

### 8.4 데이터 결함

| key | value |
| :--- | :--- |
| no_trades coverage(top30, 연도별) | 2021:0.84 → 2022:0.34 → 2023-2025:0.00 |
| avg_trade_size(오염) full Sharpe | +0.311(사실상 2021-2022만) |
| 프로덕션 반영 결과 | coverage gate가 avg_trade_size 자동 제외(§8.5 excluded) |

### 8.5 프로덕션 실측 (`multi_feature_diagnostic`, 28차)

| 피처 | window_0(2021-23) | window_1(2024-25) | sign_consistent | decay |
| :--- | ---: | ---: | :--- | ---: |
| mom_168h | +0.597 | −0.258 | False | −0.855 |
| mom_336h | +0.582 | +1.461 | True | +0.879 |
| taker_imb_168h | +2.001 | +1.105 | True | −0.896 |
| rev_24h | +0.441 | +0.052 | True | −0.389 |
| turnover_chg | +0.077 | +0.657 | True | +0.581 |
| taker_imb_24h | +0.791 | −0.439 | False | −1.230 |
| amihud | −0.269 | +0.891 | False | +1.160 |
| lowvol_168h | −0.458 | −0.168 | True(둘다 음) | +0.291 |
| hl_range_168h | −0.578 | +0.024 | False | +0.602 |

excluded: `avg_trade_size`(failing_year=2021, coverage gate 자동 차단)

| key | value |
| :--- | :--- |
| feature_book_effective_breadth.n_eff | 3.493/9 |
| feature_book_effective_breadth.mean_corr | 0.139 |
| combined.net_sharpe_per_tier(수정 전, 버그) | optimistic +1.016, base +0.754, stress +0.432 |
| combined.book_mean_gross(수정 전) | 175.58(비정상, 리스크패리티 정규화 누락 버그) |
| bugfix | `evaluation.py::_multi_feature_diagnostic`에 `n/Σ(1/sdᵢ)` 정규화 적용, lean_check PASS 확인 |
| books/folds/research_go(§1/§2) | bit-identical to 24차 |

## 9. 6인 위원회(committee) 및 자산증식 목적함수 — 29차 실측 (2026-08-14)

trigger: 사용자 요청 — 위원회 규모/구성/ML 조율/자산증식 극대화 검토.

ADR: `ADR_20260814_MHS_COMMITTEE_DESIGN_AND_WEALTH_OBJECTIVE`, module: `src/mhs/committee.py`.

protocol: purged walk-forward(train-only 336h purge, 6개월 블록), 비용 2-tier 대수분해(`decompose_cost`)로 net→gross/turnover_cost 복원, 롱온리 강제, train-window 15% 변동성 타게팅.

### 9.1 후보 31종 → 경제 패밀리별 최고 (base tier full-period Sharpe)

| family | best | sharpe |
| :--- | :--- | ---: |
| flow | flow_imb_168h | +1.251 |
| trend | xs_mom_336h | +0.765 |
| moments | mom3_skew_168h | +0.627 |
| carry | carry_funding_chg | +0.506(데이터 결함, §9.4) |
| reversal | xs_rev_6h | +0.280 |
| volatility | vol_idio_168h | +0.104 |
| liquidity | liq_turnover_surge | +0.084 |
| beta | beta_low | +0.017 |
| micro | micro_hl_range_168h | −0.076 |

### 9.2 조합기 비교 (sandbox OOS walk-forward, 30종 전체, 비용부호보정 후)

| 조합기 | Sharpe | opti/base/stress |
| :--- | ---: | :--- |
| equal_risk | +0.402 | +0.751/+0.402/−0.026 |
| sharpe_weighted(long-short) | −0.209 | — |
| shrunk_MV λ=0.8(long-only) | +0.650 | +0.872/+0.650/+0.544 |
| shrunk_MV λ=0.8(**long-short**) | **−0.568** | +0.232/−0.568/**−1.813** |
| top-12(train-Sharpe 선택) | +0.776 | +0.892/+0.776/+0.555 |
| top-15 | +0.867 | — |
| regime-conditional | +0.404 | — |
| **큐레이션 k=6 + equal_risk** | **+1.379** | **+1.500/+1.379/+1.230** |

verdict = LEARNED_COMBINER_NOT_JUSTIFIED (모든 학습기반 조합기가 경제적 큐레이션에 전패; 무제약 조합기는 전 tier 음수)

### 9.3 위원회 규모/구성 sweep (sandbox, 경제 패밀리 순차 증설, OOS)

| k | 구성 | Sharpe | CAGR | MDD |
| ---: | :--- | ---: | ---: | ---: |
| 1 | flow | +0.951 | +14.47% | −13.3% |
| 2 | flow+trend | +1.141 | +17.24% | −11.7% |
| 4 | flow×2+trend×2 | +1.029 | +15.20% | −11.4% |
| **6** | **+idio_mom+skew** | **+1.379** | **+22.02%** | −12.5% |
| 7 | +rev | +1.444 | +23.17% | −15.7% |
| 8 | +carry | +1.273 | +21.78% | −20.1%(악화 시작) |

`MHS_COMMITTEE_MEMBERS`(k=6): flow_imb_720h, flow_imb_168h, xs_mom_336h, xs_mom_720h, xs_idio_mom_336h, mom3_skew_168h. n_eff=2.168/6, mean_corr=0.290.

### 9.4 데이터 결함 (신규)

| key | value |
| :--- | :--- |
| `bar_funding_panel` 2021 그리드 정렬 성공 심볼 | 45/452 |
| 나머지 처리 | `fillna(0.0)` — 랭크 중앙 배치, post-fillna coverage gate로 탐지 불가 |
| 대응 | `source_coverage_audit`(fillna 이전 원천 감사) 계약 추가, carry 패밀리 위원회에서 제외 |

### 9.5 자산증식 목적함수 (sandbox, k=6 큐레이션, 15% 변동성 타게팅)

| leverage | vol | CAGR | MDD | logret |
| ---: | ---: | ---: | ---: | ---: |
| 1.0x | 15% | +22.02% | −12.5% | +0.598 |
| 1.5x | 22% | +33.61% | −18.3% | +0.870 |
| 2.0x | 30% | +45.46% | −23.9% | +1.125 |

| key | value |
| :--- | :--- |
| full_kelly | 9.02x |
| half_kelly | 4.51x |
| kelly_verdict | 채택 불가(OOS 블록 6개 표본오차 + 두꺼운 꼬리, 상한 참고치일 뿐) |
| 권고 leverage | 1.0x 시작, 1.5x 상한 |

### 9.6 프로덕션 실측 (`committee_diagnostic`, 29차)

| tier | net_sharpe | cagr | mdd | logret |
| :--- | ---: | ---: | ---: | ---: |
| optimistic | +1.798 | +26.91% | −11.27% | +1.074 |
| base | +1.664 | +24.58% | −11.99% | +0.991 |
| stress | +1.499 | +21.78% | −12.88% | +0.888 |

| key | value |
| :--- | :--- |
| admitted | 6/6(전원), excluded=[] |
| source_coverage(전원) | 5년 전 구간 1.0(funding 미사용 위원회라 §9.4 결함 비해당) |
| walk_forward.block_edges 시작 | 2021-01-01 |
| walk_forward.bars(OOS 집계) | 39,480 / 43,824(90%) |
| issue_flag | WALK_FORWARD_START_OPTIMISTIC — 블록 생성이 `_committee_block_edges(start=2021-01-01, end)`을 사용해 sandbox 설계(OOS start=2023-01-01, min_train_bars=2000)보다 이른 시점부터 테스트에 포함됨. `purged_walk_forward` 자체(purge/train-only scaling)는 계약대로 정상 — 호출부의 블록 시작점 선택만 낙관적. 보수적 신뢰 상한은 §9.2의 sandbox base=+1.379(OOS start=2023) |
| books/folds/research_go(§1/§2) | bit-identical to 24차 |

## 10. 전략 요약 (분류 데이터)

| 전략 | 방향성 | 자본 배분 | 5yr CAGR/Sharpe | 상태 |
| :--- | :--- | ---: | :--- | :--- |
| slow_momentum | market-neutral (횡단면 랭크, dollar-neutral) | 100% | CAGR +8.08%, autocorr +0.536 | capital_active |
| blend | market-neutral (=momentum, cash_fraction 25%) | 100%(=momentum) | CAGR +7.91%, MaxDD −20.2%(momentum −22.7%) | capital_active |
| fast_reversal | market-neutral | 0% | autocorr −0.840, 5yr 전 지표 음 | rejected (§1,§4) |
| funding_carry | market-neutral(펀딩비 캐리) | 0% | 5yr 전 해 net_t 음 | rejected (§4) |
| discovery_gate | N/A (검증 절차, 비거래) | N/A | momentum 3/3 candidates rejected 2021-2023 | root_cause=2022 crash dominance(§3.1-3.2) |
| trend_sleeve | directional (시계열, non-dollar-neutral) | 0%(진단 단계, gross_budget default=0.0) | net Sharpe +0.07~+0.10, corr_to_momentum=0.275 | diagnostic_only, 자본 미승인 |

## 11. 다음 스텝 후보

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
