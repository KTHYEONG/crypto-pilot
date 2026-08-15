# MHS Horizon Diagnostic — Latest Result

> **FORMAT POLICY**: 이 문서는 AI 분석용 데이터 로그다. 서술식 문장·설명·배경 스토리를 추가하지 말 것. 신규 실측은 표/키:값/코드 인용으로만 기록한다. 해석이 필요하면 `interpretation` 컬럼 또는 태그(`root_cause=`, `verdict=`)로 한 줄 이내로만 압축. 산문 문단(2문장 이상 서술)은 항상 리라이트 대상.

## META

| key | value |
| :--- | :--- |
| latest_run_seq | 30 |
| latest_run_date | 2026-08-15 |
| latest_adr | ADR_20260815_MHS_REFACTOR_LEGACY_ISOLATION_AND_OOM_BARRIER |
| domain | Research / MHS (Multi-Horizon Market State) |
| history | 29차 이전은 git 이력으로 복구 |

## RUN LOG

| seq | date | scope | cli/flags | book_mode | notes |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 30 | 2026-08-15 | Full 5y production, post-refactor | `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full` | horizon_ensemble | `docs/specs/mhs_refactor.md` 적용 후 첫 실측; report path=`docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json` |

## 0. 리팩토링 성능 A/B (`mhs_refactor` 스펙 목표 대비)

| 지표 | 24차(리팩토링 전) | 30차(리팩토링 후) | Δ |
| :--- | ---: | ---: | ---: |
| run_elapsed_seconds | 765.0 | 394.9 | −48 % |
| peak_rss_gb | 16.8 | 9.48 | −44 % |
| status | COMPLETE | COMPLETE | — |
| eligible_symbols | 445 | 446 | drift(재수집) |
| realized_execution_roster_size | 41.934 | 41.928 | ≈동일 |

optimization_map: fork_shared_payload(submit 인자 피클링 제거)/plan_worker_count(psutil 기반 워커 산정)/assert_fork_admission(fork 직전 배리어)/`_candidate_weight_books`(fold-safe↔top-level 중복 빌드 통합)/죽은 미닛프레임 캐시 제거. spec: `docs/specs/mhs_refactor.md`.

## 1. Top-level book 성과 (2021-2025, 5m 집행, horizon_ensemble, 30차)

| Book | Horizon | Autocorr Sharpe | Naive Sharpe | Net Ann | CAGR | MaxDD | Turnover(x/yr) | Stress Sharpe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slow_momentum | 168h | +0.526 | +0.462 | +9.72 % | +7.84 % | −22.7 % | 42.7 | +0.142 |
| blend | 168h | +0.520 | +0.468 | +9.18 % | +7.56 % | −20.2 % | 42.0 | +0.129 |
| fast_reversal | 48h | −0.840 | −0.789 | −7.26 % | −7.39 % | −32.2 % | 38.5 | −1.136 |

| key | value |
| :--- | :--- |
| blend.target_gross | 0.7507 |
| blend.cash_fraction | 0.2493 |
| deflated_sharpe_ratio | 0.4265 |
| trials_attempted | 70 |
| termination_counts.MISSING_DATA | 62 |
| termination_counts.UNKNOWN_TERMINATION | 0 |
| fast_reversal.verdict | autocorr/naive/stress 전부 음수, capital_allocation=0 |

## 2. Anchored folds (strict primary, horizon `frozen_default` 168h/48h, 30차)

| Fold | Autocorr Sharpe | Naive Sharpe | Stress Sharpe | Net Ann | Failures |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | +0.805 | +0.812 | +0.211 | +9.66 % | — |
| 1 | −0.267 | −0.308 | −0.820 | −4.19 % | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | +1.505 | +1.134 | +0.931 | +48.2 % | — |

| key | value |
| :--- | :--- |
| research_go.eligible | False |
| research_go.reason_codes | PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY |
| research_go.evaluated_folds | 3 |
| research_go.folds_passed | 2/3 |

## 3. Discovery gate (2021-2023, admission_t=2.0, 30차, `--discovery-gate`)

| 후보 | admitted | selected_horizon |
| :--- | :--- | :--- |
| momentum | False | null |
| reversal | False | null |
| funding_carry_long | False | null |
| funding_carry_short | False | null |

| key | value |
| :--- | :--- |
| horizon_diagnostics.slow_horizon_effective_breadth | 1.4129/19 (7.4%) |
| horizon_diagnostics.fast_horizon_effective_breadth | 1.5727/7 (22.5%) |
| horizon_diagnostics.realized_vol_48h_mean | 0.09126 |
| horizon_diagnostics.efficiency_ratio_48h_mean | 0.14640 |

## 4. `full_history_yearly_net_t` — 5년 전체 (report-only, 30차)

| 연도 | slow_momentum | fast_reversal | funding_carry |
| :--- | ---: | ---: | ---: |
| 2021 | −0.145 | −1.329 | −4.249 |
| 2022 | +0.169 | +0.171 | −1.813 |
| 2023 | −0.568 | −0.797 | −2.176 |
| 2024 | +0.249 | −0.649 | −2.163 |
| 2025 | +1.690 | +0.116 | −1.016 |

| key | value |
| :--- | :--- |
| funding_carry_worst_year_corr (2023) | −0.2657 |

## 5. 통계 진단 (전 구간, 30차)

| 계측 | 값 |
| :--- | :--- |
| `xs_rank_ic.n_dates` | 43,727 |
| `xs_rank_ic.mean_ic` | −0.04088 |
| `xs_rank_ic.t` | −46.07 (fwd=48) |
| `date_clustered_regression.n` | 8,270,040 |
| `date_clustered_regression.n_dates` | 1,826 |
| `date_clustered_regression.past_beta` | −0.01868 |
| `date_clustered_regression.past_t` | −1.39 |
| `bootstrap_ci` (net_1h) | [−6.68e-6, +2.77e-5] |
| `placebo_sharpe_percentile` | null |
| deployment.CAGR | +7.57 % |
| deployment.MaxDD | −20.15 % |
| deployment.Calmar | 0.3754 |
| deployment.P(최종재산<초기) | 16.45 % |
| deployment.P(MDD>20%) | 89.85 % |
| deployment.P(MDD>30%) | 40.0 % |
| deployment.execution/pilot/scale_go_eligible | False/False/False |

verdict: xs_rank_ic 유의 음수(역행), date_clustered t=−1.39 유의하지 않음(48h 전방가격 예측력 없음).

## 6. 전략 요약 (분류 데이터, 30차)

| 전략 | 방향성 | 자본 배분 | 5yr CAGR/Sharpe | 상태 |
| :--- | :--- | ---: | :--- | :--- |
| slow_momentum | market-neutral (횡단면 랭크, dollar-neutral) | 100% | CAGR +7.84%, autocorr +0.526 | capital_active |
| blend | market-neutral (=momentum, cash_fraction 25%) | 100%(=momentum) | CAGR +7.56%, MaxDD −20.2% | capital_active |
| fast_reversal | market-neutral | 0% | autocorr −0.840, 5yr 전 지표 음 | rejected (§1,§4) |
| funding_carry | market-neutral(펀딩비 캐리) | 0% | 5yr 전 해 net_t 음 | rejected (§4) |
| discovery_gate | N/A (검증 절차, 비거래) | N/A | momentum/reversal/funding_carry 전원 rejected | root_cause=2022 crash dominance |

## 7. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| Research-GO 미달(fold1 autocorr/stress 미달) 원인 재검토 | 미착수 |
| momentum discovery window worst-of-3(2021-2023) 설계 재검토 | 미착수, 별도 스펙+사용자 승인 필요 |
| `fast_book_mode`/`funding_carry` 자본 배분 최종 판정 | 미착수, 사용자 승인 필요 |
| `--committee-book`/`--multi-feature-book`/`--trend-sleeve` 리팩토링 후 재실측 | 미착수 |
