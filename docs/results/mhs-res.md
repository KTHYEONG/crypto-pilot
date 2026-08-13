# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-13 (24차, 실행 성능 최적화 적용 후 Full 5y 프로덕션 구성 재실측 — `ADR_20260813_MHS_EXECUTION_PERFORMANCE_OPTIMIZATION`. 23차 이전 이력은 git 이력으로 복구 가능)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run**: `start=2021-01-01 end=2025-12-31 execution_timeframe=5m execution_universe_size=30 eligible_symbols=445 realized_execution_roster_size=41.934179584940985 run_elapsed_seconds=765.0 peak_rss=16.8GB`
- **CLI**: `--slow-book-mode horizon_ensemble --fast-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full`
- **성능 최적화 적용**: C1 mark 프레임 캐시 / C2 window materialize+pass 재사용 / C3 window 단위 parquet 로드(preload 폐기) / C4 fold-safe discovery 병렬화

> ⚠️ **데이터 drift 공지**: 본 실행의 `eligible_symbols=445`로 23차의 446과 다름(Parquet 소스 재수집). 동일 데이터 기준 원본 코드 대비 출력은 **bit-identical**(아래 §0 A/B). 수치 비교 시 23차 대비 차이는 코드 변경이 아닌 데이터 drift로 해석해야 함.

---

## 0. 성능 최적화 A/B (동일 데이터, 2026-08-13)

| 지표 | 원본 코드 | 최적화 코드 (C1-C4) | Δ |
| :--- | ---: | ---: | ---: |
| Full 5y wall (기본 5m, gate off) | 684.9 s | **309.1 s** | **−55 %** |
| Full 5y peak RSS (기본 5m, gate off) | 5.40 GB | **3.15 GB** | **−42 %** |
| `slow_momentum.primary_autocorr_sharpe` | −0.549471229370105 | −0.549471229370105 | **bit-identical** |
| `blend.primary_autocorr_sharpe` | 0.452119924579761 | 0.452119924579761 | **bit-identical** |
| `realized_execution_roster_size` | 41.934179584940985 | 41.934179584940985 | **bit-identical** |
| `test_mhs_replay_resources` checksum | `b7a7ffba…` | `b7a7ffba…` | **bit-identical** |
| 6mo book worker / fold worker | 31.7 s / 61.5 s | 7.1 s / 17.5 s | −78 % / −72 % |

- gate 구성(위 Run)의 peak RSS 16.8GB는 discovery 후보 웨이트 행렬(~13GB, slow 19 + fast 7 + funding 2×8 호라이즌 full-panel)이 지배 — 기존 구조 고유 비용이며, fold-safe와 top-level gate가 동일 후보를 중복 빌드하므로 dedupe로 추가 절감 여지 있음(후속 후보).

---

## 1. Top-level book 성과 (2021-2025, 5m 집행, horizon_ensemble)

| Book | Horizon | Autocorr Sharpe | Naive Sharpe | Net Ann | CAGR | MaxDD | Turnover(x/yr) | Stress Sharpe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slow_momentum | 168h | **+0.536** | +0.472 | **+9.95 %** | +8.08 % | −22.7 % | 42.8 | +0.152 |
| blend | 168h | +0.536 | +0.484 | +9.50 % | +7.91 % | **−20.2 %** | 42.0 | +0.146 |
| fast_reversal | 48h | −0.840 | −0.789 | −7.26 % | −7.40 % | −32.2 % | 38.6 | −1.137 |

- blend `target_gross=0.7507`, `cash_fraction=0.2493`, `deflated_sharpe_ratio=0.5470`.
- `termination_counts`: `MISSING_DATA=62`, `UNKNOWN_TERMINATION=0` (전 구간 데이터 gap 62건, leak 없음).
- fast_reversal은 ensemble 구제책 적용 후에도 autocorr·naive·stress 전부 음수 — **자본 가치 없음** 재확인.

## 2. Anchored folds (strict primary, horizon `frozen_default` 168h/48h)

| Fold | Autocorr Sharpe | Naive Sharpe | Stress Sharpe | Net Ann | Failures |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | +0.805 | +0.812 | +0.211 | +9.66 % | — |
| 1 | **−0.267** | −0.308 | **−0.820** | −4.19 % | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | **+1.505** | +1.134 | +0.931 | **+48.2 %** | — |

- **Research-GO**: ❌ `eligible=False`, `reason_codes=[PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY]`, `folds_passed=2/3`.

## 3. Discovery gate & breadth (2021-2023, admission_t=2.0)

| 후보 | admitted | selected_horizon |
| :--- | :--- | :--- |
| momentum | **False** | null |
| reversal | **False** | null |
| funding_carry_long | **False** | null |
| funding_carry_short | **False** | null |

- `effective_breadth`: slow_horizon **1.4120/19** (7.4 %), fast_horizon **1.5720/7** (22.5 %) — 호라이즌 축 포화 재확인.

## 4. `yearly_net_t_diagnostic` — 5년 전체 (admission 미입력, 보고용)

| 연도 | slow_momentum | fast_reversal | funding_carry |
| :--- | ---: | ---: | ---: |
| 2021 | −0.145 | −1.329 | −4.249 |
| 2022 | +0.169 | +0.171 | −1.813 |
| 2023 | −0.568 | −0.797 | −2.176 |
| 2024 | +0.249 | −0.649 | −2.163 |
| 2025 | **+1.775** | −0.012 | −1.142 |

- `funding_carry_worst_year_corr`(2023) = **−0.2657** — 최악의 해 분산효과 방향은 있으나 funding_carry 전 해 손실로 자본화 근거 없음.
- fast_reversal은 5년 중 admission_t=2.0 상회 해 **없음**; funding_carry 전 해 음수 → edge 부재 결론 유지.

## 5. 통계 진단 (전 구간)

| 계측 | 값 |
| :--- | :--- |
| `xs_rank_ic` | n_dates=43,704, mean_ic=**−0.04086**, t=**−46.02** (fwd=48) |
| `date_clustered_regression` | n=8,257,895, n_dates=1,826, beta=**−0.01779**, t=**−1.32** |
| `horizon_diagnostics` | realized_vol_48h=0.09112, efficiency_ratio_48h=0.14636 |
| `bootstrap_ci` (net_1h) | **[−6.02e-6, +2.80e-5]** (하한 음수) |
| deployment | CAGR=+7.91 %, MaxDD=−20.15 %, Calmar=0.3926, P(최종재산<초기)=16.25 % |

- xs rank IC는 유의한 **음수** 역행 효과, 클러스터 t=−1.32로 48h 전방가격 예측력 유의하지 않음.

## 6. 회귀 불변식 & 요약

| 항목 | 값 |
| :--- | :--- |
| slow_momentum autocorr (24차, drift 데이터) | 0.5360654316354135 |
| blend autocorr (24차, drift 데이터) | 0.5359443875092911 |
| fast_reversal autocorr (24차, drift 데이터) | −0.840177372029221 |
| deflated_sharpe | 0.546963269158657 |

- 23차 값(0.525673922813482 등)과의 차이는 `eligible_symbols 446→445` 데이터 drift이며, **동일 데이터 A/B로 코드 변경이 출력에 미치는 영향은 0**임을 §0이 증명.
- **결론 유지**: momentum(168h ensemble)만 유일한 생존 후보(5년 CAGR +8 %, fold 2의 +48 % 포함), fast_reversal·funding_carry는 5년 전체에서도 자본 근거 없음, Research-GO 미달 → 자본 배분·`PHASE_1_BOOK_SPECS`/`PHASE_1_BOOK_BLEND_WEIGHTS` 무변경.

## 7. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| momentum 3년 discovery window admission floor(|t|≥2.0) 검정력 재검토 (fold-local 표본 민감도) | 미착수, 사용자 판단 대기 |
| discovery 후보 웨이트 중복 빌드 dedupe (fold-safe ↔ top-level gate 공유, RAM/시간 추가 절감) | 신규, 미착수 |
| `MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`) 등록 여부 | 미착수, 성과 무관 정책 결정 필요 |
| `pnl_vol_target` 기본값 전환 여부 (`mhs_execution_friction_and_exposure_layers.md` §6.1) | 미착수 |
| `fast_book_mode`/`funding_carry` 자본 배분 최종 판정 (본 계약 실측이 선행조건) | 미착수, 사용자 승인 필요 |
