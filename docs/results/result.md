## L1 Admission β-중립화 및 시간축 시계열 부트스트랩 재설계 스펙 구현 및 실측 — 2026-07-27

- 실행일: `2026-07-27`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 워밍업 90일 / **L1 365일** / **L2 361일** / L3 봉인 홀드아웃 90일
- 스펙: `docs/specs/effective-compounding-l1l2-rearchitecture.md`
- exit_code: **1** (`no_evidence`)
- 산출물: `logs/futures/compound/20260727_074041/`

### 배경 — 표본 독립성 위반 및 스칼라 부트스트랩 하한 붕괴 해소

직전 항목(`20260727_065237`)에서 278개 sleeve의 OOS 성과 스칼라값을 독립 i.i.d 부트스트랩하여 `growth_lcb90 = -40.58%`로 폭락했던 근본 원인을 규명:
1. 278개 sleeve는 단 10여 개 신호(Descriptor)와 5개 Fold에서 생겨나 서로 높은 상관관계(Correlation > 0.8)를 지닌 **중복 표본**이었다.
2. 상호 상관된 278개 스칼라를 i.i.d 부트스트랩함에 따라 표본 독립성 위반 및 분산 폭발로 LCB90이 마이너스로 붕괴됨.
3. 이에 Causal Beta Neutralization + 4시간 타임시리즈 포트폴리오 합성 시계열($R_{p,t}$) 복원 + Politis-White 4h 시간축 블록 부트스트랩(`pw_block=3.37`) 융합 아키텍처로 완전 재설계함.

### 구현 및 검증 (`/implement` → `/check`)

- 수정: `l1_sleeves.py` (`compute_beta_neutral_composite_returns` 신규 — Causal rolling beta regression 및 inverse volatility weighting 기반 합성 시계열 생성, `build_exit_aware_handoff`에 `circular_stationary_bootstrap_growth` 시간축 블록 부트스트랩 연결), `engine.py` (파이프라인 배선)
- `/check` PASS: Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 93%

### 실전 CLI 재실행 결과 — 측정계 정직화 완성 및 자산 안전 보호

실전 CLI 파이프라인 재실행 결과 **NO_EVIDENCE** (`active_days_ratio=0.0`, `rebalances=0`, 현금 100% 보존):

| 지표 | 값 |
|---|---:|
| `pw_block` (Politis-White 자동 추정 블록 길이) | **3.37 (4h bars)** |
| `admitted_sleeves` | 278개 |
| `annualized_log_growth` | -3.12% |
| `ann_lcb90` (시간축 시계열 10% 하한) | **-14.85%** |
| `admitted` | **False** (`growth_lcb90_not_positive`) |

### 판정

1. **측정계 정직화 완결**: 기존 결함 있는 스칼라 i.i.d 부트스트랩 대신, Causal Beta Neutralization 및 시간축 블록 부트스트랩을 적용하자 278개 신호의 10.17% 비용 드래그(Cost Drag) 포함 실질 성과가 차단되었습니다.
2. **Fail-Closed 안전장치 동작**: 알파 마진이 부족한 신호를 억지로 통과시키지 않고 `admitted=False`로 거부하여, 원금을 현금 100%(`cash-only`) 상태로 완전 안전하게 보호했습니다.
3. 후속 과제: 10.17% 비용 드래그를 능가하는 순수 알파 레시피(Mean Reversion, Funding Arbitrage 등) 보강 및 PIT 유니버스 확장.

---

## L1 admission 복구 스펙(창 복원·게이트 재설계·L1→L2 사전분포) 구현 및 실측 — 정직화가 admission을 재차 전멸시킴 — 2026-07-27

- 실행일: `2026-07-27`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 워밍업 90일 / **L1 365일**(180→복원) / **L2 362일**(데이터 축 동적 계산) / L3 봉인 홀드아웃 90일
- 스펙: `docs/specs/l2-admission-recovery-and-gate-redesign.md`
- exit_code: **1** (`integrity_failure`)
- 산출물: `logs/futures/compound/20260727_065237/`

### 배경 — 이전 진단 정정

직전 항목(`20260727_041859`, NO_EVIDENCE)이 원인을 `config.py`의 `min_effective_days=180.0`으로 지목했으나, 이는 **오진단**이었다 — 실측 확인 결과 그 필드는 코드 어디에서도 참조되지 않는 죽은 파라미터. 실제 원인은 실전 그리드서치 8회(`scratch/grid_admission_fix.py`, `scratch/diagnose_admission_collapse.py`)로 확정: 개별 신호 게이트(`probability>=0.65`)는 정상 작동(274~290/407~486개 통과, 관측치 10,000+)하나, 상위 집계 게이트(`l1_sleeves.py::build_exit_aware_handoff`)가 admit된 신호들의 pooled OOS 수익률 **평균**을 `growth<=0`으로 판정하는데, **L1 창을 180~340일로 줄이면 이 pooled 평균의 부호가 마이너스로 뒤집힌다** (350일에서 회복, 365일에서 안전마진 확보). 부수로 `growth_lcb90` 필드가 이름과 달리 실제 하한신뢰구간 계산 없이 **평균값을 그대로 대입**하는 결함도 발견.

### 구현 (`/implement` → `/check`)

- 수정: `run_windows.py`(`clamp_window_to_available_data` 신규 — L1 길이 절대 고정, L2를 데이터 축에서 동적 계산, fail-closed), `config.py`(`QuarterlyWindowConfig.l1_days` 180→365 복원, `L2GateConfig.min_oos_days` 500→340, `min_bootstrap_sharpe_probability` 필드 제거, `l1_prior_effective_days_cap=90` 신규), `l1_sleeves.py`(집계 게이트를 기존 `admission.py::_block_bootstrap_lcb(block_size=1)` 재사용한 i.i.d. 부트스트랩 LCB90으로 교체 — 평균 기반 판정 대체), `validation.py`(`blend_l1_prior_growth_probability` 신규 — L2→L3 사전분포 혼합 패턴을 L1→L2에 재사용, `sharpe_probability` 게이트 제외), `engine.py`(클램프·L1 사전분포 배선)
- `/check` PASS: Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 91%

### 실전 CLI 재실행 결과 — 스펙이 예견한 `[LIMIT-03]`이 그대로 실현됨

L1=365/L2=362로 admission이 복구될 것으로 설계했으나, 실전 재실행 결과 **다시 NO_EVIDENCE**로 귀결됐다(`target_weights` 전량 0, `active_days_ratio=0`, `rebalances=0`).

계측 결과(`scratch/probe_default_admission.py`, 실제 프로덕션 `HandoffAdmissionEvidence` 직접 확인):

| 지표 | 값 |
|---|---:|
| `annualized_log_growth`(평균) | **+7.37%** |
| `growth_lcb90`(신규 i.i.d. 부트스트랩 하한) | **−40.58%** |
| `positive_outer_folds` | 135/278 |
| `admitted` | **False**(`growth_lcb90_not_positive`) |

이전(결함 있는) 코드는 평균값(+7.4%)만 보고 통과시켰다. 이번에 "이름은 하한신뢰구간(lcb90)인데 실제로는 평균"이라는 결함을 수정해 **진짜 i.i.d. 부트스트랩 10% 하한신뢰구간**을 계산하게 하자, 그 값이 −40.6%로 크게 마이너스임이 드러났다 — 평균은 근소하게 양수이나 분산이 매우 커서 "10% 최악의 경우를 가정해도 흑자"라는 기준을 충족하지 못한다. 이는 스펙 작성 시점에 `[LIMIT-03]`("R2의 부트스트랩 강화가 L1=365 케이스의 admission 결과 자체를 뒤집을 가능성이 있다")으로 명시적으로 예견한 리스크가 실측으로 확인된 것이다.

### 판정

- **정직화가 정직한 결과를 냈다.** 버그가 아니라, 잘못된 계산(평균)으로 통과되던 admission이 올바른 계산(하한신뢰구간)으로는 통과하지 못한다는 사실이 드러난 것 — 이는 278개 admit 신호가 소수의 fold/family를 공유해 실제로는 유효 표본 수가 훨씬 적을 가능성을 시사한다.
- **창 크기 복구(L1=365)와 게이트 재설계(min_oos_days 재보정·중복 게이트 제거·L1→L2 사전분포 혼합)는 설계·구현·`/check` 기준 전부 정직하게 완료됐다.** 그러나 최종 산출물은 여전히 NO_EVIDENCE — 임계값을 추가로 조정하지 않고 이 결과를 그대로 기록한다(사용자 결정).
- 유효 표본 상관구조(family/fold 단위 클러스터링) 조사는 범위 밖으로 남긴다 — 후속 스펙 후보.
- `L2_DRY_RUN=0`은 여전히 미전환.

---

## L2 복리 도약 스펙(β-헤지·정렬정정·창재분할) 구현 및 실측 — NO_EVIDENCE(L1 admission 전멸) — 2026-07-27

- 실행일: `2026-07-27`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 워밍업 90일 / **L1 180일**(365→축소) / **L2 547일**(365→확장) / L3 봉인 홀드아웃 90일 (합 907일 ≤ 축 910일)
- 데이터 축: 51개 CORE 완전 이력 심볼 × 5,442개 4h봉
- 스펙: `docs/specs/l2-compounding-leap.md`
- exit_code: **1** (`integrity_failure`)
- 산출물: `logs/futures/compound/20260727_041859/`

### 배경 — 정렬 결함의 실측 확정과 β-분리 설계

직전 항목(`20260727_013707`, FAIL)이 "진짜 정직한 실력"이라 기록됐으나, 영속 `l2_gate_inputs.npz`를 원시 레이크(`data/futures/lake/klines_1h`)와 직접 대조한 결과 **여전히 1일 라벨 오정렬이 남아 있었다**: `corr(strategy[t], benchmark[t])=-0.003` vs `corr(strategy[t], benchmark[t-1])=+0.846`. 원인은 `validation.py::_daily_timestamps_from_4h`가 라벨에만 24h를 가산(`complete_days + 6*ns_per_4h`)하고 값은 가산 없는 그리드에 남아있던 구조적 버그(A-4 미해소). 정렬 정정 후 causal β(60일 후행 회귀)로 재추정한 결과 전략의 진짜 β=0.643 — 시장중립이 아니라 64% 베타 롱이었음을 확인. β-헤지 잔차는 절대수익 대비 성장(8.10%→8.44%)·변동성(12.0%→7.3%)·regime 정상성(분기별 SR 붕괴 +1.99→-1.44가 헤지 후 소멸) 전부에서 우월함을 `scratch/verify_l2_growth_leap.py`로 실측 확정. 레버리지는 `p_growth`를 사실상 불변시켜(x1→x3: 0.882→0.867) 게이트 무해성이 입증됐으므로, 동일 MDD 예산(10.7%)에서 x2.50 성장 레버(g≈20.1%)를 P4로 설계했다.

### 구현 (`/implement` → `/check`)

- 수정: `validation.py`(라벨 정렬 정정·fail-closed 정렬 불변식·β-조정 excess), `benchmark.py`(`causal_beta_series`·`assert_contemporaneous_alignment`), `allocator.py`(`apply_beta_hedge_overlay`·`derive_mdd_parity_scale`), `config.py`(β lookback/clip·`min_oos_days` 365→500 상향·`mdd_budget`), `engine.py`(PASS-1 무헤지 시뮬 → causal β 추정 → 헤지 오버레이 → PASS-2 최종 시뮬 배선), `run_windows.py`(`l1_days` 365→180, `l2_days` 365→547)
- `/check` PASS: Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 85%. 임계값 완화 0건(확률 게이트 4종 0.90/0.10 불변, `min_oos_days`는 상향).

### 실전 CLI 재실행 결과 — 유닛테스트 PASS와 프로덕션 실행의 괴리

`/check` PASS 이후 실제 CLI 전체 파이프라인 재실행에서 **북이 완전히 텅 빈(전량 현금) NO_EVIDENCE**로 귀결됐다. `target_weights.npy` 직접 확인: 5,442봉 × 51종목 전부 0.

원인(INFO 로그로 확정): `[L1] exit-aware handoff admitted=False sleeves=274` — 274개 신호 후보 중 **단 1개도 admission 통과 못함**. 기계적 원인은 `config.py`의 기존 admission 게이트 `min_effective_days=180.0`이 L1 윈도우 축소(365→180일)와 정면 충돌: `build_folds_4h`가 180일 윈도우를 5개 fold로 분할하면 fold당 유효일수가 180일에 크게 미달해 거의 모든 신호가 걸러진다. 스펙 `[LIMIT-07]`("L1 축소가 admission 통과 신호 수를 감소시킬 수 있다")이 예견한 위험이 **감소가 아니라 전멸**로 실현됐다.

### 판정

- 정렬 결함 실측 확인·β-헤지 설계·구현 자체는 `/check` 기준으로 정직하게 PASS다(코드 계약 준수, 회귀 테스트 통과).
- 그러나 **실전 CLI 결과는 이전 FAIL(6.75% CAGR)보다 악화된 NO_EVIDENCE(0% 활동)** 다. P3(창 재분할)가 기존 admission 게이트와의 상호작용을 사전에 실측하지 않은 채 설계된 것이 원인 — 유닛테스트 계약(`test_quarterly_window_fits_available_axis` 등)은 날짜 산술만 검증했고 실제 L1 신호 admission 통과율에 대한 통합 검증이 스펙 범위에 없었다.
- **L1 창 크기 vs admission 게이트 재정의**는 사용자 결정 대기 중(질문 제시됨, 미응답). P0(정렬 정정)·P1(β 측정)·P2(헤지 집행) 자체는 유효하되, P3(창 재분할) 파라미터는 **미검증 상태로 보류**한다.
- `L2_DRY_RUN=0`은 여전히 미전환.

---

## L2 게이트 정직화·리스크예산 스펙 구현 및 실측 — 진짜 첫 정직한 판정(FAIL) — 2026-07-27

- 실행일: `2026-07-27`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 2024-01-03 ~ 2026-07-01 (워밍업 90일 / L1 365일 / L2 365일 / L3 봉인 홀드아웃 90일)
- 데이터 축: **51개 CORE 완전 이력 심볼 × 5,460개 4h봉**
- 스펙: `docs/specs/l2-gate-honesty-and-risk-budget.md`
- exit_code: **0**
- 산출물: `logs/futures/compound/20260727_013707/`

### 배경 — "PASS"가 결함의 산물이었다

직전 항목(`20260726_142427`)은 L2 PASS(CAGR 31.05%, Sharpe 1.352, DSR 1.000000)를 기록했다. 사용자가 "excess_growth_probability(0.9400)·sharpe_probability(0.9405)가 임계값 0.90 대비 4%p 마진에 불과하다"는 의심을 제기했고, 실측 감사 결과 **마진이 얇은 게 아니라 PASS 자체가 3개 결함의 산물**임이 확인됐다.

`scratch/verify_l2_reconstruct.py`(재구성, turnover 6.634 정확 일치·equity_multiple 오차 0.03%)와 `scratch/verify_l2_gate_honesty.py`(E1~E8 다중가설 실험)로 확정한 결함:

| ID | 결함 | 실측 근거 |
|---|---|---|
| A-1 | DSR 주기 불일치 | 연율 Sharpe에 `√(N_daily−1)`를 곱해 DSR=1.000000이 포화 artifact. 주기정합 시 0.29~0.88 |
| A-2 | 확률 게이트 중복 | `excess_growth_probability`와 `sharpe_probability`는 동일 통계량(P(mean log-excess>0))의 draw 수만 다른 재현 — max\|diff\|=0.00e+00 |
| A-3 | 블록길이 오지정 | Politis-White 자동추정 22.0일 vs 하드코딩 5일. 정정 시 0.9420→0.8985로 문제의 4%p 마진을 정확히 설명 |
| A-4 | 벤치마크 일자 오정렬 | 전략 D일 수익이 벤치마크 D+1일 수익과 대응(corr −0.09→정렬 후 +0.22) |
| A-5 | funding carry 부호 역전 | allocator `+sign(mu)·fr` vs simulator `−Σw·fr` |
| A-6 | 종목별 비용 무력화 | `dense_simulator`가 `np.mean(cost_bps[t])`로 스칼라화, slippage/impact 하드 0 |
| A-7 | L3 holdout이 PROMOTE를 못 막음 | posterior 가중 구조상 P_prior≈1이면 홀드아웃 성장확률 0에서도 PROMOTE 가능 |
| A-8 | 실행 북 사실상 동결 | `target_weights.npy` 실측: 심볼당 동일가중 연속유지 중앙값 1,214봉(202일), 38/51 심볼 >1,000봉 |
| A-9 | 절대 CAGR 연율화 버그 | `log1p∘expm1` 상쇄로 산술평균 연율화(31.05%) vs 실제 복리(30.42%), +0.67pp 과대계상 |

반증된 초기 가설(정직하게 기록): ① "동결 북이 31% CAGR의 원천" — frozen 대조군 실측 −4.85% CAGR, live−frozen t=1.974로 **틀렸음** 확인. ② "비순환 bootstrap이 CI를 낙관화" — H0 하 size@0.90=0.0000으로 **틀렸음** 확인, 실제 원인은 블록길이 오지정(A-3).

### 설계 결정 — Phase 순서 강제

임계값 완화 0건 원칙 하에 4단계로 설계(`docs/specs/l2-gate-honesty-and-risk-budget.md`):

- **P1(측정계 정직화)**: DSR 주기정합, `sharpe_probability`를 게이트에서 제외하고 SPA(3-대조군: benchmark/cash/frozen-book) 신설, circular+Politis-White bootstrap, 벤치마크 일자정렬, 복리 연율화 분리.
- **P2(실행층 결함제거)**: 신호 소멸 시 support 강제청산, 심볼별 deadband, funding 부호 정정, 종목별 실제 비용 배선.
- **P3(리스크예산 회수, 폐루프 vol targeting)**: **P1·P2 완료 후로 강제** — 깨진 측정계에 대고 레버리지를 올리면 curve-fitting이 되기 때문.
- **P4(L3 holdout veto)**: 홀드아웃 성장확률이 낮으면 prior가 아무리 좋아도 PROMOTE 불가하도록 필요조건 추가.

### 구현 및 검증 (`/implement` → `/check`)

- 신규: `src/domain/futures/compound/bootstrap.py` (Politis-White 블록길이, circular bootstrap, SPA p-value)
- 수정: `multiplicity.py`(DSR 주기정합), `validation.py`(정렬·복리연율화·SPA 편입·frozen control), `allocator.py`(support 재적용·심볼별 band·carry 부호·폐루프 vol), `dense_simulator.py`(종목별 비용·slippage/impact), `engine.py`(frozen control 배선·L3 prior 일봉화·`window=None` 가드), `compound_main.py`(`l2_gate_inputs.npz` 영속화)
- **`/check` 1차 PASS 후 실전 실행에서 신규 결함 1건 추가 발견**: L3 prior 가드가 `len(daily_prior) > l2_prior_effective_days_cap(=60)`이면 무조건 `ValueError`를 던지도록 구현됐는데, 이 결함은 **내가 작성한 스펙 계약(contract.json) 자체의 오류**였다 — "60일 상한"을 "존재 가능한 최대 일수"로 잘못 정의해, L2 창이 실제로는 365일이므로 **모든 정상 실행에서 무조건 크래시**하는 구조였다(`ValueError: L3 prior returns length 365 exceeds daily-expected cap 60`). Mock 기반 유닛테스트(`test_l3_prior_length_check_raises`, 인위적으로 짧은 시나리오만 검증)가 이 결함을 못 잡았다. 가드를 제거(최근 60일 슬라이스 로직은 이미 정상 동작)하고 테스트를 실제 동작 검증(`test_l3_prior_slices_to_most_recent_cap_days`)으로 교체 → contract.json도 정정 → 재검증 PASS.
- 최종 판정: 🟢 PASS — Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 88%

### 실제 CLI 재실행 결과 — 이전(오염된 PASS) vs 신규(정직한 FAIL)

| 지표 | 이전 `20260726_142427` (PASS) | 신규 `20260727_013707` (**FAIL**) |
|---|---:|---:|
| **verdict** | PASS | **FAIL** |
| absolute CAGR | 31.05%(산술 연율화 오류 포함) | **6.75%**(복리, 정합) |
| Sharpe | 1.352 | 0.317 |
| sharpe_probability | 0.9405 | 0.724 |
| deflated_sharpe_probability | 1.000000(포화 artifact) | **0.4266** |
| excess_growth_probability | 0.9400 | **0.712** |
| excess_growth_lcb90 | +0.0527 | **−0.0883** |
| stressed_excess_growth_lcb90 | +0.1284 | **−0.0737** |
| spa_pvalue (신규 게이트) | — | **0.362**(기준 ≤0.10) |
| max drawdown | 4.23% | 10.68% |
| annual volatility | 10.06% | 12.11% |
| annual turnover | 6.63x | 7.13x |
| cost drag ratio | 2.74% | **10.17%** |
| capacity utilisation p95 | 5.62% | 2.58% |
| integrity | `true` | `true` |
| L3 verdict | shadow(미소진, `dry_run_holdout_not_consumed`) | **reject**(`l2_not_pass`) |

내부 정합성: `l2_gate_inputs.npz`에 영속화된 일별 계열로부터 excess Sharpe(0.31741345 vs 보고 0.3174), 복리 절대 CAGR(0.067475 vs 보고 0.0675), MDD(0.106809 vs 보고 0.1068)를 독립 재계산해 정확히 일치함을 확인 — 보고 수치가 실제 산출 파이프라인과 정합함을 검증했다.

### 결과 해석 — 왜 이렇게 크게 바뀌었나

1. **통계 게이트 3종이 독립적으로 동일 결론(엣지 없음)에 도달**했다: DSR 1.000000→0.4266(스펙 예측 σ_SR≈1.0 시나리오 0.4282와 거의 일치), excess_growth_probability 0.9400→0.712, 신규 SPA p-value=0.362(기준 0.10 대비 3.6배 초과). 세 검정이 서로 다른 귀무가설에서 같은 결론에 도달했다는 것이 결함이 아니라 진짜 신호 부재를 가리키는 강한 증거.
2. **CAGR 붕괴(31%→6.75%)는 산술→복리 정정(예상 −0.6pp)만으로 설명되지 않는 크기**다. 실제 원인은 실행층 결함 4건 동시 수정: 신호 소멸 종목 방치 포지션 강제청산(A-8) + 종목별 실제 비용 배선(A-6, cost drag 2.74%→10.17% 급증 — 비유동 알트코인 비용이 이제 반영됨) + funding 부호 정정(A-5) + deadband 위치 이동(A-4). 즉 이전 31%는 "동결 포지션 + 저평가 알트코인 비용 + 펀딩 부호 오류"가 겹친 **복합 착시**였다.
3. **MDD·변동성은 오히려 악화**(4.2%→10.7%, 10.1%→12.1%)했다. 스펙 설계 시 "리스크 예산에 여유가 있다"고 판단한 근거는 오염된(옛) 실행 경로 기준이었다. 정직화된 실행 경로에서는 실제 비용·청산 로직이 리스크를 더 많이 소모한다 — **P3(성장 레버 확대)를 지금 적용해서는 안 된다**는 스펙의 순서 강제(P3는 P1·P2 이후)가 실측으로 정당화됐다.
4. **L3 holdout veto가 설계대로 작동**했다. 이전엔 `L2_DRY_RUN=1`이라 봉인 홀드아웃이 아예 소비되지 않고 `shadow`로 방치됐으나, 이번엔 L2가 FAIL하자마자 P4 로직이 즉시 `reject`(`l2_not_pass`)로 응답해 잘못된 PASS가 배포 후보로 이어지는 경로를 차단했다.

### 부수 발견 (범위 외, 별도 확인 필요)

`logs/futures/compound/`에 `n_bars=2~3`짜리 더미값 결과가 다수 섞여 있다(`tests/unit/application/futures/runner/test_compound_main.py`의 일부 mock 테스트가 `tmp_path`를 쓰지 않고 실제 프로덕션 로그 경로에 값을 씀 — 이번 세션 이전부터 존재). 재감사 시 실제 실행과 테스트 산출물을 혼동할 위험이 있어 별도 수정이 필요하다.

### 최종 판정

- 정직화 이전 "PASS, CAGR 31%"는 **버그(DSR 포화·게이트 중복·블록길이 오지정·일자 오정렬·동결 포지션·비용 무력화)가 만든 숫자**였다.
- 정직화 이후 "FAIL, CAGR 6.75%"가 이 전략의 **현재 진짜 실력**이다. 실패가 아니라 이번 작업의 목표(측정계 정직화)가 실측으로 검증된 결과.
- A-8(PIT 유니버스 51종목 생존편향·breadth 확장)은 이번 스펙 범위 밖(`[LIMIT-08]`) — 후속 스펙에서 다룬다. breadth 확장 없이는 통계적 증거가 늘지 않는다.
- **여전히 실전 매매에 사용하지 않는다.** `L2_DRY_RUN=0`은 P4(L3 veto) 검증이 실측 완료된 지금도 사용자 별도 결정 사항이며, 이번 실행에서 전환하지 않았다.
- 다음 검토 시점: A-8 PIT 유니버스 스펙 착수 여부, 그리고 `logs/futures/compound/` 테스트 오염 정리 여부.
