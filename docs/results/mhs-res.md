# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-12 (16차, `mhs_alpha_engine` 실전 검증 — `--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger`)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **성격**: `docs/specs/mhs_alpha_engine.md`(RC-1 종목별 데드밴드 → 포트폴리오 리밸런스 트리거, RC-2 단일 호라이즌 argmax → 19-호라이즌 동일가중 앙상블)의 실전 검증 실행. `crash_regime_tilt_alpha`, `beta_neutralize`, `ensemble_signal=vol_normalized`는 이번 실행에 배선하지 않음(기본값 유지) — RC-1/RC-2만 격리 측정. **이 문서 수치로 `beta_neutralize`/`ensemble_signal`/트리거 임계값 등 나머지 파라미터를 확정하지 말 것**(fold-train-only 선택 절차 필요, `docs/specs/mhs_alpha_engine.md` §6.1).

## 1. Primary metrics (`slow_momentum` == `blend`, `fast_reversal` blend 자본 0%) — 15차 대비

| metric | 15차 (`crash_regime_tilt_alpha=0.2`) | 16차 (alpha-engine RC-1/RC-2) |
| :--- | ---: | ---: |
| `primary_autocorr_sharpe` | 0.1819 | **0.5257** |
| `primary_naive_sharpe` | 0.0399 | 0.1333 |
| `primary_net_ann` | 0.0029 | 0.0081 |
| `primary_geometric_cagr` | 0.00035 | **0.00631** |
| `primary_max_drawdown` | -0.3922 | **-0.2269** |
| `primary_annualized_turnover` | 5.12 | **3.56** |
| `stress_naive_sharpe` (x3 cost) | -0.0844 | **+0.0410** |
| `blend.failure` | null | null |

## 2. Research GO gate

| field | 15차 | 16차 |
| :--- | ---: | ---: |
| `eligible` | `false` | `false` |
| `evaluated_folds` | 3 | 3 |
| `folds_passed` | 1 | **2** |
| `reason_codes` | `CAPITAL_INVARIANT_BREACH`(`fast_reversal` 별개, 무관), `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `RELEVANT_EXECUTION_DATA_GAP`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |

여전히 `eligible=false` — Sharpe 플로어(0.6)와 `UNSPECIFIED_POLICY`(cap-30/연수익 게이트 미등록 정책)가 항상 조건적 차단.

## 3. Fold detail

| fold | validation | `primary_autocorr_sharpe` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 0 | 2023 | +0.8046 | -0.1089 | +0.0610 | (none) — **통과** |
| 1 | 2024 | -0.2672 | -0.1743 | -0.2367 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | 2025 | +1.5047 | -0.2185 | +0.2686 | (none) — **통과** |

fold0가 15차의 `RELEVANT_EXECUTION_DATA_GAP`(무관한 데이터 결손 스킵) 없이 완전 통과로 전환 — `folds_passed` 1→2/3. fold1(2024)만 여전히 미달.

## 4. 부수 검증 — 인과성 수정(RC-3)

동일 실행에서 `xs_rank_ic`/`date_clustered_regression`이 처음으로 거래가능 선행수익률로 계산됨:

| field | 값 |
| :--- | ---: |
| `xs_rank_ic.mean_ic` | **-0.0409** (수정 전 겹침 통계는 +0.0957이었음) |
| `xs_rank_ic.t_stat` | **-46.07** |
| `date_clustered_regression.past_beta` | -0.0187 |
| `date_clustered_regression.past_t` | -1.39 |

부호 반전은 `docs/specs/mhs_alpha_engine.md` §3의 사전 스크래치 측정(`-0.0278`, t=-33)과 방향이 일치 — 헤드라인 "신호 품질" 지표가 지금까지 거래 불가능한 겹침 구간으로 계산돼 왔음을 실제 프로덕션 리포트에서도 확인.

## 5. 최근 진단·조사 계보

- **`OHLCV_IMMEDIATE_TAKER` 결손 가드**(`ADR_20260812_MHS_IMMEDIATE_TAKER_FILL_GUARD`): 진단 완주 안정화.
- **fold Sharpe 편차 근본 원인**(`ADR_20260812_MHS_MOMENTUM_REGIME_DIAGNOSIS`): 2022(LUNA/FTX) 전원 음수 — horizon 선택이 아닌 신호의 체계적 붕괴장 취약성.
- **전략 유형 재검토**(`ADR_20260812_MHS_MOMENTUM_STRATEGY_REDESIGN_REVIEW`): 완전 시장중립과 붕괴장 생존의 트레이드오프 확인, `src/mhs/regime.py` 레짐 프록시 신설.
- **크래시 레짐 방향성 틸트 오버레이**(`ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`, 15차): `alpha=0.2` 실전 검증 — `folds_passed=1/3`, `stress_naive_sharpe=-0.0844`.
- **MHS 알파 엔진 재구축**(이번 16차, `docs/specs/mhs_alpha_engine.md`): RC-1(종목별 데드밴드가 달러중립·레짐 디리스킹을 파괴) 진단 → 포트폴리오 리밸런스 트리거로 교체, RC-2(단일 호라이즌 argmax 고분산 선택) 진단 → 19-호라이즌 동일가중 앙상블로 교체. `primary_autocorr_sharpe` 0.182→0.526, MDD -0.392→-0.227, stress -0.084→+0.041, `folds_passed` 1→2/3.

## 6. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| `beta_neutralize`, `ensemble_signal=vol_normalized`, 트리거 임계값 — fold-train-only 선택 절차로 확정 필요 | 미착수 (§6.1: 이 문서/스펙의 전 구간 표로 확정 금지) |
| fold1(2024) 잔여 미달 원인 조사 | 미착수 |
| `crash_regime_tilt_alpha` 진단 전용 격하 후 재측정 여부 | 미착수 |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본, 2025-07-14) 근본 원인 조사 | 미착수 (Research GO엔 무관) |
