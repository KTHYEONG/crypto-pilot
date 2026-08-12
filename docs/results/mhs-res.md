# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-12 (17차, `mhs_execution_annualization_fix` 적용 재실행 — `--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger`)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **성격**: `docs/specs/mhs_execution_annualization_fix.md`(체결 원장이 `execution_timeframe=5m` 격자인데 연율화 상수 `_PERIODS_PER_YEAR_1H`가 1시간 격자를 가정해 CAGR/naive Sharpe/연율수익/연환산 회전율/배포준비도 꼬리위험이 전부 12배 과소평가되던 집계 버그)를 실제 코드에 반영한 재실행. 신호·비용·체결 로직은 16차와 완전히 동일(RC-1/RC-2 alpha-engine, `crash_regime_tilt_alpha`/`beta_neutralize`/`ensemble_signal=vol_normalized` 전부 미배선) — **이번 표의 변화는 순수하게 측정 수정분이며 전략이 바뀐 게 아니다.**

## 1. Primary metrics (`slow_momentum` == `blend`, `fast_reversal` blend 자본 0%) — 16차(버그 있음) vs 17차(수정)

| metric | 16차 (연율화 버그) | 17차 (수정 후) | 배율 |
| :--- | ---: | ---: | ---: |
| `primary_autocorr_sharpe` | 0.5257 | 0.5257 | 1.0x(무영향, 설계대로) |
| `primary_naive_sharpe` | 0.1333 | **0.4616** | 3.46x (= √12) |
| `primary_net_ann` | 0.0081 | **0.0972** | 12.0x |
| `primary_geometric_cagr` | 0.0063 | **0.0784** | 12.4x |
| `primary_max_drawdown` | -0.2269 | -0.2269 | 1.0x(무영향, 설계대로) |
| `primary_annualized_turnover` | 3.56 | **42.68** | 12.0x |
| `stress_naive_sharpe` (x3 cost) | +0.0410 | **+0.1420** | 3.46x |
| `blend.failure` | null | null | — |

`docs/specs/mhs_execution_annualization_fix.md`가 예측한 정확히 그 배율(12x 지수, √12 Sharpe류)로 재현됨 — 버그 진단이 실측으로 확정됨. **연복리 수익률 7.84%/년**이 이번 코드베이스의 실행 스택이 실제로 만들어내고 있던 진짜 수치였고, 지금까지 문서화된 0.03~0.63%는 측정 오류였다.

## 2. Research GO gate

| field | 16차 | 17차 |
| :--- | ---: | ---: |
| `eligible` | `false` | `false` |
| `evaluated_folds` | 3 | 3 |
| `folds_passed` | 2 | 2 (무변화 — 게이트는 Sharpe/stress 기준이라 연율화 버그와 무관) |
| `reason_codes` | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` | 동일 |

여전히 `eligible=false` — Sharpe 플로어(0.6)와 `UNSPECIFIED_POLICY`가 차단. **Research GO 게이트 자체는 원래 Sharpe/일봉 기반이라 이번 버그의 영향을 받지 않았다** — 즉 이 버그는 GO/NO-GO 판정을 뒤집지 않았고, 오직 "자산증식이 얼마나 되는지"를 보여주는 CAGR 헤드라인만 왜곡했다.

## 3. Fold detail (17차, CAGR 신규 노출)

| fold | validation | `primary_autocorr_sharpe` | `primary_geometric_cagr` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | 2023 | +0.8046 | **+9.36%** | -0.1089 | +0.2111 | (none) — **통과** |
| 1 | 2024 | -0.2672 | **-4.99%** | -0.1743 | -0.8201 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | 2025 | +1.5047 | **+48.22%** | -0.2185 | +0.9310 | (none) — **통과** |

fold별 CAGR이 처음으로 신뢰 가능한 수치로 노출됨 — fold1(2024)만 명확히 손실 구간, fold0/fold2는 양호.

## 4. 배포 준비도(꼬리위험) — 버그 수정의 부수 효과

`compute_deployment_readiness`의 블록부트스트랩이 `mean_block_bars=168`을 "168시간(1주)"이 아닌 "168개 5분바(14시간)"로 잘못 해석하던 것도 같이 고쳐짐:

| field | 16차(버그) | 17차(수정) |
| :--- | ---: | ---: |
| `probability_mdd_over_30pct` | 62.65% | **47.0%** |
| `probability_final_wealth_below_initial` | 21.35% | **16.8%** |
| `calmar` | 0.028 | **0.345** |

여전히 낙관적인 수치는 아니지만(30%대 낙폭 확률이 여전히 47%), 이전에 보고했던 "62.65% 확률로 30% 넘게 물린다"는 진짜보다 비관적인 추정이었다.

## 5. 부수 검증 — 인과성 수정(RC-3, 16차에서 이미 확인, 17차도 동일)

동일 실행에서 `xs_rank_ic`/`date_clustered_regression`이 처음으로 거래가능 선행수익률로 계산됨:

| field | 값 |
| :--- | ---: |
| `xs_rank_ic.mean_ic` | **-0.0409** (수정 전 겹침 통계는 +0.0957이었음) |
| `xs_rank_ic.t_stat` | **-46.07** |
| `date_clustered_regression.past_beta` | -0.0187 |
| `date_clustered_regression.past_t` | -1.39 |

부호 반전은 `docs/specs/mhs_alpha_engine.md` §3의 사전 스크래치 측정(`-0.0278`, t=-33)과 방향이 일치 — 헤드라인 "신호 품질" 지표가 지금까지 거래 불가능한 겹침 구간으로 계산돼 왔음을 실제 프로덕션 리포트에서도 확인.

## 6. 최근 진단·조사 계보

- **`OHLCV_IMMEDIATE_TAKER` 결손 가드**(`ADR_20260812_MHS_IMMEDIATE_TAKER_FILL_GUARD`): 진단 완주 안정화.
- **fold Sharpe 편차 근본 원인**(`ADR_20260812_MHS_MOMENTUM_REGIME_DIAGNOSIS`): 2022(LUNA/FTX) 전원 음수 — horizon 선택이 아닌 신호의 체계적 붕괴장 취약성.
- **전략 유형 재검토**(`ADR_20260812_MHS_MOMENTUM_STRATEGY_REDESIGN_REVIEW`): 완전 시장중립과 붕괴장 생존의 트레이드오프 확인, `src/mhs/regime.py` 레짐 프록시 신설.
- **크래시 레짐 방향성 틸트 오버레이**(`ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`, 15차): `alpha=0.2` 실전 검증 — `folds_passed=1/3`, `stress_naive_sharpe=-0.0844`.
- **MHS 알파 엔진 재구축**(16차, `docs/specs/mhs_alpha_engine.md`): RC-1(종목별 데드밴드가 달러중립·레짐 디리스킹을 파괴) → 포트폴리오 리밸런스 트리거, RC-2(단일 호라이즌 argmax 고분산 선택) → 19-호라이즌 동일가중 앙상블. `primary_autocorr_sharpe` 0.182→0.526, MDD -0.392→-0.227, `folds_passed` 1→2/3.
- **실행 원장 연율화 버그 수정**(이번 17차, `docs/specs/mhs_execution_annualization_fix.md`): `execution_timeframe=5m` 원장에 1시간 격자 상수를 적용하던 집계 버그(지수 12배 축소) 규명·수정. 전략 무변경, CAGR 0.63%→**7.84%**, 연율수익 12배, 낙폭확률(30%+) 62.65%→47.0%로 재노출. Research GO 게이트(Sharpe 기반)는 버그 영향을 받지 않아 무변화.

## 7. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| `beta_neutralize`, `ensemble_signal=vol_normalized`, 트리거 임계값 — fold-train-only 선택 절차로 확정 필요 | 미착수 (`docs/specs/mhs_alpha_engine.md` §6.1: 전 구간 표로 확정 금지) |
| 시계열(방향성) 절대모멘텀 재검토 — 사전스크린 Sharpe 1.0~1.4로 횡단면 앙상블(0.72~0.86)보다 높게 관측됨 | 미착수, 연율화 버그 수정 후 fold-train-only로 재검증 필요(`docs/specs/mhs_execution_annualization_fix.md` §3) |
| 펀딩비 캐리 신호 — 손익 분해 부호 모순 재검증 필요 | 미착수, 액면 그대로 신뢰 금지 |
| fold1(2024) 잔여 미달 원인 조사 | 미착수 |
| `crash_regime_tilt_alpha` 진단 전용 격하 후 재측정 여부 | 미착수 |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본, 2025-07-14) 근본 원인 조사 | 미착수 (Research GO엔 무관) |
