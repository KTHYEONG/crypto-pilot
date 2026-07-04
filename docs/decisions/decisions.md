# Active Decisions Log (Sliding Window)

## [2026-07-04] [TASK_L3_MAJORDIAG] [ADR_20260704_L3_MAJORDIAG]
- **Context/Why:** BTC/ETH/BNB 롱 손실 집중(`ADR_20260704_L2L3_PERSYMBOL`) 확인 후, 원인이 신호 지연/사이징 정체/regime cap 미작동 중 무엇인지 미분해 상태였음.
- **Resolution/What:** 매 rebalance마다 워치리스트 3종에 대해 `(raw_mu, w, regime_risk_mult)` 스냅샷 수집(신규 수학 없음). `[L2/L3-MAJOR-DIAG]` 로그로 `mu_bullish_pct`/`stale_long_pct`/`regime_cap_engaged_pct` 5종 비율 노출.
- **Impact:** 실측: `stale_long_pct=0.0%`(전 심볼) → 사이징/no-trade-band 정체 반증. `regime_cap_engaged_pct`=BTC 98.1%/ETH 100.0%(avg_mult≈0.40, 방어 정상 작동) → cap 미작동설도 반증. 반면 `mu_bullish_pct`가 fit/cal 대비(BTC 18.4%→98.1%, ETH 6.4%→100.0%) holdout에서 거의 상시 매수신호로 고착 — regime 분류기는 holdout 70.4%를 bear/crisis로 판정했음에도 BTC/ETH 자체 트렌드 신호는 6개월 내내 거의 항상 롱 유지. 근본원인=포트폴리오 오버레이가 아닌 **BTC/ETH 트렌드 신호의 방향전환 반응속도(reversal-detection lag)**로 확정. BNB는 완만한 상승(12.2%→23.5%)에 그쳐 "고착"이 최대형주 특정 현상임을 시사.

## [2026-07-04] [TASK_L2L3_PERSYMBOL] [ADR_20260704_L2L3_PERSYMBOL]
- **Context/Why:** 롱/숏 aggregate 분해(ADR_20260704_L2L3_LONGSHORT) 이후, 롱 손실이 소수 심볼 집중인지 전체 확산인지 미측정.
- **Resolution/What:** `w_long`/`w_short`를 스칼라로 합치기 전 심볼별 배열로 누적(신규 수학 없음). `[L2/L3-LONGSHORT-TOP]` 로그로 Top-5 Long Losers/Short Winners 노출.
- **Impact:** 실측: L3 롱 손실 상위 2개(ETHUSDT -0.050, BTCUSDT -0.028) 합이 전체 롱 순손실(-0.073)보다 큼 → 나머지 ~49개 심볼은 순플러스, 손실은 BTC/ETH(+BNB)에 집중. `market_regime.py`의 regime 판정이 BTC 가격 자체로 계산되므로, regime을 정의하는 자산에 대한 롱 노출이 그 하락을 직접 맞은 구조로 설명됨 — "전체 롱 계열 문제"가 아니라 "고베타 대형주 롱 노출" 문제로 범위 축소.

## [2026-07-04] [TASK_L2L3_LONGSHORT] [ADR_20260704_L2L3_LONGSHORT]
- **Context/Why:** Regime-mix/ER 진단(ADR_20260704_L3_REGIME) 이후, L3 손실이 롱/숏 어느 쪽에서 왔는지 미측정 상태였음.
- **Resolution/What:** 기존 `_bar_price=dot(w,bar_ret)`를 `w_long`/`w_short` 부호 마스킹으로 선형 분해(신규 수학 없음). `Layer2FoldAttribution`에 `realized_price_long/short`+`bars_long/short` 추가, `[L2-LONGSHORT]`/`[L3-LONGSHORT]` 로그 라인 추가(env 게이트 불필요, always-on).
- **Impact:** 실측: fit/cal(long=+17.4% short=+32.5%, 둘 다 흑자) vs L3 OOS(long=-7.3% short=+4.8%, 롱만 부호 반전). Active Bars는 long=1086/short=1077로 거의 균등 → "롱 편향 노출 시간" 가설은 반증, "균등 노출인데 롱 판단만 붕괴"로 재조준(모멘텀 크래시 패턴).

## [2026-07-04] [TASK_L3_REGIME] [ADR_20260704_L3_REGIME]
- **Context/Why:** L1/L2/L3 5연속 add-on 실패 후, L3 -13.3%가 정말 "구간 성격 불일치(과적합)"인지 근거 없이 추측 중이었음(측정 인프라 부재).
- **Resolution/What:** L3에 `[L3-REGIME]`(bull/bear/crisis%+Kaufman ER), L2에 `[L2-REGIME]`+fold별 ER 컬럼 추가(기존 `Layer2FoldAttribution`/`compute_market_regime_context` 재사용, 신규 수학 없음). `L2_DIAG_ATTR` 미설정 시 ER이 0.000으로 조용히 기본값 반환되는 기존 결함 발견(측정 아님).
- **Impact:** 실측: fit/cal ER=0.213 vs L3 ER=0.218(사실상 동일) → "구간이 유독 횡보"였다는 가설 반증. 대신 regime 비중 이동(bull 36.1%→29.5%, bear 25.9%→35.1%) + 롱 편향 전략풀이 유력 후보로 재조준.

## [2026-07-03] [TASK_L1_XS] [ADR_20260703_L1_XS]
- **Context/Why:** xs_alpha 팩터(xs_momentum/carry/flow/oi_skew)가 factor-level spread 진단에서 견조한 gross 엣지(24/24 fold-variant LCB>0)를 보였으나, per-pair peer-exclusive incremental-edge 게이트가 구조적으로 전량 탈락(`no_incremental_edge`)시킴.
- **Resolution/What:** `XsAdmissionBasis`/`resolve_xs_alpha_admission`로 factor-level 통계를 pair-level gate 입력값에 치환하는 admission 경로 구현, `deployment_evidence` 호출부(양쪽 pipeline.py)에 배선. 기본값 `l1_xs_alpha_admission_enabled=False`.
- **Impact:** 배선 확인 후 실측 economic replay(L1→L2→L3, flag on/off) 결과 승격 36→232건으로 메커니즘은 정상 발동했으나, L3 holdout CAGR -11.3%→-17.7%(Sharpe -0.860→-1.232)로 악화 — L2 fit/cal은 개선(Sharpe 1.50→1.71)돼 과적합 패턴 확인. Default off 유지, 최종 기각.

## [2026-07-03] [TASK_L1_CROWD] [ADR_20260703_L1_CROWD]
- **Context/Why:** Prior L1/L2 crisis-defense mechanisms (reversal-kill, DR concentration gate) failed; crypto crash dispersion is idiosyncratic per-symbol (OI/LSR-driven), not a portfolio-level correlation factor.
- **Resolution/What:** Built per-symbol positioning-crowding dampener (Choueifaty-style persistence mask on OI/LSR z-scores) gating trend sleeve `raw_mu`. Fixed a real shape-mismatch bug (sleeve-dim vs symbol-dim) found during economic replay.
- **Impact:** Economic replay (15-trial, family-filter bypassed) showed CAGR/Sharpe/MDD all slightly worse with gate on. Also surfaced that `_trend_arch_families` matching is broken for BOTH this gate and the pre-existing Trend-Efficiency gate — neither has ever actually fired in production. Default remains off; family-matching fix not yet implemented.

## [2026-07-03] [TASK_L1_DIV] [ADR_20260703_L1]
- **Context/Why:** Extreme trend-beta bias in L1 strategy promotions caused high portfolio concentration risk during crashes.
- **Resolution/What:** Implemented family_admission.py and evaluated non-trend candidates via seed-matched economic replay.
- **Impact:** Replay results showed baseline outperforming treatment (CAGR collapse to -17%), leading to final rejection.

## [2026-07-03] [TASK_L2_DR] [ADR_20260703_L2_DR]
- **Context/Why:** Correlation-aware sizing was absorbed by the L* optimizer, failing to limit leverage during correlation spikes.
- **Resolution/What:** Built Choueifaty-Coignard diversification ratio (DR) haircut gate in leverage calibration step.
- **Impact:** Phase 0 test disconfirmed DR correlation during market crashes, so default was set to False.

## [2026-07-02] [TASK_L2_COV_RE] [ADR_20260702_L2_COV]
- **Context/Why:** Previous correlated covariance mode test was limited to a single reduced trial (n=1, trial=50) due to ledger crashes.
- **Resolution/What:** Re-run diagonal vs correlated covariance A/B testing on full 200-trial after repairing data pipeline bugs.
- **Impact:** Correlated mode underperformed diagonal (CAGR -5.6% vs -5.0%), confirming L* absorption effect.

## [2026-07-02] [TASK_L3_EP] [ADR_20260702_L3_EP]
- **Context/Why:** Whipsaws in post-crash trailing drawdown detection required episode-level timestamps to diagnose.
- **Resolution/What:** Implemented ReversalEpisode extraction logic and stress_gap diagnostics based on half-spread z-score.
- **Impact:** Enables empirical validation of liquidity stress discriminative power for new crash indicators.

## [2026-07-02] [TASK_L3_REPLAY] [ADR_20260702_L3_REPLAY]
- **Context/Why:** Hard verification of crash defense logic was lacking actual historical economic replay in holdout windows.
- **Resolution/What:** Wired risk_off fold attributions to L3 and created run_l3_reversal_economic_replay harness for 8 variants.
- **Impact:** Replay showed baseline outperforming all variants (reversal-kill de-grossed profitable trades), disconfirming entry/exit tuning.

## [2026-07-02] [TASK_UNI_KLINE] [ADR_20260702_UNI_KLINE]
- **Context/Why:** Missing quote_vol index in live kline API and ledger PIT-safe violations caused daily build_universe pipeline deadlocks.
- **Resolution/What:** Fixed binance client to extract quote_asset_volume and replaced static end-date ledger broadcasts with rolling continuity.
- **Impact:** continuity metrics zero-volume count dropped to 0.0, resolving L3 holdout runtime crashes.

## [2026-07-02] [TASK_UNI_VISION] [ADR_20260702_UNI_VISION]
- **Context/Why:** Datetime string parsing errors in Vision metrics downloader caused all open interest and long-short ratio data to be lost.
- **Resolution/What:** Fixed metrics dtype normalization branch and conducted 5-round real data correlation sweep.
- **Impact:** LSR/OI correlation tests fell below significance threshold, deferring raw OI/LSR features from active alpha.

## [2026-07-02] [TASK_L2_SZ] [ADR_20260702_L2_SZ]
- **Context/Why:** Kelly portfolio sizing model assumed zero correlation between active symbols, underestimating concentration risk.
- **Resolution/What:** Added Ledoit-Wolf covariance sizing options and connected portfolio optimizer to active rebalance loops.
- **Impact:** L* leverage scaling absorbed local portfolio sizing offsets, showing no performance improvement.

## [2026-07-01] [TASK_L1_REGIME] [ADR_20260701_L1_REGIME]
- **Context/Why:** Mean reversion strategy (beta_neut) was failing in transition regimes but code had no active regime masking.
- **Resolution/What:** Implemented beta_neut_gating_enabled masking for bull_quiet regime and tested on historical folds.
- **Impact:** Hard masking collapsed symbol-variant sample counts, so regime masking remains off by default.

