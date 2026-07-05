# Active Decisions Log (Sliding Window)

## [2026-07-05] [TASK_CHAMPION_REPRODUCIBILITY_AND_REGISTRY_CENSUS] [ADR_20260705_CHAMPION_REPRODUCIBILITY_AND_REGISTRY_CENSUS]
- **Context/Why:** Track2 census 항상 0(TF선택 순서 버그 의심) + Track1 dampener 판정(BLOCK)이 재현되는지 미검증 상태.
- **Resolution/What:** `awf_sim.py`의 `compute_major_symbol_registry_census` isinstance 체크가 `signals.contracts`(잘못된 중복 클래스)를 참조하던 버그 수정(`candidate_contracts`로 교정) + 관련 mock 테스트 2건 동시 수정. 격리된 Optuna storage로 seed=42 200-trial replay 2회 독립 재현 실험.
- **Impact:** registry_census_count 0→6(첫 실측: BTC/ETH 정확히 어떤 family가 hard_eligible/observed인지 확인). 재현 실험 결과 두 실행이 부동소수점 잡음 수준까지 완전 일치(PASS, trades=273) — 파이프라인 비결정성 가설 반증. 저장된 200-trial CSV(BLOCK)와의 차이는 실행 비결정성이 아니라 **공유 Optuna study가 세션 간 누적되며 다른 챔피언에 수렴**했기 때문으로 확정. 다른 기각된 economic replay ADR들도 동일 재검증 필요성 있음(후속 조사 대상).

## [2026-07-05] [TASK_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] [ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC]
- **Context/Why:** spec/apply안 실측을 seed-matched replay로 고정해야 했고, `env` 후주입 A/B는 champion selection과 final config를 분리해 정본 측정이 아니었음.
- **Resolution/What:** `MAJOR_SYMBOL_REGISTRY_REPLAY=1` 내부 harness와 `--seed` SSOT를 배선하고, `run_tiered_pipeline`이 L2 직후 baseline/treatment replay CSV를 생성하도록 연결.
- **Impact:** 200-trial, seeds `42/123/7` replay 데이터 확보 후 adoption gate는 `below_median_total_return_delta`로 BLOCK; L3 개선/registry census 실측은 미발생.

## [2026-07-05] [TASK_L1_DIVERGENCE_DAMPENER] [ADR_20260705_L1_DIVERGENCE_DAMPENER]
- **Context/Why:** Phase 0 실측(ADR_20260705_L1_MAJOR_REVERSAL_ALPHA)이 BTC(outvoting)/ETH(반대신호 부재)로 갈렸음. Boost-only 설계는 실측 magnitude 격차(16배)로 수학적 기각 — dampener 병행 필요, ETH는 fix 전 admission/activation-gap 선행 진단 필요.
- **Resolution/What:** Track1: `IntraSymbolDivergenceState` 상태기계(기존 veto 패턴 재사용)로 dominant(`dual_momentum`) `raw_mu` 감쇠 + dissent(`ichimoku_trend`) `quality_weight` 부스트(안전상한 clip), `_combine_sleeve_signals_to_symbol` 직전 적용. Track2: `compute_major_symbol_registry_census`로 L1 registry vs holdout 관측 대조. `L2_INTRA_SYMBOL_DIVERGENCE` env A/B 하네스 신규 추가.
- **Impact:** 실측(A/B): BTC mu_bull 98.3%→61.1%, L3 CAGR -17.1%→-12.2%, MDD 26.8%→22.4%, trades 214→273(붕괴 없음) — breakeven 미달이나 유의미한 손실 축소 확인. Track2는 `_aggregate_per_tf_l1`이 멀티-TF 병합 시 `deployment_registry`를 보존 안 해 표준 런에서 미발화하는 별도 인프라 갭 발견(후속 이슈). Check 단계에서 `_regime_now` UnboundLocalError(l2_routing_mode="pool" 시) 발견·수정 완료.

## [2026-07-05] [TASK_L1_MAJOR_REVERSAL_ALPHA] [ADR_20260705_L1_MAJOR_REVERSAL_ALPHA]
- **Context/Why:** Risk-overlay 트랙(veto/cap/kill-switch) 전부 손실 완화 천장 확인(`ADR_20260705_L2_VETO_REPLAY_PARITY` 최선도 L3 total_return -5.1%). 근본원인(BTC/ETH reversal-detection lag)을 L1 sleeve-pooling 단계에서 outvoting(가설 A) vs 반대신호 부재(가설 B)로 분해 필요.
- **Resolution/What:** `_combine_sleeve_signals_to_symbol` 직후 major 심볼(BTC/ETH/BNB) family별 `raw_mu`/`quality_weight`/풀링후 부호를 스냅샷(`MajorSymbolSleeveContributionSnapshot`), `summarize_major_symbol_sleeve_contribution`로 (symbol,family)별 sign-mismatch 비율 집계, `[L2/L3-MAJOR-SLEEVE-DIAG]` 로그 배선(신규 수학 없음, 로그 전용).
- **Impact:** 실측 결과 원 가설(코드 조사 기반 `trend_ma` 지목)은 부분 반증 — BTC는 가설 A 확정이나 범인은 `dual_momentum`(mu+3.678,qw=1.0)이 `ichimoku_trend`(mu-0.222, adverse_mismatch=63.3%)를 magnitude로 압살하는 구조. ETH는 가설 B(holdout 활성 2개 family 전부 대형양수, mismatch=0%, 반대신호 자체 부재). `trend_ma`는 fit/cal(BTC)에만 존재하고 holdout엔 미등장 — 다음 단계는 심볼별로 분기(BTC: contrarian 가중부스트, ETH: L1 admission/selection 재조사).

## [2026-07-05] [TASK_L2_VETO_REPLAY_PARITY] [ADR_20260705_L2_VETO_REPLAY_PARITY]
- **Context/Why:** Contextual veto replay(`ADR_20260704_L2_CONTEXTUAL_DIRECTIONAL_VETO`)의 baseline_parity=False로 adoption 판단 불가 상태. 코드 추적 결과 replay가 `prebuilt_cache`/`eval_memo` 없이 L2 캐시를 즉석 재빌드해 메인 L2(CAGR 58.2%)를 재현 못하고 24.2%를 냄.
- **Resolution/What:** `run_directional_veto_economic_replay`에 `prebuilt_cache`/`eval_memo` 배선(5-arm 전체 공유, cache는 config-independent라 안전), `_baseline_parity`를 검증된 `assert_selection_replay_parity`(L2 leg) + 기존 `cagr` 비교(L3 leg)로 교체.
- **Impact:** 재실행 결과 baseline_parity=True 전 행 확정(replay baseline CAGR 58.19%=메인 일치). L3 수치는 버그 전후 불변(L3는 원래 원인 아니었음 확인). 단 올바른 baseline 기준 fit-cost 재계산 결과 `contextual_cap_mu/zero_mu`가 `fit_cagr_degradation`(1.65%p>0.5%p 예산)로 adoption 탈락, 유일한 adoption 통과 후보는 `contextual_crisis_only`(fit cost≈0, L3 total_return -5.1%, 여전히 <0).

## [2026-07-04] [TASK_L2_CONTEXTUAL_DIRECTIONAL_VETO] [ADR_20260704_L2_CONTEXTUAL_DIRECTIONAL_VETO]
- **Context/Why:** 기존 adverse-only veto가 BTC/ETH holdout long 고착을 56.2% 개선했으나 단순 binary 차단으로 과잉 차단 우려. Regime 상태를 persistence+loss trigger로 단계적 관리해야 fit CAGR 보존 + 손실 감소를 동시에 달성 가능.
- **Resolution/What:** `Layer2AllocationConfig`에 contextual 모드(11개 knob) 추가, `_compute_contextual_directional_veto_signal` 상태기계(idle→watch→armed→veto→cooldown), `_compute_symbol_rolling_return` causal window 구현. Replay 5-arm(`baseline`/`veto_adverse_only`/`contextual_cap_mu`/`contextual_zero_mu`/`contextual_crisis_only`), adoption gate fit-CAGR/total-return/long-loss 조건 강화.
- **Impact:** L3 CAGR -17.1%→-3.3%(contextual_cap_mu, +13.7%p), MDD 26.8%→17.0%. Loss reduction 80.8%. 단 baseline_parity=False로 adoption gate 불신 → 메인 L2/L3와 동일 config/leverage parity 선행 필요.

## [2026-07-04] [TASK_L2_DIRECTIONAL_VETO] [ADR_20260704_L2_DIRECTIONAL_VETO]
- **Context/Why:** BTC/ETH holdout에서만 long 고착이 재현되고 BNB는 control로 정상이라, regime adverse 구간의 major long만 causal neutral 처리하는 개입이 필요했음.
- **Resolution/What:** `Layer2AllocationConfig`에 directional veto flag/symbols/adverse codes/action/budget knobs를 추가하고, `awf_sim` snapshot/summarize + `pipeline` 2-arm replay/adoption gate + `tiered_logging` render 경로를 배선했음.
- **Impact:** holdout CAGR -17.1%→-7.5%, MDD 26.8%→18.2%로 개선됐지만 fit/cal net veto value가 음수여서 기본 채택은 거절됐음.

## [2026-07-04] [TASK_L3_INCOHERENCE] [ADR_20260704_L3_INCOHERENCE]
- **Context/Why:** `ADR_20260704_L3_MAJORDIAG`로 BTC/ETH 신호 고착(mu_bullish 98~100%) 확인 후, 원인이 "앙상블이 구조적으로 느리다"는 가설 vs "holdout 구간 특이성"인지 미분해 상태였음. fit/cal과 holdout의 regime 분포는 유사(bear+crisis 63.9% vs 70.4%)해 regime 자체 차이는 아님.
- **Resolution/What:** 동일 `major_symbol_snapshots`에서 fold-boundary-safe 스캔으로 `regime_adverse_mu_bullish_pct`(불일치율) + `mean_reversal_lag_bars`(전환속도) + `censored_pct`(미전환율) 집계. `MajorSymbolIncoherenceSummary` dataclass + `summarize_major_symbol_regime_incoherence` 함수 추가. `[L2/L3-MAJOR-INCOHERENCE]` 로그 라인 배선.
- **Impact:** 실측 결과 fit/cal에서는 BTC/ETH 모두 adverse regime에서 즉시 반응(lag 0.0~0.9bar, censored 0%) → "앙상블이 구조적으로 느리다"는 원래 가설은 반증. Holdout에서만 BTC/ETH가 144bar/영구 고착 → 근본 원인은 "대형주+holdout 구간 조합"의 가격 패턴 질적 변화(grind-up이 breakout 신호를 계속 재진입시키면서 regime은 변동성 급등만으로 crisis 트리거). Phase 2 veto gate 설계는 유효하나 false-positive 발동률 측정이 스펙에 추가되어야 함. [ADR_20260704_L2_META_PARSER]
- **Context/Why:** Regime 분류기 성능 재검토 중 `_parse_meta_group_ids`가 정본 `"{family}:{variant}"` 콜론 포맷을 무시하고 슬라이스해 family가 variant까지 포함하는 버그를 발견. `L2_POSITIONING_CROWDING_GATE`/`L2_TREND_EFFICIENCY_GATE`(둘 다 `_trend_arch_families` set-membership 사용) 및 `l2_routing_mode="bucket"`(기본값) 버킷 라우팅의 family-level shrinkage/pooling 전부에 영향.
- **Resolution/What:** 콜론으로 family/variant 우선 분리 후 variant에서만 tf 접미사(`_{N}h`) 추출, 콜론 없는 legacy 포맷은 구 로직으로 폴백(회귀 없음, 실행 검증 완료). 두 게이트 기본값은 계속 off 유지(경제적 효과는 별도 replay 필요).
- **Impact:** 동일 설정(`--phase l3 --trials 200`) 재실행 결과 L3 CAGR -11.3%→-11.3%(동일), `[L3-MAJOR-DIAG]` BTC/ETH 수치 완전 동일 — **버킷 라우팅 버그는 L3 홀드아웃 손실의 원인이 아니었음을 확인**. `ADR_20260625_L2_ROUTING`(Stage A GO)와 독립적인 별개 결함. 근본 병목은 `ADR_20260704_L3_MAJORDIAG`의 BTC/ETH 트렌드 신호 방향전환 지연으로 재확정.

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
