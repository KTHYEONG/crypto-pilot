---
title: Layer 1 Decision Log (Compressed)
domain: futures.strategy
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## Phase 1: SWF 구조 & 초기 게이트 (ADR-001~009, 6/13~19)
- Nested SWF 도입, prequential evidence grid 분리(outer_n×multiplier≤max), outer warm-up blocks=2로 fold 0 underpower 해소
- 통계적 MDES gate(t_crit+검정력 80%), 5-Gate로 standardization(fold_cov/match_ratio/sym_count/fold_ratio/probe_lcb_bps)
- IC 지표 제거(Arch-Only mode에서 noise), mu_quality_shrinkage dead-code 제거(validation_rank_ic=0 → lam=0 붕괴)
- PIT state_cube 통합(lifecycle gate: promotion_available_at>l2_start 차단), capacity clip(5 USDT min+proportional)
- L1→L2 게이트 재설계: 4-condition(LCB 하드게이트 worst-plausible edge>RT cost, binding FDR, qw>0), Q:hi/mid/lo 티어 제거
- Promotion Summary PASS/FAIL 분리, 우측절단 진단(censored count), STATUS PROMOTED/WATCH/REJECTED 제거
- Archetype 라벨 정합(5→7종, MRV/TMO/UNW/BTN/CRY/FLO), evidence fold warm-up skip WARNING 억제

## Phase 2: 성능 최적화 1~3차 (ADR-009~016, 025~026, 6/18~21)
- PERF 로깅 도입(레벨 15→10 통일, 계층적 타이밍, [PERF] prefix 일원화)
- Numba JIT rolling z-score(prime 27.78→7.75s, L1 total 47.64→25.17s)
- q-value FDR vectorization→loop 롤백(N≤200 소표본 회귀)
- Python loop 제거+벡터화(_by_q_values, _compute_incremental_bps, .copy() 제거, itertuples→cumcount)
- Ingestion 병렬: ProcessPool→ThreadPool(I/O bound pickling 제거), meta pre-conversion, datetime64 bypass
- Parquet I/O: baggage 3col drop, numeric early exit guard(>O(N) 스킵), 불필요 .copy() 제거
- Post-SKIP: copy() 조건부 게이트, _to_unix_ms TF당 1회, merged audit 경량화, COL_GROUP_CACHE 도입
- Numba membership(_calculate_warm_ready_numba) + DatetimeIndex.isin 벡터화
- L1/Universe 루프 불변식 호이스팅(searchsorted N→1, timeline 104→1), vol_2d 행렬화(pd.DataFrame.rolling())
- signal_batch_convert iterrows 벡터화(iterrows→np.flatnonzero, loop 2.64→0.04s/fold, L1 total 53.5→37.4s)
- P1~P5: prime→_warm_aligned_2d_cache 분리, inner groupby pre-compute, waterfall 게이팅, PERF 타이머 세분화, audit_tables 타이머 이동

## Phase 3: 신호 패밀리 & MTF 확장 (ADR-017~024, 026~034, 6/21~22)
- Flow family 3종(funding_flow_carry/unwind, flow_exhaustion_reversal) + cell-level taker_imbalance_2d
- 8 저성과 family 제거(trend_donchian, OI 5종, basis 2종, taker_exhaustion) 40→31, per-symbol ENS-DIAG 진단
- FLO 회귀 수정: flow_trend_continuation archetype flow_rev→ts_mom, lsr_oi_regime_filter active화(side_hint 방향성)
- MTF 람다 late-binding 수정(기본 파라미터 캡처), flow 캐시 재사용(funding_z_96/168)
- CRY/FLO 안정화 3종 추가(funding_term_structure_carry, flow_trend_continuation, lsr_oi_regime_filter), positioning_warm[:168] barrier
- Bayesian prior ENS(n_prior=100, prior_mean=0로 소표본 수축), min_display_events로 insuf 표시
- Adaptive evidence gate(early snapshot threshold 완화, snapshot_index 파라미터), promotion advisory 모드
- TF별 signal pool 6종 추가(gap_fade_1h, vwap_reversion_1h, volume_climax_1h, macd_4h, supertrend, ichimoku_trend)
- Per-TF L1 pipeline: run_per_tf_l1/_resolve_l2_master_tf/_aggregate_per_tf_l1, native_tf 필터링, oos_stacked tf:: prefix
- Quality weight floor(l1_qw_floor=0.05)+probe prior boost(l1_qw_probe_boost=0.3), per-TF gate overrides
- LTF mean aggregation(ltf_mode="mean" cumsum window), resample metadata 보존, holding_bars cost 보정
- TF Probe virtual source resample(data-stage/bridge-stage 분리), ENABLE_TF_PROBE=False 기본
- Effective-N FDR 보정(diversity_corr 기반 m_eff), 동일(symbol,family) cluster r_bar

## Phase 4: 후반 최적화 v2~v3 (ADR-035~037, 6/22~23)
- L1 Gate+Signal Pool Optimization: per_TF_gate_overrides 자동 fallback, fdr_alpha 0.10→0.15, qw_floor 0.05, 2h trend_ma 제거
- OOM 방지: resolve_safe_nested_workers adaptive cap(max_workers=min(cpu_limit-2,8), oversubscription guard), fork 내 gc.disable()
- P5-R: prequential ThreadPoolExecutor 제거→순차 복원(GIL+cache thrashing 역효과, 11.4→7.9s/TF, -31%)
- PERF-GAP1: [WORKER-CALC]→[PERF] worker_calc(memory 추정+worker decision 노출)
- PERF-GAP2: [PERF] l1_nested_ipc_collect(IPC+fold compute vs pool_setup 분리)
- v3 실측: L1 108.6s(-52% vs baseline), Grand Total 152.8s(-49%), EXIT_CODE=0
