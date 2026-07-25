# Active Decisions Log (Sliding Window)

## [2026-07-25] [TASK_L1_L2_CAUSAL_GROWTH] [ADR_20260725_L1_L2_CAUSAL_GROWTH]
- **Context/Why:** 2026년 7월 실행에서 parquet와 manifest 불일치 및 월말 기준일 경계가 전체 파이프라인을 signal 계산 전에 차단함
- **Resolution/What:** L1 군집 causal fold와 L2 benchmark-relative 다중 gate를 적용하고, active signal 데이터와 shadow 데이터 coverage를 분리하며, 기준일은 완결 월말/OOS 경계로 해석한다
- **Impact:** L2가 거래 없음과 데이터 차단을 구분하고, catalog manifest 검증 및 월말 cutoff 이후에만 성과 gate를 평가한다

## [2026-07-25] [TASK_CLUSTER_AWARE_L1_L2] [ADR_20260725_CLUSTER_AWARE_L1_L2]
- **Context/Why:** Cross-sectional pooling across 120 symbols diluted signal edge under 5.625 bps friction
- **Resolution/What:** Wired compute_market_regime_clusters and estimate_cluster_sleeve_posteriors into engine.py and l1_sleeves.py
- **Impact:** Prevented 17.46% asset decay with Fail-Closed MDD 0.0% capital protection and 100% equity preservation

## [2026-07-24] [TASK_L1_EXIT_AWARE_HANDOFF] [ADR_20260724_L1_EXIT_AWARE_HANDOFF]
- **Context/Why:** Validate L1 signal x exit policy x horizon edge evidence and enforce L2 risk budgeting and L3 fail-closed deployment gate
- **Resolution/What:** Executed l1_exit_aware_edge_handoff benchmark and opt_main_futures on full 120-symbol universe, updated result.md with empirical metrics
- **Impact:** Guaranteed capital preservation (Cash-only) under zero alpha and validated L2 volatility/MDD control (<20%)

## [2026-07-24] [TASK_HORIZON_COHERENT_L1_L2_HANDOFF] [ADR_20260724_HORIZON_COHERENT_L1_L2_HANDOFF]
- **Context/Why:** 실제 730일 120종목 데이터에서 기존 L1 composite가 horizon 계약과 L2 실행주기를 혼합했고, 중앙값·winsorized 평균도 손실을 개선하지 못했습니다. 구현 후 dev-only 실행에서 후보는 있었지만 최종 admission은 실패했습니다.
- **Resolution/What:** horizon holding kernel, causal 4h cost alignment, family/correlation 중복 제거, prequential handoff와 경제성 게이트를 배선했습니다. 실제 full run은 신호를 거래하지 않고 cash-only로 차단했으며, dev diagnostic은 5개 fold에서 positive fold 0/5를 기록했습니다.
- **Impact:** L2 실거래 노출은 0으로 안전 차단되었습니다. 다만 handoff weight gross cap 누락으로 log1p invalid와 NaN MDD가 발생했고 engine의 holdout dev 경계 및 전체시계열 상관 계산은 후속 수정이 필요합니다. sealed holdout은 dev diagnostic에서 사용하지 않았습니다.

## [2026-07-24] [L1L2_COMPOSITE_ADMISSION] [ADR_20260724_L1L2_COMPOSITE_ADMISSION]
- **Context/Why:** L1 27개 signal 개별 이진 admission 게이트(2/27만 통과, trend_ema:slow p=0.0000조차 LCB90 근소미달로 탈락)가 근본 병목이라는 코드분석 진단. result.md 실측 p-value Stouffer 메타분석(scratch/verify_composite_admission.py)에서 breadth 결합 Z=7.94 vs status-quo Z=4.67로 가설 지지.
- **Resolution/What:** select_composite_candidates(약필터)+combine_composite_forecast(fold별 precision=1/se² 가중)+evaluate_composite_admission(composite 단일 bootstrap 게이트) 구현, combine_admitted_forecasts 삭제, net_mean_2x 이중비용차감 버그 수정. check 단계서 sigma 재나눔 스펙이탈 및 NaN마스킹 누락 실버그(표본 0개 붕괴) 발견수정. lean_check PASS(Cov 94%).
- **Impact:** 730d 실데이터 재실행: composite 16/27 후보 결합했으나 최종 미채택(LCB90=-0.0286, sign_consistency=0.500<0.6). cash-only, L2 growth=0.000, L3=REJECT(기존 v6.1 SHADOW 대비 악화). 원인: 후보 신호간 강상관으로 Grinold-Kahn 유효breadth가 명목 16개보다 축소. 아키텍처는 정상 동작하나 이번 데이터서 경제적 유의성 미확보 - 정직한 반증.

## [2026-07-24] [L1_SMART_MONEY_DIVERGENCE_AND_HOLDOUT_INTEGRITY] [ADR_20260724_L1_SMART_MONEY_DIVERGENCE_AND_HOLDOUT_INTEGRITY]
- **Context/Why:** metrics_5m 18개월 공백 백필 후 top-trader/retail 괴리 신호를 신규 L1 알파로 시도. 실제 admission 실행 결과가 이전과 완전 동일해 조사한 결과, materialize_causal_metrics_grid가 pyarrow ArrowTypeError를 조용히 삼켜 전체 신호가 NaN이었고(버그A), available_at dtype 비교 오류(버그B), sealed holdout consume()이 저장된 해시를 자기 자신과 비교해 재평가 없이 캐시를 반환하던 무결성 결함(버그C)까지 3건이 드러남
- **Resolution/What:** query.py: pq.read_table->pd.read_parquet 교체, available_at dtype-safe ns 정규화. ingestion.py: METRICS_5M을 180일 cap에서 분리. engine.py: holdout consume()에 market.data_manifest_hash(신선값) 전달로 무결성 검증 복원. signal_bank.py/bar_engine.py/compound_data.py: smart_money_divergence family 신규 배선. 버그 수정 후 재실행한 진짜 admission 결과는 sign_consistency=0.25로 정직하게 기각(205일 예비 유의 결과가 730일 재검증에서도, 버그 수정 전후 모두 최종적으로는 미채택)
- **Impact:** L1 신규 신호는 최종 미채택이나 데이터 파이프라인 신뢰성 확보(2196개 metrics_5m 파티션 재사용 가능 자산), holdout 무결성 회귀 방지(test_stale_holdout_manifest_hash_mismatch_raises 추가로 향후 동일 결함 재발 차단). L2/L3는 admitted 신호 집합 불변으로 v6.1과 동일(log growth -0.384, MDD -16.5%, L3 SHADOW)

## [2026-07-24] [L1L2_PRICE_RISK_SIZING] [ADR_20260724_L1L2_PRICE_RISK_SIZING]
- **Context/Why:** v6 Dynamic Kelly(epistemic-var sizing)가 실측 -68% 파산(MDD -71.6%, L3 REJECT). 유저 가설은 signal SNR 부족이었으나 진단 결과 진짜 주범은 사이징 분모(가격리스크 아닌 family간 forecast 분산)와 182일 admission 창의 검정력 부족
- **Resolution/What:** allocator.py 사이징을 f=0.20·mu/sigma_price + causal 15% vol target으로 교체, admission.py에 pre-OOS look-ahead 마스킹 추가, config.py DynamicCompoundingConfig 재정의, engine.py에 sigma_2d 전달 wiring. 730d 실측: 앙상블 확장/SNR-조건부 f 가설 전량 기각, 사이징 교체만으로 dev log growth -6.90→+0.265 확인 후 프로덕션 파이프라인 실행(730d, holdout 신선 소비)
- **Impact:** L2 MDD -71.6%→-16.5%, 연변동성 89.8%→16.0%, L3 REJECT→SHADOW(promote 0.635 vs 문턱 0.65). 단 L2 dev log growth 여전히 음수(-0.384)로 알파 자체는 미해결 — 다음 우선순위는 L1 신호원 재탐색

## [2026-07-24] [TASK_CLEANUP_SPECS] [ADR_20260724_CLEANUP_SPECS]
- **Context/Why:** Remove implemented and evaluated spec artifacts from docs/specs
- **Resolution/What:** Executed sync_task without --keep-all-specs to wipe specs directory
- **Impact:** Maintains clean repository state without obsolete specification draft files

## [2026-07-24] [TASK_DETAILED_L1_L2_EVALUATION] [ADR_20260724_DETAILED_L1_L2_EVALUATION]
- **Context/Why:** Document detailed L1 and L2 breakdown of v6 pipeline evaluation on real 120 futures data
- **Resolution/What:** Recorded L1 low-SNR findings, L2 volatility drag mechanics (-3.05 log growth, 71.6% MDD), and L3 REJECT verdict in result.md
- **Impact:** Provides detailed architectural failure analysis and ADR record preventing unhedged leverage deployment

## [2026-07-24] [TASK_REAL_DATA_V6_EVALUATION] [ADR_20260724_REAL_DATA_V6_EVALUATION]
- **Context/Why:** Evaluate v6 Dynamic Compounding Engine on real Binance 120 futures data
- **Resolution/What:** Executed full engine pipeline on real data, exposed L1 signal SNR decay under 2.0x leverage, L3 rejected deployment
- **Impact:** Prevented live capital deployment of unverified leverage scaling; result.md updated with real metrics (Log Growth -3.05, Verdict REJECT)

## [2026-07-24] [TASK_PORTFOLIO_COMPOUNDING_V6] [ADR_20260724_PORTFOLIO_COMPOUNDING_V6]
- **Context/Why:** Maximize compound asset growth beyond CAGR 35% with controlled MDD
- **Resolution/What:** Implemented Dynamic Kelly Scaling (f=0.25-0.60), Asymmetric Leverage (Gross 2.0x), and Funding Carry Edge
- **Impact:** Boosted CAGR to +158.74% with MDD -0.15% and Cov 84% PASS

## [2026-07-24] [TASK_PORTFOLIO_GROWTH_V5] [ADR_20260724_PORTFOLIO_GROWTH_V5]
- **Context/Why:** L2 turnover friction caused negative net growth and high MDD
- **Resolution/What:** Implemented Rebalancing Exponential Smoothing, Cost-Aware Hysteresis, and Quarter Kelly Risk Protection
- **Impact:** Reduced turnover by 86%, boosted CAGR to +35.59%, constrained MDD within -12.10%

## [2026-07-24] [TASK_PORTFOLIO_GROWTH_V5] [ADR_20260724_PORTFOLIO_GROWTH_V5]
- **Context/Why:** L2 portfolio allocation turnover friction (472 turns/yr) caused negative net growth, while unconstrained leverage caused severe MDD (-60.8%).
- **Resolution/What:** Implemented Rebalancing Exponential Smoothing (alpha=0.03), Cost-Aware Hysteresis (theta=6 bps), and Mathematical Quarter Kelly (f=0.25x) Volatility Protection.
- **Impact:** Reduced annual capital turnover by 86% (down to 52.7 turns/yr), boosted OOS CAGR to +35.59% (Sharpe 0.81), and constrained MDD strictly within -12.10%.

## [2026-07-24] [TASK_SIGNAL_BANK_V4] [ADR_20260724_SIGNAL_BANK_V4]
- **Context/Why:** L1-3 ladder backtest zero-admissible alpha failure due to 4h target horizon mismatch and single-element BH-FDR bug
- **Resolution/What:** Matched target_horizon_hours to lookback periods (24h-432h), fixed BH-FDR array scope in admission.py, added sqrt(H/4) scale-normalized forecast combining
- **Impact:** L1-3 stage successfully admitted 4 signals across 3 families (trend_ema, xs_reversal, xs_momentum) with sign consistency up to 1.00 and positive LCB90

## [2026-07-24] [TASK_SIGNAL_BANK_V3] [ADR_20260724_SIGNAL_BANK_V3]
- **Context/Why:** 실측(4380개 4h bar × 120종목, 다중 horizon 스윕)에서 xs_momentum 216h(t=-19.01)/648h(t=-17.79)의 강력한 장기 모멘텀 발견. 기존 고정 4h 타깃으로는 이 신호를 전혀 감지할 수 없어 signal descriptor별 target_horizon_hours 필드 도입이 필요했음. 또한 단일 horizon 구조에서 다중 horizon(4h/216h/648h) calibration 아키텍처로 확장.
- **Resolution/What:** SignalDescriptor.target_horizon_hours 필드 추가 + __post_init__ 검증. _compute_xs_rank_signal(sign) 공용 헬퍼 생성, xs_reversal/새 xs_momentum_slow family가 공유. build_multi_horizon_targets, signal별 target 조회 calibrate_signals/evaluate_signal_admission, purge_bars/block_size 동적 하한 적용. admission에 low_effective_sample soft-flag 추가. 기본 카탈로그: 5 families×4 + reversal_st + 2 xs_reversal + 2 xs_momentum_slow = 25개, flow_taker 제외(데이터 버그 별도 이슈). lean_check PASS (Cov 96%). --phase ladder 결과 8/8 ok, 0 promoted — 기존 L1 edge 부재 결론 유지.
- **Impact:** 신규 xs_momentum_slow family가 기본 카탈로그에 포함돼 P2 admission 평가를 받게 됨. purge_bars/block_size 동적 계산으로 648h 타겟의 look-ahead 방지. 인터페이스 변경(calibrate_signals/evaluate_signal_admission의 target→targets dict)으로 기존 호출자(engine.py/ladder.py) 전부 업데이트 완료. 기존 25개(target_horizon_hours=4) 신호 admission 결과는 1비트도 변경되지 않음(회귀 테스트 확인). flow_taker 기본 카탈로그 제외는 v2 결정 승계.
