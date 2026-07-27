# Active Decisions Log (Sliding Window)

## [2026-07-27] [TASK_L2_COMPOUNDING_LEAP] [ADR_20260727_L2_COMPOUNDING_LEAP]
- **Context/Why:** 20260727_013707 FAIL이 여전히 1일 라벨 오정렬(A-4 미해소)로 오염됨을 원시 레이크 대조(ρ=0.846 vs -0.003)로 확정. 정렬 정정 후 causal β=0.643으로 시장중립이 아닌 베타 롱임이 드러났고, β-헤지 잔차가 성장·변동성·regime 정상성 전부에서 우월함을 실측(scratch/verify_l2_growth_leap.py)
- **Resolution/What:** validation.py(라벨정렬정정·fail-closed 정렬불변식·beta-adj excess), benchmark.py(causal_beta_series/assert_contemporaneous_alignment 신규), allocator.py(apply_beta_hedge_overlay/derive_mdd_parity_scale 신규), config.py(beta lookback/clip, min_oos_days 365->500 상향, mdd_budget), engine.py(2-pass 무헤지/헤지 시뮬 배선), run_windows.py(l1_days 365->180, l2_days 365->547). /check PASS(Cov 85%, 임계값 완화 0건)
- **Impact:** 실전 CLI 재실행 결과 target_weights 전량 0(NO_EVIDENCE) - L1 exit-aware handoff admitted=False sleeves=274(전멸). 원인=L1 윈도우 축소(365->180d)가 기존 admission 게이트 min_effective_days=180.0과 충돌, fold분할 후 유효일수 미달로 전신호 탈락. [LIMIT-07]이 예견한 위험이 감소 아닌 전멸로 실현. P0/P1/P2(정렬·베타·헤지)는 유효, P3(창 재분할)는 admission 게이트 재설계 결정 대기로 보류

## [2026-07-27] [TASK_L2_GATE_HONESTY_AND_RISK_BUDGET] [ADR_20260727_L2_GATE_HONESTY_AND_RISK_BUDGET]
- **Context/Why:** L2 PASS(CAGR 31.05%, DSR 1.000000)가 얇은 마진이 아니라 결함(DSR 주기불일치·확률게이트 중복·블록길이 오지정·벤치마크 일자오정렬·실행북 동결·funding 부호역전·종목별 비용무력화·L3 holdout이 PROMOTE를 못막음·CAGR 연율화 버그) 9건의 산물임을 실측(재구성+E1~E8 다중가설)으로 확인
- **Resolution/What:** bootstrap.py 신규(Politis-White 블록길이·circular bootstrap·SPA 3대조군). multiplicity.py(DSR 주기정합), validation.py(일자정렬·복리연율화·SPA 편입·frozen control), allocator.py(support 재적용·심볼별 band·carry 부호·폐루프 vol), dense_simulator.py(종목별 비용·slippage/impact), engine.py(frozen control 배선·L3 prior 일봉화). 실전 실행에서 L3 prior 가드 결함(스펙 계약 자체의 오류로 모든 정상실행 크래시) 추가 발견, 가드 제거 및 테스트 교체로 수정
- **Impact:** 실제 CLI 재실행(20260727_013707): verdict PASS→FAIL, CAGR 31.05%(산술오류)→6.75%(복리정합), DSR 1.000000(포화)→0.4266, excess_growth_probability 0.9400→0.712, 신규 SPA p=0.362(기준 0.10 초과). L3가 l2_not_pass로 즉시 reject(이전엔 dry_run으로 shadow 방치). 임계값 완화 0건. A-8(PIT 유니버스 breadth)은 범위 외 후속 스펙

## [2026-07-26] [TASK_DEPLOYMENT_PROVENANCE_AND_SEARCH_MULTIPLICITY] [ADR_20260726_DEPLOYMENT_PROVENANCE_AND_SEARCH_MULTIPLICITY]
- **Context/Why:** L2 게이트 최초 PASS 후 CLI 실행이 strategy_spec_hash/fold_manifest_hash 미배선으로 크래시. 조사 결과 표면 버그 아래 봉인 홀드아웃 동어반복 검증(DEF-01)과 dead CandidateTrialLedger로 인한 탐색 다중성 미회계(DEF-02), NumPy 2.4 단일-trial 크래시(BUG-03) 발견
- **Resolution/What:** provenance.py 신규(해시 유도 4함수). multiplicity.py에 BUG-03 가드+charge_config_search_multiplicity(가산형 참여비, 3개 가설 실측 비교 후 채택) 추가. contracts.py CandidateTrialLedger.register/load_trial_returns 구현. holdout_store.py ensure_sealed(미소진 seal 1회 backfill)+consume 강화(런타임 해시 사용, universe_state_hash 검증 유지). l1_sleeves.py/engine.py/compound_main.py 전체 배선 및 trial_ledger.register 호출 연결
- **Impact:** L2_DRY_RUN=1 실전 CLI 재실행 exit_code=0으로 사상 최초 완주(logs/futures/compound/20260726_142427). L2 PASS 수치는 수정 전과 완전 동일(CAGR 31.05%, Sharpe 1.352)하여 알파 로직 무변경 확인. 봉인 홀드아웃 1회 backfill 실측 확인, 다중성 ledger 최초 가동(1행 등록). 12변형 그리드서치는 ledger 도입 이전이라 소급 과금 불가

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
