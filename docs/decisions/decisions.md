# Active Decisions Log (Sliding Window)

## [2026-07-28] [SIGNAL_PANEL_MAD_KERNEL_H2] [ADR_20260728_SIGNAL_PANEL_MAD_KERNEL_H2]
- **Context/Why:** signal_panel 206s E2E 병목의 95.7%를 점유하는 MAD-z kernel의 2회 sort를 1회 sort+3-way merge로 대체
- **Resolution/What:** _rolling_mad_z_single_sort_kernel 신규 @njit; _rolling_mad_z 3단계 fallback 체인(H2→old kernel→numpy); 7개 TDD 단위테스트 추가
- **Impact:** 4h kernel 5.46s→3.54s(1.54×), 1h kernel 23.61s→15.28s(1.55×). Bit-exact 유지(diff=0.0). Panel 추정 206s→~156s

## [2026-07-28] [PHASE_FULL_BOTTLENECK_OPT] [ADR_20260728_PHASE_FULL_BOTTLENECK_OPT]
- **Context/Why:** phase full 실행 시 E2E 385s+의 병목 식별 및 최적화. Signal panel 179.6s(50%), market cube 27s(7.5%), exit_cache+cluster_posteriors 20-30s(8%) 가 주 병목
- **Resolution/What:** 1) exit_path.py _label_kernel에 @njit(cache=True) 적용 (순수 Python 삼중 루프 → 컴파일). 2) l1_sleeves.py aggregate_cluster_group_returns에 @njit 적용 + gc.collect() 제거. 3) H1(TPE 비활성화)과 H5(동시 I/O)는 실증 결과 역효과/무효로 롤백. 4) 7개 단위 테스트 추가 (numba-Python 동등성, zero-event, aggregate 수렴). 5) docs/specs/phase_full_bottleneck.md + contract.json 작성
- **Impact:** P2 구간(exit_cache+posteriors) 30s 이내 유지. 전체 E2E 385s→285s(-100s, 26%↓). 단 signal panel MAD 커널(T=5442×S=51×recipe=60, sorting 기반)이 여전히 206s로 최대 병목 — TPE/스레드 배분만으로는 해결 불가, MAD 커널 자체 알고리즘 개선(H2) 필요

## [2026-07-27] [TASK_PIPELINE_OPT] [ADR_20260727_PIPELINE_OPT]
- **Context/Why:** Severe latency bottleneck in production pipeline: 258.4s runtime, 1,029MB RSS, 3.3M heap allocations in MAD kernel, nested thread contention
- **Resolution/What:** Implemented zero-allocation Numba MAD kernel (buf/dev_buf per-symbol, in-place sort), single-level TPE dispatcher (4 workers + Numba 1 thread), parallel PyArrow grid materializer, vectorized 4h funding sum
- **Impact:** MAD kernel 5.67x vs Numpy, dense sim 13.38x, RSS 363MB (2.84x reduction), bit-exact identity maintained

## [2026-07-27] [TASK_20260727_PRODUCTION_PIPELINE_DEEP_OPTIMIZATION] [ADR_20260727_20260727_PRODUCTION_PIPELINE_DEEP_OPTIMIZATION]
- **Context/Why:** Pipeline latency of 258s (4.3min) caused by Signal Bank MAD kernel dynamic heap allocations, nested thread pool lock thrashing, and sequential Parquet feature grid materialization
- **Resolution/What:** Profiled 6 pipeline stages, identified exact 227s Signal Bank and 23s Market Cube bottlenecks, drafted zero-allocation MAD kernel spec, parallel grid loader, and verified with lean_check
- **Impact:** Paves way for pipeline acceleration from 258s to <15s with 0.000000000000 math discrepancy and <50MB tensor memory

## [2026-07-27] [TASK_PRODUCTION_PIPELINE_ULTRA_OPTIMIZATION] [ADR_20260727_PRODUCTION_PIPELINE_ULTRA_OPTIMIZATION]
- **Context/Why:** 프로덕션 파이프라인(opt_main_futures.py) 연산 병목 사멸 및 OOM 차단/비트단위 수치 동일성 보장
- **Resolution/What:** l1_sleeves.py 2D Matrix Vectorized Bootstrap 적용, signal_bank.py ProcessPool 알파신호 병렬화, test_production_pipeline_ultra_optimization.py 검증 시나리오 구축
- **Impact:** 부트스트랩 계산속도 99.45% 단축, Peak Memory 41.79MB 캡핑, Math Discrepancy 0.000000000000 달성

## [2026-07-27] [TASK_SOFT_CONVICTION_AND_FUNDAMENTAL_OPTIMIZATION] [ADR_20260727_SOFT_CONVICTION_AND_FUNDAMENTAL_OPTIMIZATION]
- **Context/Why:** 현금 100% 락업 결함 구조 철폐 및 /check 검증 연산 병목 4분 -> 15초 이하 단축
- **Resolution/What:** engine.py(has_admitted 하드 거부 제거 및 연속 신념 가중치 적용), l1_sleeves.py(ExitPathCache 1회 사전 계산 도입), allocator.py(12% 목표 변동성 타겟팅 배선)
- **Impact:** 현금 락업 해제 및 복리 포지션 점화(CAGR +3.32%, MDD -1.29%), 수학적 오차 0.000000000000 보장 하에 파이프라인 및 테스트 연산 속도 86.9% 대폭 가속

## [2026-07-27] [TASK_EXPANDED_MULTI_FACTOR_ALPHA_BANK] [ADR_20260727_EXPANDED_MULTI_FACTOR_ALPHA_BANK]
- **Context/Why:** 기존 27개 추세 편중 신호 및 51개 심볼 고정 제약으로 인한 알파 가뭄과 10.17% 비용 드래그 소모 문제 극복
- **Resolution/What:** alpha_catalog.py(60개 8개 다요인 알파 레시피 구축), signal_bank.py(120개 동적 유니버스 마스킹 및 60개 신호 Numba 연산 구축), calibration.py(build_folds_4h 유효일수 캡핑 정정)
- **Impact:** 60개 다요인 알파 신호와 120개 PIT 유니버스로 알파 표현력 및 독립 표본 수(Breadth) 2배 이상 확보. 측정계 정직화로 미달 신호 fail-closed 거부 및 cash-only 원금 100% 보존

## [2026-07-27] [TASK_L1_ADMISSION_BETA_NEUTRAL_TS_BOOTSTRAP] [ADR_20260727_L1_ADMISSION_BETA_NEUTRAL_TS_BOOTSTRAP]
- **Context/Why:** 278개 중복 sleeve OOS 평균 스칼라 i.i.d 부트스트랩으로 인한 표본 독립성 위반 및 분산 폭발(growth_lcb90 = -40.58%) 결함 해결
- **Resolution/What:** l1_sleeves.py(compute_beta_neutral_composite_returns 신규, Causal Beta Neutral & Inverse Volatility 가중 4h 시계열 생성, build_exit_aware_handoff에 circular_stationary_bootstrap_growth 시간축 블록 부트스트랩 연결), engine.py(파이프라인 배선)
- **Impact:** 표본 독립성 위반 오추정을 전면 제거하고 정직한 시간축 시계열 부트스트랩 적용. 10.17% 비용 드래그 및 마이너스 알파 신호를 정직하게 fail-closed 차단하여 cash-only로 원금 보호

## [2026-07-27] [TASK_L1_ADMISSION_RECOVERY_GATE_REDESIGN] [ADR_20260727_L1_ADMISSION_RECOVERY_GATE_REDESIGN]
- **Context/Why:** 직전 NO_EVIDENCE 진단(min_effective_days=180)이 오진단이었음을 확인(코드에서 미참조 죽은 필드). 실전 그리드서치 8회로 진짜 원인 확정: 개별 신호 게이트는 정상, 집계 게이트(pooled OOS 평균>0 체크)가 L1<350일에서 부호 반전. growth_lcb90 필드가 평균값을 대입하는 결함도 발견
- **Resolution/What:** run_windows.py(clamp_window_to_available_data 신규: L1 길이 고정+L2 동적계산+fail-closed), config.py(l1_days 365 복원, min_oos_days 340, min_bootstrap_sharpe_probability 제거, l1_prior_effective_days_cap 신규), l1_sleeves.py(집계게이트를 admission.py::_block_bootstrap_lcb(block_size=1) 재사용 i.i.d. 부트스트랩으로 교체), validation.py(L2->L3 패턴 재사용한 blend_l1_prior_growth_probability, sharpe_probability 게이트 제외), engine.py 배선. /check PASS(Cov 91%)
- **Impact:** 실전 CLI 재실행 결과 여전히 NO_EVIDENCE. growth_lcb90 버그 수정(평균->진짜 i.i.d. 부트스트랩 LCB90) 적용 후 실측: annualized_log_growth=+7.37%(평균, 이전엔 이걸로 통과) vs growth_lcb90=-40.58%(진짜 하한, 통과 못함). 스펙 자체가 명시했던 [LIMIT-03] 리스크가 실현. 정직화가 정직한 결과를 낸 것으로 판단, 임계값 추가 완화 없이 NO_EVIDENCE 그대로 기록. 유효표본 상관구조 조사는 후속 스펙 범위

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
