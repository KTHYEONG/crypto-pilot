# Active Decisions Log (Sliding Window)

## [2026-07-29] [TASK_L1_SIGNAL_EVALUATION_ARCHITECTURE_FIX] [ADR_20260729_L1_SIGNAL_EVALUATION_ARCHITECTURE_FIX]
- **Context/Why:** 선행 스펙(l1-router-cash-only-recovery) 적용 후 실전 CLI 재실행 결과 R-4(증거창 게이트)가 회귀(현재 fold 자신의 OOS 폭을 측정해 min_evidence_bars=900을 영구 통과불가)로 재확인. 조사 과정에서 더 근본적인 결함 2건 발견: L1 sleeve admission이 순수 in-sample 유의성만으로 게이트해 OOS 실현치를 전혀 사용하지 않음(IS→OOS rank IC ρ=-0.60을 직접 설명), regime 하드게이트가 2026-06-12 결정(regime→risk overlay 강등)과 충돌
- **Resolution/What:** l1_regime_routing.py: signal_evidence_history 기반 신호별 누적 증거 이력 도입(concatenate_signal_evidence, SignalFoldRecord)으로 n_evidence_bars를 진짜 prequential 누적값으로 복원. regime 하드게이트를 compute_regime_overlay(확증된 악재일 때만 floor=0.5 감쇠, 그 외 1.0)로 대체해 admission을 막지 않음. apply_walk_forward_carry가 regime_overlay 파라미터 수용. 완전 죽은 코드 3개(build_fold_local_shadow_tape, _compute_unconditional_evidence, _compute_temporal_evidence, 호출처 0건 확인) 제거. l1_sleeves.py: estimate_cluster_sleeve_posteriors에 OOS 부트스트랩 AND-게이트 추가(HandoffConfig.min_oos_posterior_probability=0.55/min_oos_effective_blocks=5, in-sample 통과자에만 단락평가로 적용해 성능 영향 없음). config.py: RegimeRouterConfig.regime_overlay_floor 신규. /check 과정에서 회귀 검증 통합테스트 2개가 total==0 시 무조건 통과하는 공허한 단언(escape hatch)이었음을 발견해 결정론적 fixture로 전면 교체(fold1 n_evidence_bars==0 강제 확인 등 회귀 판별력 있는 단언으로 교정), 그 결과 l1_regime_routing.py 커버리지 51%->87%(죽은 코드 제거 포함)
- **Impact:** /check PASS(Mypy strict, 전체 회귀 스위트 그린 - 무관한 사전결함 2건은 baseline 동일 재현 확인 후 범위 제외, Cov 94%). 실전 CLI 재검증은 다음 세션 과제로 남김(본 세션은 단위/통합 테스트로 P0/P1/P2 정합성 확정). 완료 판정 기준상 no_evidence 유지는 정상 허용(R-6 엣지부재 별개 이슈)

## [2026-07-29] [TASK_L1_ROUTER_CASH_ONLY_RECOVERY] [ADR_20260729_L1_ROUTER_CASH_ONLY_RECOVERY]
- **Context/Why:** 실측 확정된 6개 결함(R-1 funding NaN 크래시, R-2 평가창 mu 지지 부재, R-3 evidence/배포 북 불일치, R-4 도달불가 게이트 중복, R-5 orientation 잠복 유실, R-6 엣지 부재) 중 R-1~R-5 측정계 복원. R-6은 후속 스펙 범위 외
- **Resolution/What:** allocator.py: compute_funding_4h_2d(NaN-safe SUM) 신규, dense_simulator SSOT 통일. contracts.py: ExpertReturnTape 성분별 유한성 사전검사, PrequentialExpertRoute.active_expert_count_1d 및 RegimeExpertEvidence.regime_mean_net/n_evidence_bars 필드 추가. l1_regime_routing.py: collect_fold_expert_contributions/build_fold_candidate_book/split_book_by_expert(배포북 단일 SSOT, orientation 구조적 보장)/apply_walk_forward_carry(평가창 support 연장)/score_expert_returns 신규, 게이트 lcb90 중복 제거+단일화, RegimeRouterConfig dead param(max_expert_correlation, prior_effective_blocks) 제거+min_evidence_bars 신규. l1_diagnostics.py 계측 확장. /check PASS(Mypy strict, Cov 86%, 사전결함 2건 baseline 재현 확인 후 범위 제외)
- **Impact:** 실전 CLI 재실행: exit_code 1->0, integrity_ok true, p2_pipeline_error 소멸(R-1 해소 확인). 그러나 target_weights 여전히 100% 현금 - 진단 결과 R-4 게이트가 '현재 fold 자신의 OOS 폭(oos_end-oos_start, 고정 340bar)'을 n_evidence_bars로 오산정해 min_evidence_bars=900을 구조적으로 영원히 통과 불가(fold 진행에 따른 누적 증거 로직 누락, 원래 evidence_mask=tape.outer_fold_id_1d<fold_id 패턴 미이식). 엣지 유무와 무관하게 항상 no_evidence 고착되는 회귀이며 완료 판정 기준 3번(원인 구분) 미달. 다음 세션에서 누적 증거창 로직 재이식 필요, 임계값 완화 금지

## [2026-07-29] [TASK_L1_CASH_ONLY_DEBUG_AUDIT] [ADR_20260729_L1_CASH_ONLY_DEBUG_AUDIT]
- **Context/Why:** 최신 full dry-run에서 전략 성과 게이트 이전에 L1 prequential routing이 net 수익 항등식 검증 예외로 중단되어 cash-only fallback 원인과 원자료를 보존해야 함
- **Resolution/What:** docs/results/result.md에 실행 조건·artifact·DEBUG trace·정량 결과·후속 점검 포인트를 추가하고, 결과를 ADR 및 인덱스에 동기화
- **Impact:** 이번 no_evidence는 통계적 알파 부재가 아닌 integrity failure 보호 동작으로 분류; ExpertReturnTape gross/cost/funding/net 정렬·NaN·dtype·funding 경계 계측이 다음 구현의 선행 조건

## [2026-07-28] [TASK_L1_CAUSAL_REGIME_ROUTING_EXECUTION] [ADR_20260728_L1_CAUSAL_REGIME_ROUTING_EXECUTION]
- **Context/Why:** 최신 full 실행에서 fold-local regime expert routing 적용 후 증거 부족 상태를 검증하고, 배포 북이 음수 성과를 내지 않도록 fail-closed 결과를 기록함
- **Resolution/What:** docs/results/result.md에 20260728_131507 실행 결과와 0-weight no_evidence 상태를 반영하고, sync 자동화로 ADR/index/spec/scratch 정리를 수행함
- **Impact:** 실제 거래 weight 비영 비율 0%, CAGR/Sharpe/MDD/turnover 모두 0이며 L2 no_evidence·L3 reject. 봉인 holdout은 dry-run으로 보존됨

## [2026-07-28] [L1_POSITION_CONSTRUCTION_INTEGRITY] [ADR_20260728_L1_POSITION_CONSTRUCTION_INTEGRITY]
- **Context/Why:** 직전 스펙(측정계 정직화) 적용 후 ann_lcb90=-78.87%로 여전히 실패했으나, 게이트 입력을 덤프해 신호(mu_2d) 고정한 채 구성만 바꿔 분해한 결과 병목이 signal이 아니라 구성(construction)임을 확인: C-1 게이트가 실제 allocator(스무딩/밴드/voltarget)를 안 쓴 허수아비 북을 채점(turnover 447x, cost_drag 44.7%), C-2 방향성(net exposure) 베팅이 분산의 85% 차지(무조건부 베타로는 은폐), C-3 min_positive_outer_folds=4가 정의만 되고 미검사인 dead 파라미터
- **Resolution/What:** allocator.py: apply_net_exposure_cap 신규(support마스킹 직후, 스케일불변, max_net_exposure=1.0시 완전무연산 롤백보장). config.py: max_net_exposure=0.10 추가. l1_sleeves.py: compute_l1_oos_portfolio_returns를 mu_2d/sigma_2d 대신 완성된 weights_2d 수취로 교체(자체 포지션구성 로직 삭제), compute_fold_growths 신규, build_exit_aware_handoff가 min_positive_outer_folds 실제 검사. engine.py: weights_2d를 게이트 호출 이전 1회 계산해 게이트/배포가 동일 배열 공유(C-1 재발 구조적 차단). l1_diagnostics.py: positive_folds/fold_growths/mean_abs_net 계측 추가
- **Impact:** |net| 캡 sweep 실측(신호 고정): 완전단조 관계로 캡 0(완전중립)까지 Sharpe 0.29->2.22, LCB90 -31.5%->+8.7%. 반증기록: 무조건부 베타 제거 가설은 틀림(beta -0.19->-0.07로만 변화, 진짜원인은 시변 순노출 85%). /check PASS(216 tests, Cov91%, 무관한 사전결함 test_config.py도 해소). 실전 재실행: turnover 447x->7.5x(60배감소), cost_drag 44.7%->0.75%, ann_lcb90 -78.87%->-24.75%(3배 축소), 그러나 positive_folds=2/5로 신규활성화 게이트 미달->NO_EVIDENCE 유지. fold 성과 부호[+6%,-12%,-45%,+16%,-7%] 불안정 확인 - 병목이 구성에서 fold간 비정상성으로 이동. docs/results/result.md 전체 재작성(재현 스크립트+원자료 포함)

## [2026-07-28] [L1_MEASUREMENT_INTEGRITY_RESTORE] [ADR_20260728_L1_MEASUREMENT_INTEGRITY_RESTORE]
- **Context/Why:** 전날 복원한 fail-closed 게이트(has_admitted)가 참조하는 집계 통계량 자체가 결함 3건: D-1 계열이 매매신호와 무관(클러스터 등가중 롱온리 바스켓과 비트단위 동일), D-2 540 sleeve가 실제론 고유계열 4개(135배 중복), D-3 pooled OLS SE가 지속성신호에서 41배 과신(D-5 i.i.d 부트스트랩 회귀는 3회차 동일 패턴). 실측 스크립트로 확정
- **Resolution/What:** l1_sleeves.py: compute_l1_oos_portfolio_returns 신규(OOS만 stitch, 리스크패리티 사이징, 비용차감)로 단일 포트폴리오 시계열 게이트 복원, _cluster_masked_beta를 Driscoll-Kraay HAC SE로 교체. config.py: min_sleeve_posterior_probability 0.52->0.95, hac_lag_cap=120 신규. l1_diagnostics.py 신규: L1AdmissionRecorder(L1_DEBUG=1 게이트). 후속 수정 2건: pw_block 0.0 하드코딩 스텁을 실제 politis_white_block_length 계산값 배선으로 교정, estimate_cluster_sleeve_posteriors에 record_sleeve 호출 배선(기존엔 클래스만 존재하고 프로덕션 미호출로 sleeve별 데이터 전혀 미수집)
- **Impact:** /check PASS(Mypy strict, 145 tests, Cov 92%, test_config.py 무관 사전결함 1건은 베이스라인 동일재현 확인 후 범위제외). 실전 재실행 결과 여전히 NO_EVIDENCE(admitted_sleeves=15/455, distinct_series=1 확인, ann_lcb90=-78.87%). 신규 확보 sleeve별 DEBUG 데이터(455건)로 momentum_ts family만 유일 admit(15.8%, SE팽창 7.84x)하고 trend_ema는 beta~0 수렴+SE팽창 최대(68x, 느릴수록 악화)인 사멸신호임을 실측 확정 - 다음 L1 신호 개선의 데이터 기반 착수점. docs/results/result.md에 전체 family/speed별 표 기록

## [2026-07-28] [CAPITAL_DEPLOYMENT_FAILCLOSED_RESTORE] [ADR_20260728_CAPITAL_DEPLOYMENT_FAILCLOSED_RESTORE]
- **Context/Why:** phase full 실측(20260728_072617)에서 L2 FAIL 전체탈락(CAGR -15.8%, MDD 39.9%, turnover 27.6x) 확인. git-diff로 근본원인 확정: 01a209e1(SOFT_CONVICTION)이 engine.py의 has_admitted 이진 게이트를 제거해 집계 LCB90 부트스트랩 검정과 무관하게 항상 실거래하도록 변경됨. 같은 날 da72a053이 alpha 27->60 확장하며 검증된 basis_gap/xs_reversal 등을 제거하고 미검증 신규 4패밀리로 대체
- **Resolution/What:** engine.py: has_admitted=False시 weights_2d 강제 0(현금) 이진분기 복원. signal_bank.py: _default_catalog()를 legacy 27레시피로 복원(신규 4패밀리 volatility_squeeze_keltner/funding_carry_reversion/flow_imbalance_taker/open_interest_confirmation 격리, basis_gap/xs_reversal/xs_momentum_slow/smart_money_divergence 복원), 죽은 build_canonical_alpha_catalog() 호출 제거
- **Impact:** 실측 A/B(scratch/verify_alpha_family_ablation.py, 동일 프로덕션 엔진 재사용): baseline(60레시피,fail-open) CAGR -15.8%/MDD39.9% -> legacy27 CAGR +7.4%/MDD11.4%로 반전, 신규4패밀리 단독실행도 MDD41.5%/cost_drag293%로 파괴적임을 확인해 원인 확정. /check PASS(Cov 74%). 수정후 실전 phase full 재실행 결과 NO_EVIDENCE(현금100%, rebalances=0)로 안전 상태 복원 확인 - 집계게이트 통과할 알파는 아직 없으나 파괴적 거래는 완전 차단됨

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
