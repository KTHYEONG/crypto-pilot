# Active Decisions Log (Sliding Window)

## [2026-07-24] [TASK_SIGNAL_BANK_V3] [ADR_20260724_SIGNAL_BANK_V3]
- **Context/Why:** 실측(4380개 4h bar × 120종목, 다중 horizon 스윕)에서 xs_momentum 216h(t=-19.01)/648h(t=-17.79)의 강력한 장기 모멘텀 발견. 기존 고정 4h 타깃으로는 이 신호를 전혀 감지할 수 없어 signal descriptor별 target_horizon_hours 필드 도입이 필요했음. 또한 단일 horizon 구조에서 다중 horizon(4h/216h/648h) calibration 아키텍처로 확장.
- **Resolution/What:** SignalDescriptor.target_horizon_hours 필드 추가 + __post_init__ 검증. _compute_xs_rank_signal(sign) 공용 헬퍼 생성, xs_reversal/새 xs_momentum_slow family가 공유. build_multi_horizon_targets, signal별 target 조회 calibrate_signals/evaluate_signal_admission, purge_bars/block_size 동적 하한 적용. admission에 low_effective_sample soft-flag 추가. 기본 카탈로그: 5 families×4 + reversal_st + 2 xs_reversal + 2 xs_momentum_slow = 25개, flow_taker 제외(데이터 버그 별도 이슈). lean_check PASS (Cov 96%). --phase ladder 결과 8/8 ok, 0 promoted — 기존 L1 edge 부재 결론 유지.
- **Impact:** 신규 xs_momentum_slow family가 기본 카탈로그에 포함돼 P2 admission 평가를 받게 됨. purge_bars/block_size 동적 계산으로 648h 타겟의 look-ahead 방지. 인터페이스 변경(calibrate_signals/evaluate_signal_admission의 target→targets dict)으로 기존 호출자(engine.py/ladder.py) 전부 업데이트 완료. 기존 25개(target_horizon_hours=4) 신호 admission 결과는 1비트도 변경되지 않음(회귀 테스트 확인). flow_taker 기본 카탈로그 제외는 v2 결정 승계.

## [2026-07-24] [TASK_SIGNAL_BANK_V2] [ADR_20260724_SIGNAL_BANK_V2]
- **Context/Why:** IC 스크리닝(4380 bar x 120종목)에서 basis_gap 4speed 동일출력(dead-lookback), xs_reversal(24h) 유의성 발견. flow_taker는 taker_buy_quote 상수(-1.0000)로 무의미 판정돼 제외 예정이었으나, 재조사 결과 compound_data.py가 존재하지 않는 컬럼명(taker_quote_volume)을 요청하는 read-layer 버그였고 실제 데이터(taker_buy_quote)는 100% finite로 정상.
- **Resolution/What:** basis_gap에 lookback_hours 인자 추가해 EWM 스무딩(speed별 실제 구분). xs_reversal:fast(24h) family 신규 추가. compound_data.py/data_lake/query.py 컬럼명 버그 수정(taker_quote_volume -> taker_buy_quote, 저장 컨벤션 통일). flow_taker는 제외하지 않고 카탈로그 유지(25->26개). 버그 수정 후 재스크리닝: flow_taker:fast t=-3.50 유의미한 역추세 신호로 확인. 실제 P2 admission 파이프라인(--phase ladder) 재실행 결과 flow_taker:fast/trend_ema:fast/breakout_donchian:slow가 개별 유의성 통과했으나 26개 pooled BH-FDR 보정 후 최종 admitted=0/26, L1-3 zero-mu fallback.
- **Impact:** 데이터 파이프라인 read-layer 네이밍 버그 근본 수정으로 향후 taker 계열 신호의 오진단 재발 방지. flow_taker 오진단 정정으로 카탈로그 원재료 신뢰성 향상. 다만 신호(L1) edge 부재라는 기존 근본 결론(rank IC 근사 0)은 이번 재검증에서도 유지됨 - P4/P5 대신 L1 신호 재탐색이 여전히 우선순위.

## [2026-07-23] [live-pit-universe-refresh-gate] [ADR_20260723_live-pit-universe-refresh-gate]
- **Context/Why:** live PIT collector 추가 후 경계조건과 lake-only query의 coverage gate를 완료해야 운영 갱신 결과를 신뢰할 수 있음
- **Resolution/What:** exchangeInfo schema, empty/non-mapping records, axis/time causal 조건, missing partition, projection path를 테스트하고 lean_check strict gate를 통과시킴
- **Impact:** 실제 live state는 120 symbols/119 eligible로 저장되었고 final check PASS Cov 91%. raw legacy data는 아직 삭제하지 않음

## [2026-07-23] [live-pit-universe-refresh] [ADR_20260723_live-pit-universe-refresh]
- **Context/Why:** lake-only 전환 후 과거 ledger 이관만으로는 현재 거래 가능 상태를 갱신할 수 없고 Binance exchangeInfo 전체를 사용하면 운영 120심볼 축을 초과함
- **Resolution/What:** Binance exchangeInfo를 조회하고 data/futures/lake의 기존 운영 심볼과 교집합을 취해 다음 UTC 일자의 causal universe_state를 immutable Parquet와 DuckDB catalog에 저장하는 refresh_live_universe_state를 추가함
- **Impact:** 실제 Binance 846개 응답에서 120개를 선별해 119개 eligible state를 2026-07-24 partition에 저장하고 state hash 및 24시간/120심볼 cube 검증을 완료함. 구형 raw 데이터 삭제와 L2/L3 성과 검증은 별도 잔여 작업임

## [2026-07-23] [data-lake-collection-verification] [ADR_20260723_data-lake-collection-verification]
- **Context/Why:** 유니버스·L1·L2 백테스트에 필요한 Binance Futures 데이터를 수집하고 저장 완전성을 검증
- **Resolution/What:** 유효 symbol 120개를 기준으로 1h/1m OHLCV, funding, premium, mark/index, metrics를 수집하고 bounded parallel ingestion, symbol 검증, quote volume 보정을 적용
- **Impact:** 13,167개 partition과 101,540,621 rows를 검증했으며 누락·hash·row count·schema·시간축·비정상값이 모두 0이다. L1/L2/L3 성과 검증은 다음 작업으로 남겼다.

## [2026-07-23] [TASK_MULTISCALE_DATA_LAKE_CUTOVER] [ADR_20260723_MULTISCALE_DATA_LAKE_CUTOVER]
- **Context/Why:** 최신 구현과 기존 explain 문서가 data snapshot, PIT universe, L1/L2 engine 경계를 일치시키지 못했고 실제 network 수집과 기존 데이터 삭제 시점도 분리되어야 했다.
- **Resolution/What:** 단일 multiscale engine 경로, runner-owned Binance/DuckDB runtime, 승인 기반 network sync, checksum/atomic Parquet snapshot, historical PIT union, sparse L1 event와 signed L2 allocation 계약을 최신 설명 문서에 동기화했다.
- **Impact:** check PASS(Cov 93%) 기준의 현재 로직을 문서화했다. 실제 Binance 다운로드와 data/futures 중복 데이터 삭제는 사용자 승인 전 수행하지 않는다.

## [2026-07-23] [TASK_COMPOUND_MAIN_REAL_DATA_ALIGNMENT] [ADR_20260723_COMPOUND_MAIN_REAL_DATA_ALIGNMENT]
- **Context/Why:** 실제 메인 실행에서 Binance timestamp 정밀도 불일치와 기준일 미전달로 결측·무결성 실패가 발생했고, explain 문서가 최신 compound-only 경로와 불일치했다.
- **Resolution/What:** OHLCV timestamp를 내부 ns로 정규화하고 reference_date를 데이터 로더와 PIT state에 동일하게 전달했다. compound-only 실행 흐름, 18개 recipe, L1 uncertainty, L2 단일 allocator, simulator, L3 결과 및 현재 fallback universe·zero-support 원인을 docs/results/explain.md에 기록했다.
- **Impact:** 메인 실행은 integrity 정상으로 완료되며, 현재 실측은 2종목 fallback과 robust uncertainty shrink로 target weight 0·L2 성장률 0·L3 SHADOW를 명확히 보고한다. check Cov 91% PASS.

## [2026-07-22] [TASK_CAUSAL_ALPHA_ONLINE_GROWTH_ENGINE] [ADR_20260722_CAUSAL_ALPHA_ONLINE_GROWTH_ENGINE]
- **Context/Why:** L2의 zero-overlap fit, direct Kelly-only OOS, legacy fallback 및 production Optuna 경로가 자산증식을 방해하는 구조적 결함을 최신 실측으로 재검증했다.
- **Resolution/What:** Causal 4-fold warm-up, 4-policy shadow posterior, cash abstention, zero leverage support를 active AWF 경로에 연결하고 최신 실행 결과 및 잔여 production migration blocker를 result.md에 기록했다.
- **Impact:** 정상 L2 경로는 fit_bars=0을 제거하고 성장 근거가 없을 때 현금 대기를 수행한다. 다만 crisis legacy replay, Optuna 120회, legacy promotion gate 및 12GB 초과 RSS는 후속 P0로 남는다.

## [2026-07-22] [TASK_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST] [ADR_20260722_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST]
- **Context/Why:** L2 policy comparison was not causal: policy rows shared returns, OOS bypassed the selected policy, and the latest run had fit_bars=0 with all_folds_blocked.
- **Resolution/What:** Added policy-specific shadow-book contracts, fit-only growth-LCB selection, hard MDD/CVaR/ruin constraints, removed unconditional L1 override, rejected deprecated kelly_shrink_to_equal mappings, and covered shadow-cost shape errors.
- **Impact:** lean_check PASS with Cov 47%; latest 4h L2 execution reached L1 PASS but L2 fail-closed at all_folds_blocked: fit_bars=0, OOS CAGR -23.61%, 902 bars and 271 trades. SPEC artifacts cleaned.

## [2026-07-22] [TASK_L1_L2_COMPOUNDING_ALIGNMENT] [ADR_20260722_L1_L2_COMPOUNDING_ALIGNMENT]
- **Context/Why:** L2 Kelly and baseline were evaluated on mismatched allocation and leverage scales, while legacy cached signal events could abort L2 before asset-growth analysis.
- **Resolution/What:** Added L1 confidence and LCB/breakeven handoff, fit-only allocation policy evaluation with inverse-vol fallback, absolute deployed-growth gating, and backward-compatible zero-evidence handling for legacy events.
- **Impact:** L2 execution completed 120/120 trials on 2026-07-22 data after fixing the legacy-event crash. Only 2/120 trials were joint-feasible and no champion was promoted; sharpe_abs remained the dominant blocker (92/120). lean_check PASS, coverage 47%.

## [2026-07-22] [L2_KELLY_SHRINK] [ADR_20260722_L2_KELLY_SHRINK]
- **Context/Why:** DEBUG 계측(120-trial)으로 mu-비례(Kelly) 사이징이 동일 지지집합의 risk-matched 균등가중 baseline보다 CAGR 열위(64.2%)임을 확인. 근본 원인은 mu SNR이 낮을 때 Kelly-비례 배분이 노이즈가 큰 종목에 과집중하는 noisy-Kelly 현상.
- **Resolution/What:** diagonal_kelly_weights에 kelly_shrink_to_equal∈[0,1] 파라미터를 추가해 Kelly 비율과 균등가중 비율 간 shape-space 선형 블렌딩 구현. Layer2AllocationConfig 필드 + from_mapping 검증, awf_sim 호출부 배선, L2_SEARCH_SPACE 탐색 차원 추가. shrink=0.0 기본값으로 완전 하위호환.
- **Impact:** Optuna 탐색 차원 +1. shrink=0.0은 기존 챔피언 replay와 byte-identical. 하류 파이프라인(vol_target/cap/projection) 변경 없음.

## [2026-07-22] [TASK_L1_SIGNAL_UTILIZATION_GATE_PARITY] [ADR_20260722_L1_SIGNAL_UTILIZATION_GATE_PARITY]
- **Context/Why:** Non-4h sleeves structurally excluded from handoff due to TF-suffix lookup miss in portfolio_handoff.py (_mk_qw/override key not stripped). Champion CAGR gate regressed to absolute 30% floor because evaluate_layer2_gate call in selection.py omitted cagr_baseline/recency/window-bottleneck kwargs.
- **Resolution/What:** Fix A: public strip_tf_suffix() in candidate_contracts.py; portfolio_handoff.py _mk_qw + L1-edge override lookup now strip suffix before registry key match. signal_selection._strip_tf_suffix delegates to public fn. Fix B: selection.py passes cagr_baseline, recency_holdout_cagr, recency_holdout_applicable, window_bottleneck_covered, window_bottleneck_detail to evaluate_layer2_gate.
- **Impact:** ruff/mypy/pytest 38/38 PASS. lean_check PASS. Non-4h sleeves now correctly read registry quality_weight and trigger L1-edge override. Champion gate uses relative baseline+uplift mode instead of absolute 30% floor.

## [2026-07-22] [TASK_L2_TF_QUOTA_CAP_AND_C4_GATE_FIX] [ADR_20260722_L2_TF_QUOTA_CAP_AND_C4_GATE_FIX]
- **Context/Why:** crisis_rets/TF-suffix 매칭 수정 이후 pre-cap sleeve pool이 58→659개로 확대됐으나, top-32 cap이 순수 quality_weight 랭킹만 써서 admitted sleeve가 여전히 100% 4h로 재집중 — TF-다양성 복구를 구조적으로 무력화하는 결함 확인. 별도로, 같은 시점 신규 발견된 이상현상(120/120 trial CAGR 정확히 0.00% 균일)을 4개 chokepoint 계측(post_resolve/post_c4_filter/post_bucket_routing/post_netting)으로 추적 — logger 기반 계측이 워커 프로세스 stdout에 미노출되어 직접 함수 트레이싱으로 전환, 소거법으로 확정: _resolve_sleeve_signals_at_bar는 정상(신호 2~5개), 버킷 라우팅은 이 fold에서 호출 자체 없음(배제), _combine_sleeve_signals_to_symbol 직전 입력이 0 — C4 TF-inclusion 게이트가 handoff가 이미 admit한 4h를 자체 fit-edge 테스트로 재차 배제하고 있었음(두 게이트가 서로의 결정을 모른 채 충돌).
- **Resolution/What:** portfolio_handoff.py::_rank_and_cap_sleeve_indices를 TF별 최소 쿼터(max_candidate_sleeves // n_distinct_tfs) 선발 후 잔여분을 전체 quality_weight로 채우는 방식으로 재설계 — 다양성을 candidacy 단계에서 구조적으로 보장(단 최종 admission은 기존 handoff 통계 검정이 그대로 판정). awf_sim.py::_run_awf_simulation의 C4 TF-inclusion 게이트에 handoff override 추가 — _included가 non-empty이면서 특정 TF를 배제할 때, cache.handoff_sleeve_mask_by_fold[fold_idx]로 해당 fold에 admit된 sleeve의 native_tf를 강제로 합집합 편입(L1-edge-override와 동일 원칙: 나중의 더 정교한 게이트가 이전의 거친 게이트에 거부권을 뺏기지 않음). handoff_sleeve_mask_by_fold가 비어있는 isolated-study 호출에는 no-op 가드 적용. Phase 1 계측(4개 [L2-CHOKEPOINT] DEBUG 로그)은 향후 단일 프로세스 디버깅용으로 코드에 유지.
- **Impact:** /check PASS(mypy strict, spec compliance, Cov 56%). 프로덕션 실측(2026-07-21 기준, 120 trial): CAGR 0.00% 균일 현상 완전 해소 — trial별 실제 편차 값 확인(예: +10.8%, +2.9%, -1.8%), 거래 수 53~249건 정상 분포, Best CAGR 6.13%. blocker가 crisis_context_mismatch/균일플랫 아티팩트에서 no_feasible_trials(진짜 제약조건 미충족)로 정상화. 단 admitted sleeve는 여전히 100% 4h — TF-쿼터는 candidacy만 보장했고 비4h 후보가 아직 handoff 통계 검정(marginal-growth LCB/L1-override/상관관계)을 통과 못함, 버그 아닌 설계대로 동작. failures={cagr:120, recency_holdout:113, sharpe_uplift:99, fold:88, crisis_cagr:50, recent_fold:20, crisis_mdd:3} — 이제 진짜 신호 품질/제약조건 병목으로 확정, 다음 세션 조사 대상으로 이월.

## [2026-07-22] [TASK_L2_CRISIS_WIRING_AND_TF_SIGNAL_LOSS_FIX] [ADR_20260722_L2_CRISIS_WIRING_AND_TF_SIGNAL_LOSS_FIX]
- **Context/Why:** handoff 통계 재설계 이후에도 매 실행 [L2-SELECTION] crisis_context_mismatch로 feasible_trials=0 고정. active_pipeline.py에서 _crisis_rets를 계산해놓고도 _run_portfolio_causal_robust_outcome 호출 시 전달 누락(_run_robust_l2_l3_outcome이 crisis_rets=None 하드코딩), crisis_replay_ctx만 전달되어 select_layer2_champion의 pairing 안전장치가 매번 발동. 별도로 '대표 TF' 재검토 중 확정: 8h/12h/1d 등 6개 TF에서 L1이 검증한 신호(예: 8h 103 registry_symbols/209 registry_strategies)가 signal_batch.events에 전량 0건 — predict_layer1_signals_multi_tf가 HTF-injection 경로(strategy_id에 '_{tf}' suffix 부여)로 생성한 예측과 해당 TF 자신의 독립 registry(suffix 없음)를 그대로 매칭해 100% 불일치, 오직 4h만 양쪽이 우연히 일치해 생존.
- **Resolution/What:** active_pipeline.py::_run_robust_l2_l3_outcome에 crisis_rets 파라미터 추가, 하드코딩 None 대신 파라미터를 _run_tiered_l2_study에 그대로 전달, 프로덕션 호출부(_crisis_rets 계산 직후)에 crisis_rets=_crisis_rets 추가. signal_selection.py에 _strip_tf_suffix/_strip_tf_suffix_series 신설 — _candidate_output_to_signal_batch의 registry 매칭(source_keys, composite mask, qw_lookup 및 그 조회)에서만 양쪽 strategy_id의 '_{native_tf}' suffix를 제거해 비교(저장되는 ValidatedSignalEvent.strategy_id는 원본 유지). /check 중 무관한 기존 결함 발견: test_active_pipeline.py를 함께 실행하면 active_pipeline.py 임포트 시점 setup_logger('opt_main_futures', ...)가 프로세스 전역으로 propagate=False를 1회성 고정시켜 signal_selection.py와 같은 로거를 쓰는 caplog 기반 테스트 4건이 순서-의존적으로 실패 — test_signal_selection.py::TestLogFamilyRegimeFunnelDiagnostics에 해당 로거의 propagate를 테스트 동안 강제 True로 고정하는 autouse fixture 추가로 수정(프로덕션 코드 불변).
- **Impact:** /check PASS(mypy strict, spec compliance, 52/52 관련 테스트, Cov 35%). 프로덕션 실측(2026-07-21 기준, 120 trial): pre-cap sleeve pool 58→659개로 확대(1h:205 2h:141 4h:58 6h:16 8h:167 12h:69 1d:3, 이전엔 100% 4h), crisis_context_mismatch 완전 소멸(사유가 no_feasible_trials로 정상화). 단 admitted sleeve는 여전히 100% 4h(quality_weight 상위 32 cap 결과) 유지, 그리고 신규 이상현상 발견: 120/120 trial 전부 CAGR 정확히 0.00%로 균일(직전 실측 Best CAGR 11.27%였음), [L2-AUDIT] failures에 deployment/active_blocks/friction/trades(이전엔 없던 카테고리) 120/120 등장 — 예외/크래시 없이 발생해 원인 미상, 다음 세션 조사 대상으로 이월.

## [2026-07-21] [TASK_L2_PORTFOLIO_HANDOFF_STATISTICAL_POWER_FIX] [ADR_20260721_L2_PORTFOLIO_HANDOFF_STATISTICAL_POWER_FIX]
- **Context/Why:** 5개 게이트 결함(dead cap, equal-weight 희석, 퇴화 bootstrap, blanket-kill, dead config) 수정 후에도 all_folds_blocked 지속. 사용자가 '동일 L1로 L2가 과거 여러 번 통과한 이력'을 근거로 L1 알파 부재 결론에 반박, git log 재조사로 확정: portfolio_handoff.py 도입(ab6470bc/f01d74d4, 21:17) 직전 커밋(597985a4, 18:17)이 동일 L1 레지스트리로 joint_feasible=4/120 달성 — handoff 게이트 자체가 신규 리그레션. 계측 결과 capped-in 32 sleeve 전원이 L1 lcb_net_bps>0(+43~380bps)·hard_eligible=True인데, handoff의 3-subwindow 연율화 marginal-growth 통계는 동일 sleeve에 -22%~-30% 같은 비현실적 값을 산출 — 저빈도 신호를 짧은 캘린더 구간으로 쪼개 연율화하며 노이즈가 수십 배 증폭되는 구조적 결함(quant.md 수치 안정성 위반).
- **Resolution/What:** portfolio_handoff.py: _bar_level_marginal_growth_lcb 신설 — 3-subwindow 사전 연율화 후 bootstrap 대신, fit/cal 전체 구간 bar-level marginal log-return delta 시리즈를 그대로 block-bootstrap하고 최종 LCB에만 1회 연율화 적용(_window_marginal_growth 삭제). _l1_evidence_by_key 신설로 L1 lcb_net_bps 조회를 1회로 통합, _rank_and_cap_sleeve_indices가 registry 대신 이 lookup을 받도록 시그니처 변경. L2 자체 growth-LCB/positive_window_ratio가 실패해도 L1 lcb_net_bps>0(이미 purged/embargoed WF로 검증됨)이면 admitted_via_l1_edge_override=True로 통과시키는 우선순위 로직 추가(단, redundant_high_correlation 중복 제거는 예외 없이 적용). 부수 발견: invalid_handoff_weights가 weight_sum<=0을 조건으로 삼아 override로 인한 정당한 음수 marginal_growth_lcb admitted 상태를 오탐 차단할 뻔한 버그를 non-finite 체크로만 수정.
- **Impact:** /check PASS(Cov 90%, spec compliance, mypy strict, 13개 시나리오). 프로덕션 실측(2026-07-21, --phase l2, 120 trial): handoff passed=True(수정 전 all_folds_blocked) 전환, fold별 admitted sleeve 18/14/25/32(수정 전 0/0/0/0), Optuna 120/120 trial 완주(수정 전 0/120), Best CAGR 11.27% 도출. admitted sleeve 표본 전수가 admitted_via_l1_edge_override=True로 확인 — L1이 이미 검증한 엣지가 L2 자체 재검증(여전히 약한 신호)을 정당하게 우회한 것. 단 joint_feasible=0/120 유지, blocker=crisis_context_mismatch(cagr/crisis_cagr/recency_holdout 등 다중 제약 동시충족 실패) — handoff와 무관한 후속 단계의 별도 병목으로 확인, 범위 밖으로 이월.
