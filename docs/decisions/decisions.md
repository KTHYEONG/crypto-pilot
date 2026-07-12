# Active Decisions Log (Sliding Window)

## [2026-07-12] [TASK_L0_CROSS_TF_BATCH_CORRELATION] [ADR_20260712_L0_CROSS_TF_BATCH_CORRELATION]
- **Context/Why:** `resolve_cross_tf_shared_context`의 O(N²) per-pair `np.corrcoef` 루프(2,556회/N=72, 각 호출 mean/std 재계산)가 cross-TF 구간 마지막 Python for-loop. microbenchmark 168.8ms(N=72, T=1000, S=10). 선행 ADR(batch jaccard+dict greedy)과 동일한 stacking+matmul 패턴 적용 가능.
- **Resolution/What:** `_batch_pairwise_corr()` 신규 helper: 4회 BLAS matmul(cross_sum, cross_count, row_sum, row_sq)로 N×N Pearson matrix 1-pass. `X = np.where(C_f > 0, P, 0.0)` NaN-safe. `resolve_cross_tf_shared_context` per-pair 루프 대체. per-pair fallback 유지.
- **Impact:** N=72 168.8ms→7.6ms(22×), max error 5.55e-17. 63/63 PASS, ruff+mypy PASS, RSS 8.4GB(budget 10GB 이내). L0 total(157-193s)에서 160ms saving은 noise이나 N≥100에서 42× scaling.

## [2026-07-12] [TASK_L0_CROSS_TF_BATCH_ACCELERATION] [ADR_20260712_L0_CROSS_TF_BATCH_ACCELERATION]
- **Context/Why:** 실측(l0_postimpl.log) cross-TF audit+pruning+bookkeeping ~223s(49.5%)가 여전히 최대 병목. 선행 ADR의 shared context로 cross-TF 단독 ~640s→~85s로 축소됐으나, `compute_cross_tf_redundancy`의 O(N²) per-pair jaccard(bool array 2,556회×140K셀) 및 O(N⁴) leader greedy list scan(13.2M string 비교)이 하위 병목으로 확인됨.
- **Resolution/What:** `resolve_cross_tf_shared_context()`에 entry_pos_flat/entry_neg_flat/n_entries(OPT-2 batch jaccard용 int8 flat arrays) 및 valid_stack(OPT-1-a corr-loop mask broadcast) precompute 추가. `compute_cross_tf_redundancy()`에 batch matmul jaccard(pos_stack@pos_stack.T + neg_stack@neg_stack.T → O(N²) 1회) 및 dict-lookup leader greedy(pair_map→O(1) 조회, O(N⁴)→O(N²)) 도입. per-pair fallback 경로 유지로 하위호환 보장. 실측 N=200 leader greedy 64x 단축, N=72 전체 파이프라인 1.3x 개선.
- **Impact:** 전체 파이프라인 0.437s→0.325s(-25.7%), leader greedy 0.030s→0.001s(-96.7%), N≥200 스케일에서 dict lookup 64x. 수학적 결과 byte-identical 보존(assert 검증 완료). 신규 메모리 ~120MB(증분, budget 60% 이내). 104 regression PASS. N=72 본 규모에서는 shared context 캐시 기반 per-pair fallback도 이미 빠르므로 실질 개선 0.1s 수준이나, fallback 경로 없거나 N≥200인 시나리오에서 batch alg improvent 본격 발휘.

## [2026-07-12] [TASK_L0_CROSS_TF_PRUNING_PERFORMANCE] [ADR_20260712_L0_CROSS_TF_PRUNING_PERFORMANCE]
- **Context/Why:** cross-TF pruning fix(직전 ADR) 후 cProfile 실측(72 candidates, 1h canonical) 결과 `compute_cross_tf_redundancy` 398.7s 중 `project_signal_to_canonical_grid`(72회 필요한데 5,184회), `_causal_projected_side_and_entry`(72회 필요한데 5,112회), `corrcoef`(2,556회 필요한데 7,740회) 전부 필요량 대비 3~72배 중복 재계산. audit+pruning 동시 활성화 시 두 함수가 각자 독립적으로 동일 계산을 반복하는 것도 확인.
- **Resolution/What:** `resolve_cross_tf_shared_context()`(신규, `CrossTFSharedContext`) 도입 — 캐시(proj_cache/side_entry_cache/corr 상삼각-미러링 행렬)를 1회 구축해 `compute_cross_tf_pair_evidence`/`compute_cross_tf_redundancy`/`audit_l0_selected_recipe_independence`에 `precomputed_shared_context`(additive)로 주입. `project_signal_to_canonical_grid` 반환 dtype float64→float32(정밀도 요구 없는 상관계수/자카드 비교용). 캐시 구축 전 `resolve_effective_memory_budget()`/`admit_memory_stage()` 가드 추가. check 단계에서 발견한 신규 타이밍 로그의 로거 가시성 버그(`_logger.info`→`setup_logger("opt_main_futures")`, 이 프로젝트 3회+ 재발 패턴) 및 caplog/capsys/capfd 전부 무력화되는 `propagate=False` 싱글톤 로거 테스트 이슈(`mocker.patch`로 우회)도 함께 수정.
- **Impact:** 실측(동일 조건 재실행) — 총 벽시계 **908.32s→450.58s(-50.4%)**, cross-TF 단계 자체 ~640s→~85s(-86.7%). `n_selected_total=72 n_independent_clusters=39 n_demoted=33 pruning_applied=True` 완전 동일(정합성 100% 보존, 순수함수 리팩터 검증). L0 게이트(Phase1+3) 157.1s→156.9s 불변(손대지 않은 영역 확인).

## [2026-07-12] [TASK_L0_CROSS_TF_CANONICAL_CALENDAR_CONTAINMENT_FIX] [ADR_20260712_L0_CROSS_TF_CANONICAL_CALENDAR_CONTAINMENT_FIX]
- **Context/Why:** 실측(`--phase l0`, `L0_CROSS_TF_DIVERSITY_AUDIT=1 L0_CROSS_TF_PRUNING=1`) 결과 cross-TF pruning/audit이 매번 `panel.datetimes must fall within canonical_datetimes range`로 100% fail-open — TF마다 독립적으로 정렬된 `AlignedMarketData`의 캘린더 범위가 서로 달라, 자동 선택된 canonical TF(가장 세밀함)가 다른 TF의 범위를 포함한다고 보장 못 함.
- **Resolution/What:** `project_signal_to_canonical_grid()`의 범위 밖 하드 `raise`를 제거(2줄) — 하류 루프가 이미 `np.searchsorted` clamp로 범위 밖 샘플을 안전 처리하도록 되어 있었음. Monotonic 체크는 유지. `min_common_active_bars` 가드는 그대로 안전장치로 유지.
- **Impact:** 실측 — pruning 사상 최초로 실제 작동(`pruning_applied=True`, 72개 중 33개(46%) 중복 강등, 예측치 34/72와 일치). 부수 발견: pruning이 처음 완주하며 O(n²) pairwise 재투영(캐시 미사용, `compute_cross_tf_pair_evidence`가 `proj_cache` 재사용 안 함) 비용이 노출됨 — L0 gate 자체는 157s(정상)인데 cross-TF 단계가 ~640s 추가 소요, 후속 최적화 과제로 별도 분리.

## [2026-07-12] [TASK_L0_MEMORY_BOUND_DATAFLOW] [ADR_20260712_L0_MEMORY_BOUND_DATAFLOW]
- **Context/Why:** LTF 1분 데이터(exec_1m) 전량 적재 시 상주 메모리(RSS)가 심볼 수에 비례하여 급증하여 OOM-killer가 발생함.
- **Resolution/What:** 전역 exec_1m 맵 적재를 완전히 제거하고, LTF 스트리밍 경로에서 필요한 심볼의 1분 데이터를 순차적/제한적으로 로드하도록 구현(bounded 1m reader).
- **Impact:** RSS 메모리 사용량 16,438 MiB에서 3,649 MiB로 77.8% 획기적으로 절감함.

## [2026-07-12] [TASK_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION] [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION]
- **Context/Why:** 데이터 지원이 불충분한 LTF 가설의 진입 제어 및 trades 컬럼 유실 parquet 스키마로 인한 데이터 파이프라인 중단 방지 필요.
- **Resolution/What:** bridge -> coverage plan -> streaming LTF panel 연결 구조 강화 및 trades 컬럼 누락 parquet에 대한 optional 스키마 처리.
- **Impact:** L0 실행 시간 171.2초에서 20.24초로 88.2% 단축 및 L0 alpha gate 정상 검증 통과함.

## [2026-07-12] [TASK_L0_GATE_EARLY_EXIT_OPTIMIZATION] [ADR_20260712_L0_GATE_EARLY_EXIT_OPTIMIZATION]
- **Context/Why:** L0 cheap gate에서 이미 기각이 확정된 후보들에 대해 canonical gate의 무거운 중복 연산(Bootstrap LCB, Triple Barrier 등)이 반복되어 불필요한 리소스 낭비 및 latency 유발.
- **Resolution/What:** `evaluate_alpha_gate_batch` 시그니처에 `cheap_evidences` 인자를 추가하고, cheap gate 탈락 후보는 즉시 `_empty_gate_evidence`를 반환하도록 조기 탈락(Early-Exit) 구현.
- **Impact:** 실측(sequential) — Phase 3 canonical gate **96.74s→75.99s(-21.4% 단축)**, L0 전체 **193.39s→171.21s(-11.5% 단축)**. 정합성 100% 동일 및 E2E 통과.

## [2026-07-12] [TASK_L0_GATE_EVENT_FILTERING_OPTIMIZATION] [ADR_20260712_L0_GATE_EVENT_FILTERING_OPTIMIZATION]
- **Context/Why:** L0 gate(`cheap-gate` & `canonical-gate`) 평가 시, 매 panel마다 전체 time과 symbol에 대해 불필요하게 sparse 이벤트를 전량 추출하는 `candidate_panels_to_events` 내부 연산 및 메모리 낭비 병목 확인.
- **Resolution/What:** 상류에서 확정된 `event_mask`를 `panel.metadata["l0_event_mask_2d"]`에 임시 주입하여, `candidate_panels_to_events` 내부에서 필요한 이벤트들만 희소 필터링하도록 최적화. 호출 후 `try-finally`로 메타데이터에서 제거.
- **Impact:** 실측(sequential) — **Phase 1 135.94s→96.64s(-29% 단축)**, L0 전체 **236.63s→193.39s(-18.3% 단축)**. 정합성 100% 동일(8h n_ready=53, 12h n_ready=98, 2h n_ready=19, gate_passed=True). 대규모 메모리 할당 방지로 OOM 위험 차단.

## [2026-07-12] [TASK_L0_PHASE1_CHEAP_GATE_DEDUP] [ADR_20260711_L0_PHASE1_CHEAP_GATE_DEDUP]
- **Context/Why:** Phase1/Phase3 분리 계측(직전 ADR) 결과 Phase3(236.63s 중 100.69s)가 4-worker 병렬화했는데도 여전히 큼 → 코드 추적으로 Phase1(`evaluate_alpha_cheap_gate_batch`, evidence_by_tf 구축용)과 Phase3(`run_alpha_foundry_l0_pipeline` 내부)가 완전히 동일한 입력으로 같은 순수/결정론적 함수를 중복 호출 중임을 확인. `docs/specs/l0_phase1_cheap_gate_dedup.md`.
- **Resolution/What:** `precomputed_cheap_evidences`(additive, keyword-only, 기본 None=기존 재계산 동작) 파라미터를 `run_alpha_foundry_l0_pipeline`→`run_alpha_foundry_l0_gate`→Phase3 호출부까지 전체 스레딩. `build_cheap_gate_evidence_frame_from_evidences()` 신규 추출(DataFrame 투영 로직 분리). check 단계에서 Scenario 4 회귀 테스트의 mocking 타겟 오류(`cheap_gate` 모듈 속성만 패치, `pipeline.py`의 module-level import는 미교체되어 회귀를 못 잡는 상태) 발견·수정, 실제로 dedup을 임시로 깨서 수정된 테스트가 잡아내는지 실증까지 완료.
- **Impact:** 실측(`L0_PARALLEL_MAX_WORKERS=4`) — **Phase3 100.69s→56.65s(-44%)**, L0 게이트 전체 236.63s→197.93s(-16.4%), 전체 파이프라인 523.11s→498.91s(-4.6%), 이번 세션 원본 baseline(741.22s) 대비 누적 **-32.7%**. n_ready(53/98/19)/gate_passed 전부 baseline과 동일. 스펙의 "~387-410s" 예측은 낙관적이었음(Phase3의 44%만 순수 중복, 나머지 56%는 canonical gate/diversity/budget 등 필요 작업으로 실측 확인) — 정직하게 재보정. SSOT: `docs/architecture/layer0.md` §Phase-1/Phase-3 Cheap-Gate Deduplication.

## [2026-07-12] [TASK_L0_L1_PIPELINE_LATENCY_PROFILING] [ADR_20260711_L0_L1_PIPELINE_LATENCY_PROFILING]
- **Context/Why:** 실측(`4h_1783781808`, 741.22s) 로그 분해 결과 L0 게이트(272.87s, TF 6개 완전 순차·내부 병렬처리 전무)가 확실한 병렬화 대상으로 확인됨; L1은 TF당 이미 ProcessPoolExecutor로 8코어 포화 중이라 병렬화 금지 대상으로 명시. `docs/specs/l0_l1_pipeline_latency_profiling.md`.
- **Resolution/What:** `run_alpha_foundry_l0_gate_multi_tf`에 additive `parallel_max_workers`(fork mp_context + prefork COW 캐시 `_L0_TF_INPUT_CACHE`) 추가, 시그니처/반환타입 불변 유지. `panel_construction`/`tf_probe_scoped` 신규 타이밍 계측 추가. check 단계에서 발견한 배선 누락(bridge.py가 `parallel_max_workers` 미전달로 기능 완전 비활성) 및 로거 가시성 버그(3번째 재발, `_logger`→`_run_logger`) 수정.
- **Impact:** 실측(`L0_PARALLEL_MAX_WORKERS=4`) — **전체 741.22s→547.87s(26% 단축)**, L1 결과(n_ready 53/98/19, gate_passed) baseline과 완전 동일 확인, peak RSS 16,717MB→16,396MB(오히려 소폭 감소). 신규 계측이 `panel_construction`(34s)/`tf_probe_scoped`(5.75s)를 드러냈으나 합계 40s뿐 — 이전 "미계측 283s"의 주 원인이라던 가설은 **반증**됨. 여전히 상당한 미계측 구간 잔존, 3차 계측 라운드 필요(미해결). SSOT: `docs/architecture/layer0.md` §Phase-3 Cross-TF Parallel Execution, `docs/results/result.md`.

## [2026-07-12] [TASK_L0_CROSS_TF_PRUNING_ADMISSION] [ADR_20260711_L0_CROSS_TF_PRUNING_ADMISSION]
- **Context/Why:** Cross-TF 독립성 감사가 읽기 전용이라 L1이 72개 중 34개 known-redundant 후보에도 전체 walk-forward compute를 소모(`docs/specs/l0_cross_tf_pruning_admission.md`). check 단계에서 치명적 순서 버그(pruning 계산 후 `multi_results` 재할당이 `base_result`/`project_htf_panels_to_base` 소비 시점보다 늦어 무효화) 및 survival-floor set-membership 카운팅 버그 발견.
- **Resolution/What:** `apply_cross_tf_survival_floor`/`assemble_l0_strategy_delivery_manifest`(additive, `run_alpha_foundry_l0_gate_multi_tf` 시그니처 불변) 신규. bridge.py 호출 순서를 `base_result` 이전으로 이동해 순서 버그 수정, `Counter` 기반 카운팅으로 floor 버그 수정, 로거를 `setup_logger("opt_main_futures")`로 교체(모듈 로거 미노출 재발 방지), `total_l1_verification_budget` 하드코딩 제거.
- **Impact:** 실측(`4h_1783781808`, `L0_CROSS_TF_PRUNING=1`) — **1h 후보 존재 시 canonical_tf=4h가 `compute_cross_tf_redundancy`의 LIMIT-02(canonical은 모든 입력 TF보다 세밀해야 함) 가드에 걸려 실패**, fail-open으로 정상 폴백(L1 결과 baseline과 완전 동일, `gate_passed=True`, 741.22s). Pruning 자체는 아직 실전 미적용 상태 — canonical TF 선택 전략 재설계가 다음 과제. SSOT: `docs/architecture/layer0.md` §Cross-Timeframe Diversity Audit & Pruning Admission.

## [2026-07-11] [TASK_L0_STRATEGY_DELIVERY_HARDENING] [ADR_20260711_L0_STRATEGY_DELIVERY_HARDENING]
- **Context/Why:** L0 diversity dedup은 TF별 독립 호출이라 cross-TF 중복을 전혀 못 봄; 78개 selected_for_l1 후보 중 진짜 독립 알파 수는 미측정 상태였음(`docs/specs/l0_strategy_delivery_hardening.md`).
- **Resolution/What:** `project_signal_to_canonical_grid`/`compute_cross_tf_redundancy`/`audit_l0_selected_recipe_independence`(diversity.py) + `L0IndependenceAudit`/`L0StrategyDeliveryManifest`(contracts.py) 신규, `bridge.py`에 opt-in 배선(`enable_cross_tf_diversity_audit`, env `L0_CROSS_TF_DIVERSITY_AUDIT`). 배선 중 발견한 3개 별도 버그(모듈 로거 DEBUG 미노출, `panels_for_l1` recipe_id 메타데이터 누락, canonical TF 선택 오류)도 함께 수정. `empty_opportunities` locus 분리, 1h/2h widened pool(`l1_ltf_family_pool_widened`) A/B knob도 추가.
- **Impact:** 실측(`4h_1783775628`) — **72개 selected_for_l1 중 진짜 독립 클러스터는 38개(53%)**, 34개는 `btc_regime_pullback` 등 동일 테제의 TF 간 재측정으로 확인(가설 확정). SSOT: `docs/architecture/layer0.md` §Cross-Timeframe Diversity Audit, `docs/architecture/layer1.md` §Outer-Fold Opportunity Blocker Loci, `docs/results/result.md`.

## [2026-07-11] [TASK_L0_NAN_COST_HTF_BLIND_REJECTION] [ADR_20260711_L0_NAN_COST_HTF_BLIND_REJECTION]
- **Context/Why:** `AlignedMarketData.execution_cost_bps_2d`가 소스 컬럼 없을 시 `None`이 아니라 전량 NaN 배열로 기본초기화됨. `has_cost_2d = ... is not None`이 NaN을 유효로 오판 → 비-4h(및 일부 4h) 패널의 net edge가 전량 NaN 오염, `net_lcb_bps`/`nw_tstat`가 0.0으로 폴백되며 게이트가 실제 알파 유무와 무관하게 100% 자동기각(`non_positive_lcb`/`weak_tstat` 상시 발동, 수학적 확정).
- **Resolution/What:** `_is_usable_cost_array()`(NaN-aware) 도입, `compute_triple_barrier_returns`/`label_candidate_events` 양쪽 동일 버그 지점 수정. 진단 로깅 4곳 추가 중 모듈 로거가 실제 파이프라인에서 DEBUG 미노출되는 별도 이슈 발견 → `_ensure_debug_visible()`(opt-in 시 자체 레벨/핸들러 강제)로 견고화, `evaluate_panel_gate`→`compute_triple_barrier_returns` 플래그 배선 완료(`align_data_maps` 배선은 상류 다계층 관통 필요해 후속 과제로 보류).
- **Impact:** 실측(`--phase l1 --timeframe 4h`, 742개 진단 로그 확보) — **NaN 오염 recipe 0건(edge_finite=1.000 전량)**. gate_passed 후보 16(4h만)→78(전 TF), L1 최종 게이트 사상 최초 `PASSED`(8h n_ready=53, 12h n_ready=98, 2h n_ready=19). 수 주간 반복된 "1h/2h/6h/8h/12h gross alpha 부재" 결론이 가짜 음성이었음을 raw evidence 값 레벨까지 완전 실증. SSOT: `docs/architecture/layer0.md` §Cost Array Usability Guard.

## [2026-07-11] [TASK_L0_HTF_RESAMPLE_ALIGNMENT_FIX] [ADR_20260711_L0_HTF_RESAMPLE_ALIGNMENT_FIX]
- **Context/Why:** 2h/6h/8h/12h는 네이티브 데이터가 없어(`data/futures/`에 1h/4h/1d만 존재) 1h를 리샘플한 합성 캔들로 L0 게이트를 평가해왔음. `_resample_probe_source_frame`/`_resample_ohlcv`가 `closed="right",label="right"`(틀린 컨벤션) 사용 — 라이브 Binance 6h fetch와 로컬 리샘플을 직접 대조해 `closed="left",label="left"`가 정답임을 실측 확정(byte-identical).
- **Resolution/What:** 두 함수 모두 open-time 컨벤션으로 정정, 위치 기반 `iloc[:-1]` 완결성 판정을 표본개수 기반(`infer_source_bar_hours` mode 추론 + ratio 비교)으로 교체. 회귀 80/80 PASS, 라이브 스냅샷 고정 테스트 추가.
- **Impact:** 실측(`--phase l1 --timeframe 4h`, 2026-07-11 재실행) — 4h/1h는 완전 불변(회귀 없음, 예상대로). baseline에서 6h/8h/12h 3개 TF가 완전 동일했던 reject-reason이 12h만 갈라짐(`15,15,15,4`→`16,16,16,2`)해 버그가 real이었음을 확증. 단 **6h/8h는 수정 후에도 여전히 완전 동일**(별도 원인 의심, 미해결) — 2h/6h/8h/12h 전부 `gate_passed=0` 유지, 새 알파는 아직 미발견. SSOT: `docs/architecture/layer0.md` §Non-Native Timeframe Synthesis.

## [2026-07-11] [TASK_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION] [ADR_20260711_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION]
- **Context/Why:** L0 4h 13개 pooled systematic 후보(net_lcb 15~97bps, 8 family)가 L1 nested-pairwise 원자화 게이트에서 0 qualified로 소멸. `peer_exclusive` incremental 테스트가 상관된 systematic 신호를 상호 카니벌리제이션할 가능성 가설.
- **Resolution/What:** Phase 0(`diagnose_strategy_atomization`, log-only) 실측으로 가설 확정(13/13 pooled_gross>0, dominant_reject=no_incremental_edge 만장일치). Phase 1(`compute_xs_factor_spread_diagnostics.xs_archetypes` 일반화 + `l1_pooled_admission_archetypes=("xs_alpha","trend","ts_mom")`)로 9/13에서 no_incremental_edge 해소 확인, 표본적정성 게이트는 그대로 보존됨(atomized_median==pooled_gross로 안전 확인).
- **Impact:** 메커니즘은 설계대로 정확히 동작 검증됐으나, L1 최종 게이트는 여전히 `BLOCKED`(0/5) — walk-forward outer-fold `empty_opportunities`(Fold#1~3 대부분 Symbols=0/Events=0, Phase 0/1 양쪽 동일 22건)가 새로운 상류 병목으로 확인됨, 별도 후속 과제로 분리. 신규 필드/함수는 기본값 비활성(`False`/`("xs_alpha",)`) 유지로 하위호환.
