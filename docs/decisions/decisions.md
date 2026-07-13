# Active Decisions Log (Sliding Window)

## [2026-07-13] [TASK_L0_READINESS_HARDENING] [ADR_20260713_L0_READINESS_HARDENING]
- **Context/Why:** L0 준비도 실측 59/100(`docs/results/result.md`) — 4h/6h/1h L1 봉쇄 원인 로그 부재, `DEPRIORITIZED_FAMILY_PRIOR`가 실측 통과 중인 2개 family와 모순, `vol_breakout` 전 TF 미검증, `cross_tf_pruning` 기본 비활성으로 87개 중 76개가 11개 family로 중복.
- **Resolution/What:** `[L1-PERTF-REGISTRY-DIAG]`에 `gate_report.blockers` 필드 추가, `DEPRIORITIZED_FAMILY_PRIOR`에서 `vol_term_structure_gate`/`trend_donchian` 제거, `_DEFAULT_PER_TF_FAMILIES`(6h/8h/12h)에 `vol_breakout` 추가, `_l0_cross_tf_pruning_enabled()` 기본값 True(opt-out)로 반전.
- **Impact:** 실측(`4h_1783923826`) — blockers 로그로 4h(`match_ratio:0.500,fold_ratio:0.250`)/6h(`match_ratio:0.750`)/1h(`sym_count:1.600,fold_ratio:0.000,probe_lcb_bps:-inf`) 최초 확인, 근본원인이 L0 후보 품질이 아닌 L1 검증 임계값임을 규명. `vol_breakout` 6h/8h/12h 전부 통과(net_lcb 78.1/81.7/92.9bps, 최상위권), family 11→12개. `cross_tf_pruning`은 최초로 fail-open 아닌 `status=applied` 실행됐으나 selected_for_l1 중복 패턴 불변(하류 매니페스트 미배선) 및 전체 wall-clock 282.63s→337.84s(+19%) — 컴퓨트 절감 효과 미입증, 후속 재검토 필요. L0 준비도 59→62/100. SSOT: `docs/results/result.md`, `docs/architecture/layer0.md`.

## [2026-07-13] [TASK_L0_L1_SPEED_OPT] [ADR_20260713_L0_L1_SPEED_OPT]
- **Context/Why:** L0 cheap gate Phase 1 (162.50s) 및 Phase 3 (20.70s) 내부의 Pandas `.iloc` 기반 Python loop 인덱싱 오버헤드로 인한 속도 저하. L0 Phase 1이 타임프레임별 순차 평가되어 다중 코어가 미활용됨.
- **Resolution/What:** 이벤트 정렬/필터링을 NumPy vectorized indexing으로 교체하여 파이썬 인터프리터 연산 오버헤드 제거. L0 Phase 1에 `ProcessPoolExecutor (fork)` 및 global COW 캐시 기법 기반 per-timeframe 병렬 처리 도입. 가용 코어 수 자동 탐색 적용.
- **Impact:** L0 Phase 1 소요 시간 **162.50초 ➡️ 24.43초 (85.0% 단축)**, STRATEGY 전체 실행 시간 **410.34초 ➡️ 282.63초 (31.1% 단축)** 달성. E2E 검증 수치 및 최종 승인 100% 일치.

## [2026-07-13] [TASK_FUTURES_DATA_LAYOUT_OPT] [ADR_20260713_FUTURES_DATA_LAYOUT_OPT]
- **Context/Why:** 선물 시계열 데이터 저장 공간(11GB) 및 백테스팅 데이터 로드 I/O 속도 최적화 필요. `datetime` 중복 적재로 인한 역직렬화(로드) 속도 저하 및 단순 `snappy` 압축으로 인한 디스크 낭비 식별.
- **Resolution/What:** `FuturesStorageLayout` 신규 도입하여 `ohlcv/enriched/funding/metrics/metadata` 폴더 트리로 파티셔닝(오토 마이그레이션 지원). OHLC 가격 `float32` 다운캐스팅, `datetime` 필드 디스크 제거 및 메모리 내 vectorized 동적 복원 적용.
- **Impact:** 전체 데이터 용량 **28.2% 절감(11GB ➡️ 7.9GB)** 및 I/O 성능 **1.46배 향상**. 백테스팅(E2E L1) 수치 정합성 및 최종 프로모션 완전 일치 실증 완료.

## [2026-07-13] [TASK_L0_MTF_FUSION_PERF_OPT] [ADR_20260713_L0_MTF_FUSION_PERF_OPT]
- **Context/Why:** mtf_fusion 실측 후 panel_construction +266%(27s→99s) 확인. 마이크로벤치마크로 `_htf_hma_slope_filter`가 `_weighted_moving_average_2d`를 `np.apply_along_axis`로 불필요하게 심볼별 래핑(126회 재호출)하는 게 주범(3.0s/call)임을 발견 — 함수 자체는 이미 2D 벡터화되어 있었음. `_htf_adx_dmi_filter`/`_htf_ichimoku_cloud_filter`도 심볼별 Python loop로 dispatch.
- **Resolution/What:** `apply_along_axis` 래퍼 제거(함수 불변, 109x). ADX/Ichimoku를 멀티컬럼 시그니처로 재작성 + `_resample_ohlc_to_htf_and_project()` 신규(searchsorted 1회 배치, 16.7x). **버그 발견 및 수정**: 실측 재검증(`docs/results/result.md` 대비) 중 `net_lcb_bps` 최대 5.8bps 불일치 확인 — `_htf_ichimoku_cloud_filter`의 `np.maximum`/`np.minimum`이 pandas `.max(axis=1)`(skipna=True) 대비 NaN 전파, 클라우드 신호 발생 26기간 지연. `np.fmax`/`np.fmin`으로 교체, 회귀 테스트의 "reference" 구현도 동일 버그를 갖고 있어 못 잡았던 것 확인 후 fix. `check` 89/89 PASS.
- **Impact:** 실측 — panel_construction **98.68s→33.10s**, 전체 wall-clock **619.59s→547.58s**. 수정 후 재검증: `gate_passed`/`selected_for_l1` 100% 동일, `net_lcb_bps` 최대오차 5.8bps→0.13bps(잔차는 부동소수점 비결합성, 게이트 판정 무영향). SSOT: `docs/results/result.md` §6.

## [2026-07-13] [TASK_L0_MTF_FUSION_FACTORY] [ADR_20260713_L0_MTF_FUSION_FACTORY]
- **Context/Why:** L0 archetype 감사 결과 gate_passed의 96~100%가 `trend` 단일 archetype으로 수렴, 유일한 고성과 패턴(HTF필터×LTF트리거 MTF융합)이 3개 하드코딩 family로만 존재. 지표 확장 2차 검토(Stochastic/일목/HMA/ADX 등) 결과 필터5종×트리거4종 조합이 근거 확보됨(`docs/specs/l0-mtf-recipe-factory.md`).
- **Resolution/What:** `rule_signals.py`/`signals/rules.py`에 `mtf_fusion` family 신규 추가(양쪽 `ALL_SIGNAL_FAMILIES` 동기화) — HTF 필터(ema_slope/macd_cross/hma_slope/ichimoku_cloud/adx_dmi) × LTF 트리거(rsi_band/macd_cross/donchian_retest/stochastic_cross) 조합 팩토리. `config.py` `_DEFAULT_PER_TF_FAMILIES`(4h/6h/8h/12h)에 편입. `check` 81/81 PASS, ruff+mypy clean.
- **Impact:** 실측(`4h_1783901398`) — mtf_fusion 180개 조합 중 177개(98.3%) gate 통과, net_lcb 최고 107.2bps. 부수효과로 6h/8h/12h의 diversity dedup 미작동 미스터리 해소(후보 밀도 부족이 원인이었음 확인). 트레이드오프: wall-clock +41%(439.81s→619.59s, 주범은 ichimoku/adx 필터의 심볼별 Python loop), 8h/12h `n_ready` 소폭 감소(53→44, 98→92, 원인 미확정, 후속 조치 대상).

## [2026-07-12] [TASK_L0_GATE_PIPELINE_OPTIMIZATION] [ADR_20260712_L0_GATE_PIPELINE_OPT]
- **Context/Why:** L0 gate 실측(Phase1 84-96s + Phase3 72-97s = 157-193s)에서 Phase 3 canonical gate가 Phase 1 cheap gate와 70% 중복 연산(triple-barrier/block_means/bootstrap/rank_IC/cost_drag/turnover)을 재수행. `aligned.symbols.index()` O(S) 호출 4회 및 중복 ATR Yang-Zhang vol 연산이 추가 부하. production runtime에서 cache path의 `rank_ic`가 cheap gate `_compute_rank_ic`의 NaN 미필터로 인해 `ValueError: numeric field must be finite, got nan` 발생.
- **Resolution/What:** O-1: `CheapGateEvidence`에 3개 optional dict(cheap_event_arrays/cheap_block_stats/cheap_meta_stats) 추가, `evaluate_panel_gate` cache path에서 6개 중복 연산 skip. cache path `rank_ic`는 NaN-safe `compute_rank_ic_with_tstat` 사용. O-2: `_symbol_map` O(1) dict 도입(4곳 aligned.symbols.index() 대체). O-3: `precomputed_atr_2d` 파라미터로 중복 ATR compute 회피. 벽시계 **alpha gate 20.56s→2.89s(-86%, 7.1×)** 안정적 완주. 170/170 PASS, ruff+mypy PASS, RSS 6.4GB(budget 10GB 이내).
- **Impact:** L0 게이트 Phase 3의 70% redundant computation 해소. `rank_ic` NaN은 `_compute_rank_ic`가 `compute_rank_ic_with_tstat`처럼 finite filtering을 하지 않아 발생 — cache path가 직접 spearmanr를 재계산하여 workaround. O-4(TF fusion hoist)/O-5(parallel Phase 1)/O-6(float32 memory)/O-7(stage rename)는 성능 예산이 2.89s로 충분해 후순위 보류.

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
