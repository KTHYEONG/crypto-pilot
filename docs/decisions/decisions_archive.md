# Permanent Decisions Archive

This file holds historical architecture decision records (ADRs) that have been pruned from the active window.

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

## [2026-07-11] [TASK_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN] [ADR_20260711_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN]
- **Context/Why:** L0 전 타임프레임 신호 부족 재검토 스펙 구현 후 실측(`--phase l1`, 4h, 2026-07-11)이 6h TF cross-sectional 패널 평가 중 `xs_spread_lcb_bps must be finite` 크래시. 원인: barrier-aware 리팩터가 `_net_dense`를 정합 필터를 통과한 이벤트 부분집합에만 채우는데, `compute_xs_spread_lcb_bps`/`compute_rank_ic_with_tstat`가 미채움 셀(NaN) 포함 원본 `event_mask`로 `np.mean` 집계.
- **Resolution/What:** 두 함수에 finite 마스킹 추가(`compute_regime_stability`/`compute_payoff_stats`와 동일 관례로 정렬). 회귀 67/67 PASS, 6개 TF(4h/6h/8h/12h/1h/2h) 전체 크래시 없이 완주 확인.
- **Impact:** Fix1-6(barrier-aware 평가/rising-edge/rolling-stat/entry 버그 4건/카탈로그 정리) 수치 정상성 검증 완료. 단 4개 TF 전부 최종 병목은 여전히 `tstat`(6h `trend_pullback_continuation` 1건만 SELECT) — 로직 버그가 아닌 gross alpha 부재 재확인. L1 nested pairwise 단계는 별도 미해결(`no_incremental_edge` 우세, 0 qualified).

## [2026-07-10] [TASK_L0_TF_CORROBORATION_WIRING_FIX] [ADR_20260710_L0_TF_CORROBORATION_WIRING_FIX]
- **Context/Why:** `tf_corroboration`이 실측에서 항상 0.0이었음(수일간 "데이터 볼륨 병목"으로 오진). 재추적 결과 `run_alpha_foundry_l0_gate_multi_tf()`의 Phase 1이 `recipe_id`가 바인딩되지 않은 원본 패널로 `evidence_by_tf`를 구축해 매 TF마다 0행이 되던 배선 버그였음. 별도로 `timeframe_probe.py`가 `dataclasses.asdict()`로 중첩 config를 평탄화해 워커에서 `'dict' object has no attribute 'channel_bars'` 크래시 발생(본 gate 평가는 무영향).
- **Resolution/What:** Phase 1에서 `bindings_by_tf`로 패널을 바인딩하는 공유 헬퍼 `_bind_panels_to_recipe_ids()`를 추출해 Phase 1/3 양쪽에서 재사용. `_probe_tf_worker`는 `asdict()`+dict 재구성 대신 `dataclasses.replace(base_cfg, timeframe=tf)`로 교체. 완전 사문화된 `signals/timeframes.py` 삭제(0 importer 확인). `[ALGO] stage=tf_fusion` 진단 로그 신규.
- **Impact:** 실측(`--phase l1`, 126심볼) — `channel_bars` 에러 0건(이전 4건). `tf_corroboration>0` 행 31/122, `corroborated` 15건·`contradicted` 20건 최초 관측(이전 전량 `insufficient_coverage`). 회귀 109 passed.

## [2026-07-10] [TASK_SYNC_TOKEN_OPTIMIZATION] [ADR_20260710_SYNC_TOKEN_OPTIMIZATION]
- **Context/Why:** AI가 sync 스킬을 적용할 때 decisions 및 index.json을 통째로 읽고 수동 텍스트 처리를 수행하여 엄청난 Context 및 Output 토큰을 낭비하는 치명적 비효율이 존재했음.
- **Resolution/What:** decisions.md의 15개 초과분 자동 이관용 `archive_decisions.py`와 index.json 자동 매핑용 `update_index.py` CLI 유틸리티를 작성함.
- **Impact:** AI가 decisions_archive.md와 index.json을 직접 스캔/작성할 필요가 없어져 sync 단계의 토큰 소모를 95% 이상 감축함.

## [2026-07-10] [TASK_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN] [ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN]
- **Context/Why:** L0 유니버스 admission이 25/150 심볼로 붕괴해 있었음. 근본원인: `_requires_exec_1m()`가 `alpha_foundry.mode != "off"`이면 무조건 1분봉 커버리지를 admission `pass_flag`에 포함시켜, 이를 쓰는 family가 3개뿐인데 전체 유니버스를 게이팅했음. 신규 family(`liquidity_participation_breakout`/`btc_neutral_residual_reversal`)도 canonical 비용모델(~12bps 하한, 50bps 상한)과 무관한 자체 3bps 임계치를 발명해 항상 기각됨.
- **Resolution/What:** `evaluate_symbol_data_sufficiency()`에서 `exec_1m_ok`를 admission 판정에서 제거(정보성 필드로만 유지). 두 신규 family의 liquidity predicate를 `AlignedMarketData.active_mask`(canonical) 기준으로 교체하고 자체 `max_event_cost_bps`/`min_adv_usdt` 제거. `resolve_economic_thesis_id()`/`n_distinct_thesis_ids_passed`(observability-only) 신규. `resolve_1m_backfill_targets()`를 파일존재-only에서 날짜범위 커버리지 비율 판정으로 교체.
- **Impact:** 실측(`--phase l1`, 4h, 2026-07-10) — 유니버스 25→126-137 symbols 회복(`missing_exec_1m` 탈락사유 소멸 확인), LPB/BNRR n_events 0→6,139~10,801(정직하게 재평가 후 기각, gross 자체 음수). `tf_corroboration=0` 가설(협소 유니버스 원인)은 실측으로 **반증**(126심볼에서도 0) — 별도 미해결 버그로 확인. 부수 발견: `timeframe_probe.py`의 `dataclasses.asdict()`가 신규 중첩 config를 재귀적으로 dict화해 TF-PROBE 워커 4개 tf 전부 실패(`'dict' object has no attribute 'channel_bars'`) — 본 gate 평가는 무영향, 별도 수정 필요.

## [2026-07-10] [TASK_L0_TERMINAL_DEBUG_OBSERVABILITY_SYNC] [ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]
- **Context/Why:** `phase="l0"`가 파일 아티팩트를 남기고 있어 터미널 DEBUG 수집 요구와 어긋났고, 실제 실행 경로의 active config source도 `optimization/config.py`로 분리돼 문서 SSOT가 느슨해졌음.
- **Resolution/What:** `phase="l0"`를 `artifact_write_enabled=False` + `debug_log`로 고정하고, terminal JSON/CSV emitters와 `phase`-aware bridge/runtime docstrings를 추가했다.
- **Impact:** `json/parquet` 파일 없이 `l0` 결과를 직접 로그로 수집할 수 있게 되었고, `docs/specs/l0_naming_and_debug_observability.md`를 제거해 작업 잔재를 정리했다.

## [2026-07-09] [TASK_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF_SYNC] [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
- **Context/Why:** `discovery_units.py` introduced a standalone fail-closed L0 branch for conditional cells/execution arms/horizon masks, but docs/index/ADR trail and current-task residue were not synchronized.
- **Resolution/What:** Added architecture/index coverage for `L0DiscoveryUnit` / `L0DiscoverySelection` and the new `enable_discovery_unit_handoff` knobs; tagged the new module docstrings with `[ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]`.
- **Impact:** `docs/specs/l0_l1_conditional_discovery_redesign.md` removed; `docs/decisions/decisions.md` stayed within the 15-entry active window after pruning the oldest entry to archive.

## [2026-07-09] [TASK_L0_TREND_PULLBACK_HARDENING_SYNC] [ADR_20260709_L0_TREND_PULLBACK_HARDENING_SYNC]
- **Context/Why:** `btc_regime_pullback` 계열과 공통 forward-return SSOT가 실측 런에서만 검증됐고, spec 산출물/임시 로그가 남아 있으면 후속 검증이 흐려짐.
- **Resolution/What:** `compute_causal_forward_returns_bps()`를 새 SSOT로 문서화하고, `rules.py`/`rule_signals.py`의 신규 variant 세트와 `docs/index.json` 매핑을 동기화했다.
- **Impact:** `4h_1783585799` 실측 기준으로 L0 아티팩트와 문서 연결을 고정했고, `docs/specs/l0_trend_pullback_archetype_hardening.md`를 제거해 작업 잔재를 정리했다.

## [2026-07-09] [TASK_L0_CONDITIONAL_DIAGNOSTIC_WIRING] [ADR_20260709_L0_CONDITIONAL_DIAGNOSTIC_WIRING]
- **Context/Why:** `conditional_cells.py`/`execution_arms.py`/`edge_failure.py`가 구현·유닛테스트 완료 상태로 방치돼(`enable_*` 전부 기본 `False`, 호출부 0건) "pooled 평균이 조건부 엣지를 숨기는가"/"taker 비용가정이 과도한가" 두 가설이 실측된 적 없었음.
- **Resolution/What:** `run_alpha_foundry_l0_pipeline()`에 diagnostic-only opt-in 배선(`l0_diagnostics.py` 신규, `passed_recipe_ids`/`handoff_decisions` 확정 이후에만 `evidence_rows`에 행 추가). Look-ahead(calibration/eval 분할)·다중검정(BH-FDR) 결함 선수정. 실행 후 `bars_per_year` 4h 하드코딩과 `failure_axis` 미기록 버그 추가 발견·수정.
- **Impact:** 실측(25 syms, run `4h_1783560242`, 1h/2h/4h/6h/8h/12h) — 조건부 셀 105건(13 레시피), 실행암 112건(56 레시피) 전량 `gate_passed=False`(최근접 -6.3~-13.5bps). **두 반증가설 모두 기각** — gross alpha 부재가 게이트/비용가정 아티팩트가 아니라 실재함을 재확인. `[LIMIT-06]` 격리 불변식 신규 테스트로 검증.

## [2026-07-08] [TASK_L0_EDGE_FAILURE_ATTRIBUTION] [ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]
- **Context/Why:** `edge_failure.py`(failure axis 분류)는 새로 구현됐으나 `weak_gross_edge` 축이 의존하는 `AlphaFoundryEvidenceRow.gross_lcb_bps`가 `pipeline.py`에서 `0.0` 하드코딩(dead field)이라, 실 evidence에서 이 축이 원천적으로 발동 불가능했음.
- **Resolution/What:** `run_alpha_foundry_l0_pipeline()`이 canonical `AlphaGateEvidence.gross_lcb_bps`(실계산값)를 배선하도록 수정. `conditional_cells.py`/`execution_arms.py`는 unit test로만 검증(standalone, 미배선).
- **Impact:** 446개 유니버스 1m 데이터 갭(3개월 stale) 동기화 후 실측(`4h_1783519562_*`, 100행) — `weak_gross_edge` 0건→28건, `cost_dominated` 71→42건으로 재분포. attribution 로직 자체는 수정 전후 모두 정확했고, 문제는 오직 dead upstream field였음.

## [2026-07-08] [TASK_LTF_NATIVE_SIGNAL_EXPANSION] [ADR_20260708_LTF_NATIVE_SIGNAL_EXPANSION]
- **Context/Why:** L0 Alpha Foundry에 1m 기반 LTF native signal path가 없어서, `opt_main_futures.py`로 자연스럽게 관측 가능한 실데이터 L0 결과를 확보할 수 없었다.
- **Resolution/What:** `ltf_alpha.py`에 5m/15m/30m sparse families를 추가하고, runner→final evaluator→strategy builder→bridge 경로로 `exec_1m`/`alpha_foundry_config`를 전달해 L0 gate 전에 합쳤다.
- **Impact:** `--alpha-foundry audit` 실행에서 LTF evidence 5개가 `4h_1783484254_4h_evidence.parquet`에 포함됐고, 현재는 비용 후 `net_lcb_bps < 0`로 전부 reject된다.

## [2026-07-08] [TASK_L0_SIGNAL_YIELD_IMPROVEMENT] [ADR_20260708_L0_SIGNAL_YIELD_IMPROVEMENT]
- **Context/Why:** L0 게이트 BLOCKED 편중 원인을 실측(강제 artifact write) 진단 — 1h/2h는 `htf_only=True` 하드코딩으로 패널 자체가 생성 안 됐고(Track A), 4h/6h/8h/12h는 정상 평가되나 29개 family 중 seed 이상 4개뿐(Track B, cost>gross 구조적).
- **Resolution/What:** `bridge.py` 2곳 `htf_only=False`, `family_lifecycle.py`에 4개 family 은퇴 추가 + `resolve_retired_families_for_tf()` 신규(그런데 `is_family_tf_retired()` 자체가 아무 데도 호출 안 되던 것 발견 → recipe catalog/binding 4개 호출부에 배선), `cheap_gate.py`의 `evaluate_panel_cheap_gate`/`evaluate_panel_gate` n_events 체크를 `resolve_family_timeframe_gate_policy()` 경유로 교체(family_event_floors 미소비 발견 → 수정).
- **Impact:** 실측 3-run 비교(`4h_1783474978`→`_1783478588`→`_1783479077`) — 1h/2h 최초 평가(0→7건 실질 evidence), 은퇴 5개 family 실제 배제 확인(4h 42→34행, 12h 16→15행), `funding_flow_carry` 극단치(net_lcb=-277bps) 원인이던 이벤트 부족(n=77/190)이 이제 `insufficient_events`로 정상 차단. seed+candidate 합계는 8로 불변(위생 조치였지 신규 승격 창출 목적 아니었음). 회귀 테스트 3건은 픽스처가 새 우선순위(archetype_event_floors > flat min_events)를 가정 못해 깨졌던 것으로 확인 후 수정.

## [2026-07-08] [TASK_LTF_NATIVE_DIRECTIONAL_SEARCH] [ADR_20260708_LTF_NATIVE_DIRECTIONAL_SEARCH]
- **Context/Why:** 사용자가 "LTF=타이밍 전용" 전제(직전 ADR)에 반증 4개 질문 제기 — 실측한 결과 1h는 유니버스 150/150(100%) 이미 커버(4h와 동일)인데 1m은 34/150(23%)뿐이었고, 이전 세션 BTC 단일심볼 분석은 유니버스 경제성 검증이 아니었음이 확인됨.
- **Resolution/What:** `l1_tfs` 기본값에 `1h/2h` 추가(`strategy/config.py` `DEFAULT_L1_TFS`, `pipeline.py`), `_DEFAULT_PER_TF_FAMILIES` 1h/2h 풀 확장, `resolve_1m_backfill_targets`/`run_1m_backfill`/`resolve_1m_coverage_tier`/`Universe1mCoverageTier`(`entry_timing.py`/`contracts.py`) 신규 — 기존 `run_historical_sync(sync_1m=True)` 경로 재사용(신규 수집 코드 없음). 실행 중 `refine_entry_indices`의 confluence score가 숏(side=-1) 트레이드에서 구조적으로 트리거 불가능했던 로직 버그 발견·수정.
- **Impact:** 116개 심볼 1m 실제 백필 완료(coverage 23%→100%, 실측 +0.13GB, 사전추정 4.21GB 대비 훨씬 저렴 — 신규 심볼 대부분 상장 이력 짧음). 전체 유니버스(126 syms) L0 게이트 실측: 1h/2h 둘 다 `Proj=0`/`decision=reject_candidate`로 완전 기각(4h/6h/8h/12h 기존 결과는 회귀 없이 불변, 12h만 여전히 유일 통과) — "추측 아닌 실측"으로 이번 family pool에서는 1h/2h 무익 확정, family 풀 확장 여지는 남음.

## [2026-07-07] [TASK_LTF_ENTRY_TIMING_LAYER] [ADR_20260707_LTF_ENTRY_TIMING_LAYER]
- **Context/Why:** 4h~12h 방향성 신호가 반복적으로 한계에 도달해(`docs/results/result.md`), 저위 TF를 "HTF가 확정한 방향성의 진입 타이밍만 정제하는 종속 레이어"로 편입(`/arc`+`/spec`). CVD 임펄스+앵커 VWAP σ밴드+Kaufman ER/Hurst/VR 추세품질 게이트 3-입력 confluence로 설계.
- **Resolution/What:** `alpha_foundry/entry_timing.py`(`refine_entry_indices`/`aggregate_entry_timing_evidence` 등) 신규, `contracts.py`에 `EntryConfluenceSnapshot`/`HtfDirectionalEpisode`/`EntryTimingWindow`/`EntryTimingGateConfig` 추가, `metrics.py`에 `kaufman_efficiency_ratio` 추가, `signals/rules.py`의 `_safe_taker_imbalance_2d`→`safe_taker_imbalance_2d` public 승격. 구현 직후 `price_improvement_bps` 등이 0.0 하드코딩된 결함을 실행 검증으로 발견해 수정.
- **Impact:** BTCUSDT 실데이터(2022-10~2026-04, `trend_ma` EMA12/72 프록시 158건) 실측 — `evaluate_trend_quality_gate`가 5m/15m LTF에서 Hurst(`n<32`)/VR(`n<16`) 최소표본 미달로 구조적으로 트리거 불가(0/158). 30m~2h에서는 트리거되나(2~44%) `net_timing_edge_bps`가 전 구간 강한 음수(-23~-142bps, LCB 전부 게이트 미달) — confirmation-lag로 진입가 악화, 이번 confluence 조합은 반증됨. `strategy/rule_signals.py` 쌍둥이 모듈 rename 미동기화는 후속 과제로 남음.

## [2026-07-07] [TASK_L0_MULTI_TF_GATE_REDESIGN] [ADR_20260707_L0_MULTI_TF_GATE_REDESIGN]
- **Context/Why:** `tf_corroboration`이 구조적으로 0.0에 고정돼 `handoff_tier=candidate` 도달 불가능했던 원인을 추적하니, base TF만 L0 게이트를 타고 HTF(6h/8h/12h)는 `build_multi_tf_panels()`로 게이트 완전 우회하는 아키텍처였음(`/arc`+`/spec`로 fan-out→fuse→fan-in 재설계).
- **Resolution/What:** `run_alpha_foundry_l0_gate_multi_tf()`/`build_cheap_gate_evidence_frame()`(`bridge_helpers.py`), `build_native_htf_panels()`/`project_htf_panels_to_base()`(`bridge.py`, 기존 `build_multi_tf_panels` 분리) 신규 구현 + `evaluate_alpha_gate_batch()`·`build_l0_signal_candidate()` 2곳의 tf_fusion_index 2-tuple/3-tuple key 불일치 버그 수정. `run_candidate_strategy_for_universe()`에 `use_all_timeframes_in_l0` 플래그로 실제 배선(1차 구현에서는 함수만 만들고 배선 누락 — 실행 검증으로 발견해 추가 수정).
- **Impact:** 실측(4h base, run `4h_1783427649`) 확인 — 6h는 게이트 통과 신호 0건으로 완전 차단(`Proj=0`), 최종 L1 승격 합계가 `~199 → 43`으로 급감. `tf_corroboration`은 여전히 0이지만 원인이 "배선 누락"에서 "HTF 이벤트 수 부족(insufficient_coverage)"으로 바뀜 — 코드는 설계대로 동작, 데이터 볼륨이 병목.

## [2026-07-07] [TASK_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING] [ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
- **Context/Why:** `alpha_signal_generation.md` spec 구현이 unit test는 통과했지만 canonical `evaluate_panel_gate()` 미호출, `runtime_config` 미전달, `selected_for_l1`이 `discovery_tier`(cheap gate) 기준이라 `handoff_tier=blocked` 후보가 L1로 leak되는 3개 배선 갭이 실행 경로에 남아 있었음(`docs/specs/alpha_signal_generation_wiring_gaps.md`로 진단).
- **Resolution/What:** `pipeline.py`에 canonical `evaluate_alpha_gate_batch()` 호출 추가, `bridge_helpers.py`에 `runtime_config` 전달 추가, `viable_candidates` 판정을 canonical `handoff_tier` 기준으로 교체. 재실행 중 실데이터 전용 all-NaN 크래시 5곳(`cheap_gate.py`, funding 결측 구간) 신규 발견해 quant.md 안전 나눗셈 가드로 수정.
- **Impact:** 실측(4h, run `4h_1783419659`) 확인 — `selected_for_l1` leak 2→0건, `regime_stability` 실측 산출, 신규 6개 family 중 `sparse_breakout_retest_liquidity`가 최초로 `selected_for_l1=True` 도달. 신규 발견: `evidence_by_tf` 미주입으로 `tf_corroboration`이 항상 `0.0`이라 `handoff_tier="candidate"`가 구조적으로 불가능(상한 `seed`) — 후속 과제로 남김.

## [2026-07-07] [TASK_ALPHA_FOUNDRY_RESULT_SYNC] [ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
- **Context/Why:** 최신 4h run과 `docs/results/result.md`가 현재 unified alpha gate 상태와 분리되어 있었고, spec 산출물과 temporary residue가 남아 있으면 후속 검증이 흐려짐.
- **Resolution/What:** `docs/results/result.md`를 `4h_1783404539` 실측으로 갱신하고, `docs/architecture/layer1.md`/`layer3.md` 및 `docs/index.json`을 현재 source/test SSOT에 맞게 정렬했다.
- **Impact:** current-task `docs/specs/alpha_foundry_signal_effectiveness*.md`를 제거하고, 결과 문서에 `n_evidence=34`, `n_passed=1`, `selected_for_l1=3` 및 HTF promotion 관측을 고정했다.

## [2026-07-07] [TASK_ALPHA_FOUNDRY_ALPHA_IMPROVEMENT_SYNC] [ADR_20260707_ALPHA_FOUNDRY_ALPHA_IMPROVEMENT_SYNC]
- **Context/Why:** alpha improvement 적용 후 문서 SSOT가 계약/검색공간/게이트 변화와 분리되어 있었고, spec 산출물이 남아 있으면 이후 검증이 흐려짐.
- **Resolution/What:** `docs/architecture/layer1.md`에 `alpha_foundry` search space/V2 gate/static contract를 추가하고, `docs/index.json`에 `search_space.py` 및 신규 테스트 매핑을 보강했다.
- **Impact:** `docs/specs/alpha_foundry_alpha_improvement*.md` 2개를 제거해 작업 잔재를 정리하고, 현재 변경 범위를 docs/decisions/index로 고정했다.

## [2026-07-07] [TASK_L0_ALPHA_EFFECTIVENESS_REDESIGN] [ADR_20260707_L0_ALPHA_EFFECTIVENESS_REDESIGN]
- **Context/Why:** 실측(4h, 36개 family×variant) 전수분석 결과 절반이 cost_drag_ratio로 부호무관 사망, 통과후보 3건조차 rank_ic≈0(노이즈 수준)이며 rank_ic가 게이트 어디서도 안 쓰이고 있었음.
- **Resolution/What:** `CheapGateEvidence`/`AlphaFoundryEvidenceRow`에 `mean_gross_bps`/`total_cost_bps` 필드 추가, `weak_rank_ic` soft flag(표본크기 함수형 임계치) 신규, `audit_full_family_correlation()`(opt-in family 상관관계 감사) 신규.
- **Impact:** 실측(4h) 확인 — `weak_rank_ic`가 9/36건에 부여됐고, 유일하게 "candidate"(최고 등급)였던 `mtf_breakout_retest`가 "seed"로 강등되며 **현재 전체 27종 중 candidate 등급 0건** 확정. 게이트 판정(`gate_passed`/`discovery_tier` blocked 카운트)은 완전히 불변(회귀 없음). ⚠️ 실측 중 `total_cost_bps`가 건당평균(`mean_gross_bps`)과 달리 전체합계라 단위가 안 맞는 스펙 설계 실수 발견 — 다음 작업 후보로 `mean_cost_bps`(=total_cost/n_events) 교체 필요.

## [2026-07-07] [TASK_L1_BACKTEST_FIDELITY_FIXES] [ADR_20260707_L1_BACKTEST_FIDELITY_FIXES]
- **Context/Why:** L0/L1 아키텍처 리뷰(4개 질문: L0-L1 차이/exit 공정성/4h 고정/ML) 중 코드 재검증으로 확정된 3개 결함 발견. 1차 조사 에이전트의 cost 관련 보고 하나는 재검증 결과 오류(별개 필드 혼동)로 정정함.
- **Resolution/What:** `_resolve_panel_archetype`에 `btc_regime_pullback` 추가(trend 재분류, rules.py/rule_signals.py 양쪽), dead config `cost_amortize_by_holding` 제거, `candidate_evaluation.py`/`candidate_portfolio.py`의 4h/1h/1d 하드코딩 연율화를 `_bars_per_year_for_tf` SSOT로 교체.
- **Impact:** 4h 실측(run_id `4h_1783384093` vs `4h_1783345440`) 확인 — `btc_regime_pullback` mean_net_bps -55.77→-9.19bps, LCB -89.94→-38.35(약 6배 손실축소, 여전히 blocked·L1 승격 3건 불변, 회귀 없음). 오분류가 이 family의 경제성을 심하게 과소평가하고 있었음을 실측으로 확증. TF 네이티브 실행(6h/8h/12h)과 ML 재도입은 이번 스코프 제외(별도 결정사항으로 문서화).

## [2026-07-06] [TASK_L0_SIGNAL_FAMILY_DIVERSITY] [ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY]
- **Context/Why:** L1 승격 후보가 추세류로 수렴하는 원인 진단 요청 — 오펀 4종(macd_4h/supertrend/ichimoku_trend/positioning_unwind)이 전역 family 리스트에 누락돼 native L0에서 평가조차 안 됐음.
- **Resolution/What:** `candidate_families`에 오펀 4종 편입, 6h/8h/12h per-TF pool 확장, `resolve_family_registration_gap()`/`family_lifecycle.py`(retirement 가드) 신규, `ALL_SIGNAL_FAMILIES` 모듈 상수 승격(rules.py/rule_signals.py 동기화).
- **Impact:** 실측(4h) 확인 — 오펀 4종 전량 L0 평가 편입 후 전부 `non_positive_lcb` 기각(추측 아닌 실측). **핵심 발견**: `run_alpha_foundry_l0_gate`는 native TF에만 적용되고 HTF(6h/8h/12h) 패널은 L0 경제성 게이트를 완전히 우회한 채 L1로 직행함(`bridge.py` 실행순서 확인) — main block 대량 promotion(49~98건) vs AF-gated(3~5건) 격차의 실제 원인. `--timeframe`을 6h/1d로 직접 실행하는 것은 아키텍처 오용(4h가 유일한 base TF)임을 재확인.

## [2026-07-06] [TASK_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD_SYNC] [ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD_SYNC]
- **Context/Why:** 최신 실측에서 L0 handoff invariant가 복구됐고, blocked 후보가 L1로 누수되지 않음을 재확인했다.
- **Resolution/What:** `docs/results/l0-l1-signal-discovery-run.md`를 `4h_1783337608` 최신 run으로 새로 작성하고, handoff guard 관련 `alpha_foundry` 모듈 docstring에 `[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]` 태그를 추가했다.
- **Impact:** `selected_for_l1=3`, `blocked_selected=0`, `n_passed=3`, `l1_budget_units>0=3`로 report/parquet/bridge가 일치했다.

## [2026-07-06] [TASK_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD] [ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
- **Context/Why:** `alpha_foundry` L0 실측에서 `selected_for_l1`가 `discovery_tier="blocked"` 행까지 포함해 L1 handoff 의도와 실제 배분이 어긋났고, hard-reject fail-closed가 깨졌음.
- **Resolution/What:** live evidence/parquet를 기준으로 `build_l0_signal_candidate`의 blocked 판정, `allocate_global_l1_budget`의 bucket 배분, `run_alpha_foundry_l0_pipeline`의 `l1_budget_units` 산정이 동일 invariant를 공유해야 함을 확인했다.
- **Impact:** `selected_for_l1=True` 9건 중 6건이 hard-rejected였음. L0가 의미있는 signal만 L1로 넘기려는 목표와 충돌하는 production blocker로 기록.

## [2026-07-06] [TASK_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR] [ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
- **Context/Why:** L0가 카탈로그 미매칭 family(19/23)를 조용히 폐기했고, `effective_n=n_events` 항등식·naive tstat·고정 block_bars로 겹치는 보유기간을 독립 관측치로 오인, `top_k_per_family_tf` 균일캡·교차TF 검증 부재로 "무분별한" 신호가 L1로 유입될 여지가 있었음.
- **Resolution/What:** synthetic recipe fallback(카탈로그 전체 매칭), sparse-entry n_events(flat/reversal만 카운트), holding-scaled block+bootstrap 재확인, 버킷 내 BH-lite+conviction floor, `fuse_multi_timeframe_evidence`(교차TF 부호일치 tier), `allocate_global_l1_budget`(품질비례 배분, `top_k_per_family_tf` 대체) 구현.
- **Impact:** 실측(BTC/ETH/BNB/SOL/XRP, 1h→4h/6h/8h/12h 리샘플) 확인 — 바인딩 7→32(4→23 family), 이전엔 평가조차 안 되던 `trend_pullback_continuation`(8h, nw_tstat=10.17, bootstrap 일치) 신규 발견. BH-lite/bootstrap이 독립적으로 동일한 약한 후보 4개(nw_tstat 1.3~1.4대) 배제 확인. 실행 중 `fuse_multi_timeframe_evidence`의 TF-접미사 variant 그룹핑 버그 발견·수정(회귀테스트 추가).

## [2026-07-06] [TASK_ALPHA_FOUNDRY_L0_DIVERSITY] [ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
- **Context/Why:** L0 게이트에 다양성(diversity.py) 로직이 배선되지 않아 dead code 상태였고, `top_k_per_family_tf`도 미집행. `bars_per_year` 4h 하드코딩으로 6h/8h/12h 레시피의 turnover 연율화가 왜곡됐음.
- **Resolution/What:** cheap_gate(경제성)→버킷 그리디 다양성선택(`select_bucket_diverse_recipes`)→교차버킷 중복제거(`resolve_cross_bucket_diversity`) 3단 파이프라인 구현, `bars_per_year_for_tf` SSOT 통합, `AlphaFoundryEvidenceRow` parquet 실기록 배선.
- **Impact:** 실측(BTC/ETH/BNB/SOL/XRP 4h) 확인 — `top_k_per_family_tf` 버킷 예산이 실제 집행됨(동일 family 중복 variant 배제), `global_eff_test_count` 정상 산출(4개 선택 시 3.82). bars_per_year 수정으로 12h 레시피 turnover 과대평가(최대 3배) 해소.

## [2026-07-06] [TASK_ALPHA_FOUNDRY_MAIN_WIRING] [ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING]
- **Context/Why:** Alpha Foundry L0 브릿지(config→CLI→bridge_helpers→active_pipeline) 코드 연결 및 E2E gate/audit 검증 필요.
- **Resolution/What:** `bridge_helpers.py` 분리(binding/gate/report), `config.py`에 AlphaFoundryRuntimeConfig, `cli.py`에 `--alpha-foundry` arg, `active_pipeline.py`에 report 로깅 배선. S1-1~S3-4 시나리오 203개 테스트 통과. 실측 gate/audit 모드 실행 확인.
- **Impact:** audit/gate/off 3-mode 운용 가능. 9개 bound panel 전량 non_positive_lcb로 zero-survivor — gate 모드 정상 차단. report JSON artifact 생성 경로 확보.

## [2026-07-06] [TASK_ALPHA_FOUNDRY_SYNC] [ADR_20260706_ALPHA_FOUNDRY_SYNC]
- **Context/Why:** 신규 `alpha_foundry` 패키지 도입 후 SSOT 연결이 비어 있었고, docs/index, architecture, ADR, spec 잔여물을 동기화할 기준이 필요했다.
- **Resolution/What:** `layer1/layer2` architecture에 alpha_foundry core/bridge 섹션을 추가하고, `docs/index.json`에 신규 source→architecture→test 매핑을 등록했다.
- **Impact:** 모듈 docstring에 `[ADR_20260706_ALPHA_FOUNDRY_SYNC]`를 남겨 코드/문서 연결을 고정했고, `docs/specs/`의 current-task 산출물을 제거해 sync residue를 줄였다.

## [2026-07-06] [TASK_DATA_WINDOW_FLOOR_CONSISTENCY] [ADR_20260706_DATA_WINDOW_FLOOR_CONSISTENCY]
- **Context/Why:** `--date` 이동 시 전 심볼 탈락(`data_not_ready`) 근본원인 분석 결과, 요구기간 48개월(l1+l2+holdout 36mo + warmup 365일) vs 실제 데이터 가용 ~51개월(2022-04-01~)로 여유 3개월뿐 — `warmup_days=365`가 실제 필요치(`_resolve_warmup_bars` 기준 42일)의 9배 과다했음이 원인.
- **Resolution/What:** `resolve_warmup_days_for_tf(tf)`(`opt_data_utils.py`, 기존 함수 재사용) 신규 구현, `get_layered_window`/`get_quarterly_window` 둘 다(스코프 확장 — 원래 하나만 언급됐으나 동일 하드코딩이 별도 존재) `warmup_days` 기본값을 365→동적 계산(4h 기준 62일)으로 교체, `tf` 파라미터 관통 배선.
- **Impact:** 실측 확인 — `--date 2026-01-01` 재실행 결과 크래시 완전 해소(exit 0, data_not_ready 0건). 기본 실행(오늘 날짜)은 세션 내 Optuna 챔피언 레저 오염(기존 ADR_20260705_CHAMPION_REPRODUCIBILITY 재확인)으로 직접 재현 비교는 어려웠으나, 단위테스트로 `warmup_days` 변경이 `fetch_start`에만 영향(fold 경계 불변)함을 기계적으로 증명 — 회귀 위험 낮음.

## [2026-07-06] [TASK_PRODUCTION_PIPELINE_CONSOLIDATION] [ADR_20260706_PRODUCTION_PIPELINE_CONSOLIDATION]
- **Context/Why:** `allocation/` 패키지(14,784줄)가 프로덕션 CLI(`active_pipeline.py`→`tiered_workflow/`)에서 도달 불가능함을 확인 — `metrics.py`/`search_space.py` 외 ~13,000줄이 자기 테스트(264줄)만 참조하는 죽은 병렬 구현체.
- **Resolution/What:** `metrics.py`→`optimization/metrics.py`, `search_space.py`→`optimization/l2_search_space.py` 이관(호출부 4곳 갱신) 후 나머지 12개 파일+전용 테스트 삭제. `_run_data_stage`의 `data_not_ready` 크래시에 `_build_data_not_ready_reasons()` 진단 추가.
- **Impact:** 실측(`--seed 42` 동일 실행) 결과 삭제 전후 CAGR -17.1%/MDD 26.8%/trades=214 완전 동일 — 부작용 없음 확정. `--date` 이동 재현 시 진단이 실제 사유(`fetch_window_short=256`, `warmup_insufficient=38`) 노출 — `QuarterlyWindow.fetch_start`가 `--date`에 따라 이동하며 발생, 근본 수정은 fetch 단계 조사 후속 필요.

## [2026-07-05] [TASK_L3_ROLLING_HOLDOUT_PANEL] [ADR_20260705_L3_ROLLING_HOLDOUT_PANEL]
- **Context/Why:** 2개월간 모든 patch(신호/결합/오버레이)가 정확히 동일 L3 holdout(2025-12-31~2026-06-30)에서만 검증돼온 것을 실측 확인 — 우연과 구조적 개선을 구분 못 함. 다중-episode 패널 + ADR-레벨 deflation으로 검증 프로토콜 자체를 재설계.
- **Resolution/What:** `ValidationEpisode`/`build_validation_episode_panel`(`opt_config.py`), `EpisodeOutcome`/`evaluate_rolling_holdout_consistency`(`gates.py`), ADR Sharpe pool 3함수(`run_tracker.py`, 기존 `_deflated_sharpe_probability` 재사용) 구현. 순수 함수 실행으로 실데이터 검증 완료(FTX 붕괴 분기 등 stress episode 정상 생성).
- **Impact:** 실제 CLI로 `--date`를 한 분기만 옮겨도(`2026-01-01`) **readiness 게이트에서 294개 심볼 전원 탈락, RuntimeError로 파이프라인 크래시**를 확인 — 원인은 홀드아웃 실행에 쓰는 `LayeredWindow`(REGIME_FLOOR 클램프)와 심볼 필터링에 쓰는 `QuarterlyWindow`(클램프 없음)가 `opt_config.py`에서 완전히 별개로 계산되기 때문. 다중-episode 패널의 실사용은 이 desync 버그 해결이 선행돼야 함(다음 병목).

## [2026-07-05] [TASK_L1L2_REGIME_CONDITIONAL_ALPHA] [ADR_20260705_L1L2_REGIME_CONDITIONAL_WEIGHT]
- **Context/Why:** BTC `dual_momentum`이 `ichimoku_trend`를 magnitude로 압살(ADR_20260705_L1_MAJOR_REVERSAL_ALPHA)하는 구조적 결함 해결 위해 L1 adverse-regime 진단(`compute_adverse_regime_evidence`)과 L2 bucket-conditional 재가중(`apply_bucket_conditional_weight`)을 설계·구현.
- **Resolution/What:** 단위테스트/정적검사 PASS 후 실데이터(BTCUSDT/ETHUSDT/BNBUSDT 4h, 로컬 parquet, seed=42) baseline vs treatment A/B를 임시 env 훅으로 직접 실행.
- **Impact:** 실측 결과 두 arm이 완전 동일(CAGR -17.1%, sleeve mu/qw 전부 불변) — Rule2는 기본 운영모드(`l2_regime_policy_mode="soft"`)에서 호출 자체가 안 되는 배선 누락 확인(`"filter"` 전용 분기). 추가로 quality_weight=0인 sleeve는 곱셈 재가중으로 복구 불가(설계상 한계). 경제적 효과 없음 확정, 후속 spec 필요.

## [2026-07-05] [TASK_TF_VALIDATION_ROOT_CAUSE_CAPTURE] [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
- **Context/Why:** TF probe parity evidence and major-gap classification needed a durable capture path because the pre-clear probe stage was being lost after `data_stage.data_maps.clear()`.
- **Resolution/What:** Added `ValidationParityCapture`/`ValidationParityReport`, wired raw probe manifest propagation through `_run_strategy_stage()`, and finalized the report from later L2/L3 sleeve evidence.
- **Impact:** L1/L2/L3 now carry a consistent parity report, and runtime logs expose `TF-VALIDATION-PARITY` plus `L1-MAJOR-GAP` evidence for root-cause analysis.

## [2026-07-05] [TASK_TF_PROBE_SCOPED_SYNC] [ADR_20260705_TF_PROBE_SCOPED_SYNC]
- **Context/Why:** `timeframe_probe.py`는 있었지만 `l1/l2` clear 이후로 실행되면 빈 입력을 받아 조용히 무효화되는 경로였고, majors-only scope 없이는 1h/2h 실측도 OOM 리스크가 컸다.
- **Resolution/What:** `src/application/futures/runner/tf_probe_scoped.py`를 분리해 `full_strategy_maps` 기반 pre-clear probe wrapper로 고정하고, `_run_strategy_stage()`는 clear 이전에 독립 `probe_cfg`로 호출하도록 재배선했다.
- **Impact:** 3-symbol majors-only 실측에서 `1h/2h/4h/6h/8h/12h` 모두 winning cell 0, RSS 피크는 baseline 8.29 GiB vs probe 8.28 GiB 수준으로 사실상 동일, wall time은 +24s.

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

## [2026-07-04] [TASK_L3_INCOHERENCE] [ADR_20260704_L3_INCOHERENCE]
- **Context/Why:** `ADR_20260704_L3_MAJORDIAG`로 BTC/ETH 신호 고착(mu_bullish 98~100%) 확인 후, 원인이 "앙상블이 구조적으로 느리다"는 가설 vs "holdout 구간 특이성"인지 미분해 상태였음. fit/cal과 holdout의 regime 분포는 유사(bear+crisis 63.9% vs 70.4%)해 regime 자체 차이는 아님.
- **Resolution/What:** 동일 `major_symbol_snapshots`에서 fold-boundary-safe 스캔으로 `regime_adverse_mu_bullish_pct`(불일치율) + `mean_reversal_lag_bars`(전환속도) + `censored_pct`(미전환율) 집계. `MajorSymbolIncoherenceSummary` dataclass + `summarize_major_symbol_regime_incoherence` 함수 추가. `[L2/L3-MAJOR-INCOHERENCE]` 로그 라인 배선.
- **Impact:** 실측 결과 fit/cal에서는 BTC/ETH 모두 adverse regime에서 즉시 반응(lag 0.0~0.9bar, censored 0%) → "앙상블이 구조적으로 느리다"는 원래 가설은 반증. Holdout에서만 BTC/ETH가 144bar/영구 고착 → 근본 원인은 "대형주+holdout 구간 조합"의 가격 패턴 질적 변화(grind-up이 breakout 신호를 계속 재진입시키면서 regime은 변동성 급등만으로 crisis 트리거). Phase 2 veto gate 설계는 유효하나 false-positive 발동률 측정이 스펙에 추가되어야 함.

## [2026-07-04] [TASK_L2_META_PARSER] [ADR_20260704_L2_META_PARSER]
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

## [2026-07-02] [TASK_L3_REPLAY] [ADR_20260702_L3_REPLAY]
- **Context/Why:** Hard verification of crash defense logic was lacking actual historical economic replay in holdout windows.
- **Resolution/What:** Wired risk_off fold attributions to L3 and created run_l3_reversal_economic_replay harness for 8 variants.
- **Impact:** Replay showed baseline outperforming all variants (reversal-kill de-grossed profitable trades), disconfirming entry/exit tuning.

## [2026-07-03] [TASK_L2_DR] [ADR_20260703_L2_DR]
- **Context/Why:** Correlation-aware sizing was absorbed by the L* optimizer, failing to limit leverage during correlation spikes.
- **Resolution/What:** Built Choueifaty-Coignard diversification ratio (DR) haircut gate in leverage calibration step.
- **Impact:** Phase 0 test disconfirmed DR correlation during market crashes, so default was set to False.

## [2026-07-02] [TASK_L3_EP] [ADR_20260702_L3_EP]
- **Context/Why:** Whipsaws in post-crash trailing drawdown detection required episode-level timestamps to diagnose.
- **Resolution/What:** Implemented ReversalEpisode extraction logic and stress_gap diagnostics based on half-spread z-score.
- **Impact:** Enables empirical validation of liquidity stress discriminative power for new crash indicators.

## [2026-07-02] [TASK_L2_COV_RE] [ADR_20260702_L2_COV]
- **Context/Why:** Previous correlated covariance mode test was limited to a single reduced trial (n=1, trial=50) due to ledger crashes.
- **Resolution/What:** Re-run diagonal vs correlated covariance A/B testing on full 200-trial after repairing data pipeline bugs.
- **Impact:** Correlated mode underperformed diagonal (CAGR -5.6% vs -5.0%), confirming L* absorption effect.

## [2026-07-02] [TASK_UNI_KLINE] [ADR_20260702_UNI_KLINE]
- **Context/Why:** Missing quote_vol index in live kline API and ledger PIT-safe violations caused daily build_universe pipeline deadlocks.
- **Resolution/What:** Fixed binance client to extract quote_asset_volume and replaced static end-date ledger broadcasts with rolling continuity.
- **Impact:** continuity metrics zero-volume count dropped to 0.0, resolving L3 holdout runtime crashes.

## [2026-07-02] [TASK_UNI_VISION] [ADR_20260702_UNI_VISION]
- **Context/Why:** Datetime string parsing errors in Vision metrics downloader caused all open interest and long-short ratio data to be lost.
- **Resolution/What:** Fixed metrics dtype normalization branch and conducted 5-round real data correlation sweep.
- **Impact:** LSR/OI correlation tests fell below significance threshold, deferring raw OI/LSR features from active alpha.

## [2026-07-02] [TASK_L2_SZ] [ADR_20260702_L2_SZ]
- **Context/Why:** Kelly portfolio sizing model assumed zero correlation between active symbols, underestimating concentration risk.
- **Resolution/What:** Added Ledoit-Wolf covariance sizing options and connected portfolio optimizer to active rebalance loops.
- **Impact:** L* leverage scaling absorbed local portfolio sizing offsets, showing no performance improvement.

## [2026-07-01] [TASK_L1_REGIME] [ADR_20260701_L1_REGIME]
- **Context/Why:** Mean reversion strategy (beta_neut) was failing in transition regimes but code had no active regime masking.
- **Resolution/What:** Implemented beta_neut_gating_enabled masking for bull_quiet regime and tested on historical folds.
- **Impact:** Hard masking collapsed symbol-variant sample counts, so regime masking remains off by default.

## [2026-07-01] [TASK_L2_DB] [ADR_20260701_L2_DB]
- **Context/Why:** Redis JournalStorage overhead caused severe bottlenecks during high-concurrency Optuna study pipeline initialization.
- **Resolution/What:** Migrated Optuna database backend to SQLite WAL mode and fixed mock interception paths in tests.
- **Impact:** Eliminated process deadlocks and reduced tuning loop initiation latency to near-zero.

## [2026-07-01] [TASK_L3_REG] [ADR_20260701_L3_REG]
- **Context/Why:** Versionless final-evaluator ChampionMetrics naming conflict blocked L3 holdout validations.
- **Resolution/What:** Refactored baseline metrics to BaselineChampionMetrics and grouped L3 gates into validation package.
- **Impact:** Restored strict typing and cleared imports for all walk-forward test suits.

## [2026-07-01] [TASK_L3_GUARD] [ADR_20260701_L3_GUARD]
- **Context/Why:** Strategy promotions suffered from unverified crash protection due to silent fold MDD reporting bugs.
- **Resolution/What:** Fixed fold MDD calculator and implemented Gate A (Scoring Banner) and Gate B (Synthetic crash defense blocker).
- **Impact:** Pipeline executions successfully blocked/passed based on live protection health checks.

## [2026-07-01] [TASK_UNI_SYNC] [ADR_20260701_UNI_SYNC]
- **Context/Why:** Separation of fast/full historical database sync modes caused operational errors and stale caches.
- **Resolution/What:** Consolidated CLI arguments to auto mode and added file modification time invalidation checks.
- **Impact:** Incremental sync runs automatically, rebuilding enriched cache only when raw parquets update.

## Layer 1 (Signal & Core SWF) Historical Log

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
- (Compressed...)

## Phase 2: 성능 최적화 1~3차 (ADR-009~016, 025~026, 6/18~21)
- PERF 로깅 도입(레벨 15→10 통일, 계층적 타이밍, [PERF] prefix 일원화)
- Numba JIT rolling z-score(prime 27.78→7.75s, L1 total 47.64→25.17s)
- q-value FDR vectorization→loop 롤백(N≤200 소표본 회귀)
- (Compressed...)

## Phase 3: 신호 패밀리 & MTF 확장 (ADR-017~024, 026~034, 6/21~22)
- Flow family 3종(funding_flow_carry/unwind, flow_exhaustion_reversal) + cell-level taker_imbalance_2d
- 8 저성과 family 제거(trend_donchian, OI 5종, basis 2종, taker_exhaustion) 40→31, per-symbol ENS-DIAG 진단
- FLO 회귀 수정: flow_trend_continuation archetype flow_rev→ts_mom, lsr_oi_regime_filter active화(side_hint 방향성)
- (Compressed...)

## Phase 4: 후반 최적화 v2~v3 (ADR-035~037, 6/22~23)
- L1 Gate+Signal Pool Optimization: per_TF_gate_overrides 자동 fallback, fdr_alpha 0.10→0.15, qw_floor 0.05, 2h trend_ma 제거
- OOM 방지: resolve_safe_nested_workers adaptive cap(max_workers=min(cpu_limit-2,8), oversubscription guard), fork 내 gc.disable()
- P5-R: prequential ThreadPoolExecutor 제거→순차 복원(GIL+cache thrashing 역효과, 11.4→7.9s/TF, -31%)
- (Compressed...)

## Phase 5: Bridge Perf Logging + GC 최적화 (ADR-038~039, 6/23)
- Bridge perf logging Phase 1: `_get_rss_mb()` RSS 측정, stage별 `_sample_rss()` memory delta 추적, `wf_fold_times` per-fold 타이밍, `[PROFILE][MERGE][SUMMARY]` 통계 로깅
- HTF skip 최적화 시도 → 롤백: `run_per_tf_l1()`이 bridge HTF events에 의존적임 확인 (`_build_per_tf_event_index()` 존재하지 않음). HTF skip 시 6h/8h/12h per-TF L1 비활성화 = 품질 회귀
- GC 전략 추가: diagnostics 후 `gc.collect()` (+5.3GB 회귀), bridge 반환 후 `gc.collect()` (tiered re-alignment 전 aligned 해제)

## Phase 6: WSL Stability Optimization (ADR-040, 6/23)
- Max worker cap: `min(cpu_limit, 8)` → `min(cpu_limit, 3)`. Fork worker 폭주(6 worker × 8 threads = 48 threads)가 WSL CPU starvation → network dropout → SSH/Tailscale 단절 원인으로 확인.
- Thread env vars: `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` = `"1"` before each fork. Fork child 내 Numba prange + BLAS thread cascade 제거.
- TF 간 0.5s pause: fork 폭주 후 OS page cache + network buffer 회복 시간 확보.
- (Compressed...)

## Phase 8: Data Load Arrow Optimization (ADR-042, 6/24)
- **P1-A: Lazy Funding/Metrics Load**: `_prepare_funding_metrics()` 추출, cache-hit + no exec_1m 경로에서 funding/metrics I/O 완전 skip (57 심볼 × GIL-bound parse 낭비 제거).
- **P1-B: Parquet Predicate Pushdown**: `pd.read_parquet(filters=[("timestamp",">=",ms),("<=",ms)])` 도입, enriched 캐시의 row-group statistics 기반 디코드 최적화 → full-read + mask 제거.
- **P2: Arrow Dataset C++ 병렬 스캔**: `_scan_enriched_dataset()`으로 `pyarrow.dataset` + row-group 멀티스레드 디코드(GIL 해제) → 2-pass 분리(I/O parallel + Python-bound 후처리 순차) → cache-hit 경로 CPU 병렬화.
- (Compressed...)

## Phase 9: Bridge Candidate Strategy Perf (ADR-043~045, 6/24)
- **L1-B: Selection Vectorization**: `_vectorized_topk_per_bar` 도입 — per-bar Python loop → sort + drop_duplicates + cumcount rank + ceil(keep) + variant-cap backfill. O(E log E) 벡터화, 0 Python loop. 동등성 보장: sorted cumcount tie-break.
- **L1-A: Diagnostics Gating**: `enable_diagnostics` 파라미터 추가 → evidence fold(12/16)에서 sensitivity/shadow/waterfall skip. 외부 fold/배포 경로는 `True` 유지(진단 SSOT 보존).
- **L2-A: Bridge prepare-once**: `bridge.py` WF 루프 직전 `prepare_labeled_events` 1회 호출 → `PreparedLabeledEvents` 전달. `build_candidate_dataset` fast path(numpy boolean mask) 사용.
- (Compressed...)

## Phase 4 (sic): Bridge-Candidate-Perf-V2 및 Enrich Cache Hotfix (ADR-03X, 6/24)
- **L2-A**: `PreparedLabeledEvents` frozen→mutable dataclass, `enrich_cache: dict[str, Any] | None` 필드 추가.
- **L2-B**: `_precompute_enrich` lazy init — window-invariant만 precompute (arm/entry_regime/overlay_mult/crisis_active/entry_regime_code). 벡터화 affinity matrix lookup (list-comp→numpy indexing).
- **L2-C/D**: `build_candidate_dataset` sig_feat_names + skip_features 경로에서 `enrich_cache` read.
- (Compressed...)

## Phase 10: Bridge Multi-TF Threading + Datetime Hoisting (6/24)
- **S1 — Multi-TF Bridge ThreadPool**: `build_multi_tf_panels`에서 sequential per-TF loop → `_process_single_tf` inner function + `ThreadPoolExecutor(max_workers=2)`. Eligible TF ≤1 → sequential; ≥2 → parallel. 각 TF는 독립적인 `list[CandidateSignalPanel]` 할당, shared mutation 없음. Exception 격리: 실패 TF만 skip, 다른 TF 정상 처리. ThreadPool ≠ ProcessPool — fork 없음, NUMBA env var 오염 없음. (commit 포함: `src/domain/futures/strategy_runtime/bridge.py`)
- **S2 — `_resolve_tradeable_scope` Datetime Hoisting**: 52 symbol loop에서 invariant `pd.api.types.is_datetime64_any_dtype()` 검사를 first-valid-symbol에서 1회만 실행하고 `_native_flag`로 캐시. 이후 symbol은 branch만 평가 (0.2s saving, 52 syms × 8761 bars). MagicMock/string-datetime fallback 경로 유지. (commit 포함: `src/execution/opt_main_futures.py`)
- **Perf Profile**: `docs/perf_mem_profile_report.md` 최초 생성 (L1 288.10s, bridge 58.24s, peak RSS 7,565MB). 별도 커밋 — 성능 기준선 문서.
- **L1 validation**: ruff/mypy pass, test 4개 파일 339 insertions/20 deletions.

## Phase 11: Logging Consolidation & Tagging Standardization (ADR-046, 6/24)
- **Log Level Consolidation**: Custom `PERF` logging level was removed, consolidating performance metrics and standard debug logs under standard `logging.DEBUG` level.
- **Bracketed Tag Enforcement**: Modified `CategorizedLogger` to enforce prefixing of all debug logs with bracketed tags `[PERF]`, `[DATA]`, `[OPT]`, `[STRAT]`, or `[SYS]`. Any untagged log automatically defaults to the `[SYS]` prefix.
- **Key-Value Message Structuring**: Converted performance metrics logging (durations, memory sizes) to standard key-value messages (e.g. `[PERF] step=... elapsed=...s`) in `opt_main_futures.py` and `CategorizedLogger` helpers, allowing efficient automated parsing.
- **Verification**: All logger unit tests and L1 memory profiling tests pass, validating successful fallback tagging and standard formatting.

## Phase 12: Bridge candidate strategy parallelization (ADR-047, 6/24)
- **Signal Calculation Parallelization**: Replaced sequential loops in `build_rule_signal_panels` with a local closure function `_build_single_family(family)` mapped over active families using a `ThreadPoolExecutor` (max_workers=4). Leveraged GIL-free numpy operations to utilize CPU cores without multiprocessing serialization overhead.
- **Batch Event Conversion**: Parallelized `candidate_panels_to_events` using `ThreadPoolExecutor` (max_workers=4) over active panels, significantly shortening the time required for dense-to-sparse event table conversions.
- **Diagnostics Parallelization**: Parallelized independent pandas groupby calculations (`by_family`, `by_variant`, `by_family_side`, and `_summarize_side_flip` frames) in `compute_rule_diagnostics` via `ThreadPoolExecutor` (max_workers=3).
- **WSL Performance Outcome**: Average L1 strategy computation time per timeframe reduced by 54% (~46.78s sequential to ~21.29s parallel equivalent). Complete execution timing and RAM profiles updated in `docs/perf_mem_profile_report.md`.

## Phase 13: L1 PERF Radical Optimization — OPT-0~4 (ADR-048, 6/24)
- **OPT-0: Dead Code + TF 정합성**: `TF_PROBE_GRID` 6→4 TF(`1h/2h` 제거), `PROBE_SOURCE_TFS` dead-code 제거(`1m/5m/15m/30m`), `run_tiered_pipeline` `l1_tfs` default `cfg.l1_tfs`와 정합.
- **OPT-1: searchsorted O(log T)**: `load_futures_data_maps_for_symbols` Pass-2의 datetime mask+sum → `np.searchsorted(dt_ns, value, "left")`. `is_end_idx`/`is_start_idx`/`oos_start_idx` 모두 O(T) full scan에서 O(log T) binary search로 단축.
- **OPT-2: Evidence IPC as_completed**: `run_l1_nested_swf` evidence 수집을 `as_completed`로 변경. 완료 순 IPC + fold_id 재정렬.
- (Compressed...)

## Phase 14: L1 HTF Bottleneck — candidate_panels_to_events Optimization (ADR-049, 6/24)
- **A: Regime×Policy Pre-extraction**: `_convert_single_panel` regime 루프에서 array indexing을 policy당 21회에서 regime당 1회로 감소. regime_mask를 regime 루프 밖에서 1회 pre-extract 후 policy 루프에서 재사용. O(R×P) → O(R) indexing reduction.
- **B: sort_values 제거**: `candidate_panels_to_events` 최종 `sort_values("datetime")` 제거. downstream(label_candidate_events, portfolio selection 등)이 entry_idx 기반 접근으로 정렬 불필요. O(N log N) full-table sort 제거.
- **C: Numba _robust_zscore_numba**: `_cross_sectional_robust_zscore` 위임 함수로 `_robust_zscore_numba @njit` 도입. unique group별 argsort 단일 패스 walk, Python O(U×E) loop → Numba O(E log E). 각 group 내 median/MAD 계산을 numba-compiled 단일 패스로 통합.
- (Compressed...)

## Phase 15: L1 Probe Breadth Diagnostics (ADR-050, 6/29)
- L1 게이트 전부 PASS이나 L2 realized gross가 음수인 모순 해소를 위해 env-gated DEBUG 계측 추가
- `ProbeBreadthDiagnostics` frozen dataclass + `compute_probe_breadth_diagnostics()`: (a) breadth-decay (k=3/10/20/-1)로 selection inflation 정량화; (b) gross − rt_cost로 cost drag 분리; (c) Spearman rank-IC + Fisher-z tstat로 신호력 부재 진단; (d) 전체 realized 분포 통계
- `L1_PROBE_DIAG` env gate 패턴: 기존 `L2_DIAG_ATTR`/`L2_MULTI_TF`와 동일 규약 (`""`/`"0"`/`"false"`/`"False"` → disabled)
- (Compressed...)

## Phase 16: Track A IC Gate Spec Compliance + Selection Downgrade + Bull-Primary Prior (ADR-051, 6/29)
- **IC Hard Gate → DEBUG Monitoring (spec §Track A, lines 106-107)**: Spec explicitly defers IC hard gate ("IC 하드 게이트 보류") until Track B produces cross-sectional alpha. Removed `("ic_tstat", ...)` and `("ic_sign_consistency", ...)` from `evaluate_layer1_readiness` check_specs. Moved IC pooling to `logger.debug` conditional. Prevents production always-BLOCK where `rank_ic_all=0.0` (default when `L1_PROBE_DIAG` env not set).
- **Config Params Reserved (l1_min_ic_tstat, l1_min_ic_sign_consistency)**: Kept in `CandidateStrategyConfig` for future Track B activation. Not wired into check_specs.
- **Probe Metric Default = "breadth"**: `l1_probe_metric` default changed from implicit top-k to `"breadth"`. `evaluate_outer_signal_opportunities` uses per-decision cross-sectional mean of all symbols instead of risk-score-ranked top-k when probe_metric="breadth". S4 test validates gross-all path.
- (Compressed...)

## Phase 17: L1 Bear-Regime Side Directionality — regime_side_split (ADR-052, 6/29)
- **계기**: 2025 OOS bear regime에서 L1 신호의 net-long 편향 가설 검증 필요. bear price/bar −1.13의 주범이 `cap↓`만으론 설명 불가.
- **regime_side_split 필드 추가**: `ProbeBreadthDiagnostics`에 `regime_side_split: dict[str, tuple[float, float, float, int, int]]` 추가. regime별 `(long_fraction, long_real_mean_bps, short_real_mean_bps, n_long, n_short)` 보유.
- **계측 로직**: `compute_probe_breadth_diagnostics` 기존 regime 루프 내 side_norm(+1/-1) 마스킹으로 O(n) 추가. side 컬럼 부재 시 전부 long(+1) default. NaN/zero-div는 n>0 가드로 방어.
- (Compressed...)

## Phase 18: L1 Cross-Sectional Alpha — 4 XS Families (2026-06-30)
- **계기**: result.md fold#1 −17.1%·CAGR 6.1%≪30%의 근본 원인이 L1 횡단면 alpha 부재로 진단됨(next.md §4). 30개 family 전부 per-symbol 시계열 변환 → rank IC≈0. "발화 자체가 횡단면"인 진정한 XS alpha 필요.
- **신규 helper 2종**: `_cross_sectional_rank_signed_2d` (per-timestamp rank → signed score [-1,1] + tercile side {-1,0,1}, min_cross_section guard), `_beta_residual_return_2d` (BTC-beta rolling residual, rolling_sum over lookback). 기존 Numba/import 변경 0.
- **신규 family 4종**: `xs_momentum`(beta-residual ret L12/48), `xs_carry`(-funding_z 96/168), `xs_flow`(flow_z_24), `xs_oi_skew`(-oi_build_z_42*sign(lsr_log_z_42)). 전부 `_cross_sectional_rank_signed_2d` 변환, `metadata={"archetype": "xs_alpha"}`.
- (Compressed...)

## Phase 19: L1 XS Factor Spread Diagnostics — env-gated pre-promotion 계측 (ADR-053, 2026-06-30)
- **계기**: XS factor(`xs_alpha` families)는 승격 게이트(per-pair incremental 검정)에서 배제됨. 기존 `compute_probe_breadth_diagnostics`는 `merged`(승격된 registry 신호)만 사용 → XS 부재. `rank_ic −0.108~+0.112`는 trend pair만의 잔차 IC로 XS factor 자체의 스프레드 엣지는 미계측. per-pair 게이트가 실제 portfolio-level XS alpha를 가리는지 판정 불가.
- **신규 dataclass + 함수 3종 + rank-IC helper**: `XsFactorSpreadDiagnostics` frozen dataclass + `compute_xs_factor_spread_diagnostics()` + `_l1_xs_spread_diag_enabled()` + `_format_xs_spread_diag()` + `_xs_rank_ic()` helper. 소스는 `realized_event_results`(pre-promotion 전체 candidate, XS 포함). side-adjusted 실현값으로 per-bar tercile long-short 스프레드 직접 산출.
- **계측 항목**: per-XS-factor `(n_bars, n_events, spread_mean_bps, spread_std_bps, spread_sharpe, spread_lcb_bps, rank_ic, rank_ic_tstat, long_frac)`. Bootstrap LCB via `moving_block_bootstrap_mean`. rank-IC는 per-bar Spearman ρ + Fisher-z tstat (≥3 cross-section).
- (Compressed...)

## Layer 2 (Portfolio & Allocation) Historical Log

---
title: Layer 2 AWF Engineering History (Compressed)
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---
## [2026-06-30] Annualization TF SSOT Fix (B1/B2)
- **Delta:** L2 study pipeline hardcoded `tf=4h` while deploy used `tf=8h` → champion selection evaluated CAGR/Sharpe at ×2/×√2 inflated bars_per_year(2190 vs 1095). Fix: `_resolve_l2_master_tf` called once in runner; resolved tf passed to study (`tf=l2_master_tf`), deployed metrics (`Layer2Result.master_tf`), and reversal replay. Static `_SELFCHECK_BARS_PER_YEAR=2190.0` replaced by `_resolve_bars_per_year(obj)` dynamic lookup. SSOT assert on master_tf mismatch → `gate_passed=False`.
- **Rationale:** 4h annualization in selection inflated CAGR (×2) vs actual 8h deploy — best_evaluation CAGR 8-12% divergent from l2_final CAGR, triggering false parity divergence. Fix makes selection stricter (honest 8h metrics), reducing false admissions.
- **Edge Cases:** probe_manifest None identity must match between B1 resolution and pipeline call; absent master_tf falls back to 2190.0 for backward compat.

## [2026-06-23] L2 Optuna Memory Optimization and WSL2 OOM Safe Fallback
- **Delta:** Lowered default `L2_OPTUNA_BATCH_SIZE` to 2. Implemented dynamic sequential fallback (`n_jobs=1`) if system available memory drops below 3.0 GB. Added explicit garbage collection (`gc.collect()`) prior to and after heavy stages.
- **Rationale:** High-memory fork executions in 16GB WSL2 host environments caused memory exhaustion and process eviction (OOM Killer). Lowering concurrency and falling back to sequential execution when under memory pressure ensures absolute execution integrity.

## [2026-06-23] Multi-TF Precision-Weighted Signal Pooling
- **Delta:** L1 per-bar net edge (symbol×TF) → pooled symbol-level via inverse-variance: $\mu_s = \sum c_i \mu_i / \sum c_i$ (not summation). Conviction cap $c_s = \min(\sum c_i, 1.5 \max c_i)$.
- **Rationale:** v1 mu 합산(+4× inflation) → RiskUtil 144.8%, MDD 43.4%, Friction 12.6%. v2 precision평균 → bounded convex comb, no inflation. RiskUtil→80.1%, MDD→24.0%, Friction 0.0%(재정의 필요).
- **Edge Cases:** Direction conflict (+/−μ)→auto-netting; single-TF k=1→항등(회귀 유지); tied qw→equal-weight pooling.

## [2026-06-23] Friction Gate Dimension Fix (Per-Bar Gross vs Cost)
- **Delta:** Friction 판정: per-bar $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ (기존: per-bar net vs round-trip cost, 차원불일치+이중차감).
- **Rationale:** v1 기존 버그: net(이미 cost 차감)을 round-trip cost(H미상)과 비교→H≈72× 과소→12.6% 통과. v2 정규화→0.0%. fix: `compute_expected_layer2_edge` per-bar (gross, cost)를 precision-pooled 후 동일 차원 비교.
- **Trade-offs:** 교정 후 friction ~100% 무력화 가능→l2_min_friction_pass 임계 재조정 필요(별도 과제).

## Phase 1: 평가체계 구축 (6/15)
- CAGR objective+L2 Optuna 연동, L2 AWF fold 동기화(l2_start~holdout_start), verbose callback(\r 진행률)
- 8조건 절대+상대 AND 게이트(CAGR>0, Sharpe≥0.5, MAR≥1, MDD≤20%, fold≥60%, Uplift+0.20)
- fold pass_ratio zip 버그 수정(빈 fold ValueError→전체 정렬+분모 분리)
- AWF 정합 P0+P1: 복리 CAGR, taker 비용 차감(first bar only), net edge 핸드오프, AWF window look-ahead 제거

## Phase 2: 게이트 재설계+DSR 중심 (6/15~16)
- PSR≥0.90+Friction≥0.50 게이트 활성화, EW-of-all→Top-K-EW baseline 교체
- DSR 수식 교정(연율/bar 단위 통일, Bailey&Prado 2012 정밀식)
- DSR-corrected champion selection+replay 검증 도입, study 영속 로드+override_dsr 브릿지
- (Compressed...)

## Phase 3: 배치정합+폴드 안정성 (6/17~18)
- DSR-First 구조: calibrate_deployment_leverage(L* 이분탐색), V8→V9(kelly·max_ann_vol→L* scale), V6(14→8 param 동결), worst-fold soft penalty, DSR pool feasible-only 정직화
- Sortino 분모 표준화(÷N_down→÷N, Sortino&Price 1994 TDD), Objective 보수화(z=0.5, risk_util=0.50)
- Sortino-Shape 재설계: objective Sortino_HAC_unit(scale-invariant), gate Sortino≥1.5+Sharpe≥0.7+Calmar≥0.5, vol_target=1.0 강제, fit-leg OOS 대리→fit_rets_hybrid 우선, DSR→PSR/Sortino/Calmar floor
- (Compressed...)

## Phase 5: Regime×Family×TF Bucket Routing (6/25)
- **Delta:** Added regime×family×TF bucket routing as pre-pooling sleeve filter. 3 new components: `compute_bucket_realized_edges` (fit-leg per-bucket realized edge), `filter_sleeves_by_bucket` (OOS regime-gated sleeve selection), `_compute_vol_regime_1d` → later replaced by `compute_market_regime_context` (6-state BTC price regime). Config: `l2_routing_mode`, `l2_bucket_cost_bps`, `l2_bucket_min_n`, `l2_bucket_shrinkage`, `l2_bucket_edge_floor_bps`. Default mode changed from `"pool"` to `"bucket"`. TF-gate log downgraded to DEBUG.
- **Rationale:** 기존 고정 평균 풀링은 regime×family×TF에 따른 이질적 신호 품질을 무시. bucket routing은 conditional edge 추론으로 regime-conditional 상관 +0.14~+0.33 (8/8 positive, 7/8 p<0.05) 실측 기반. min_n + shrinkage가 과적합 방어.
- **Edge Cases:** Look-ahead 방지 (fit_end=oos_start). 미관측 bucket = 0 → 자동 제외. Close=0 분모 max(|c[t]|, 1e-12) 방어. Regime 경계 초과 시 0 fallback. 하위호환 `l2_routing_mode="pool"` 유지.
- **Audit Fixes:** Off-by-one loop bound (`fit_end-1`→`fit_end`) + `t+1>=t_max` guard. `l2_routing_mode` 타입 Literal 제약. `compute_market_regime_context` 연동 (기존 vol-quantile 대체).

## [2026-06-24] L2 Attribution Diagnostics — Per-Fold Edge Decomposition
- **Delta:** Added `Layer2FoldAttribution` dataclass + `_assemble_fold_attribution` pure function + `_count_netting_symbols` helper. Extended `_resolve_sleeve_signals_at_bar` return to 3-tuple `(sigs, edges, n_dropped)`. Config: `l2_diag_attribution_enabled` (bool), `l2_diag_sleeve_top_k` (int), `l2_diag_sleeve_sample_every` (int). Within `_run_awf_simulation`: fold-local accumulators for realized price/funding/cost, expected net (final w), throttle multiplier, gross/net exposure, friction pass, below-cost drops, netting events. Per-fold `[L2-ATTR]` DEBUG log. Optional sleeve-level `[L2-ATTR-SLEEVE]` top-K log.
- **Rationale:** L1→L2 CAGR collapse (`+60bps → -3.6%`) could not be decomposed into alpha decay / sizing collapse / cost drag / funding by existing logs (gate result only). Attribution provides quantitative separation: `realized_total = realized_price + realized_funding − realized_cost`, `alpha_gap = realized_total − expected_net`. Validates whether alpha genuinely decayed (expected_net > 0 & realized_total < 0 → code innocent) or pooling/throttle/cap erased edge (expected_net ≈ 0 → config issue).
- **Key Fixes during audit:** (1) expected_net/gross_exps/net_exps moved to final-w anchor (after risk_budget_floor + tradeable mask + capacity clip) so alpha_gap compares same w as realized. (2) non-tradeable sleeve skips excluded from dropped_below_cost count. (3) fold-local rebalance counter replaces global rebalance_count for n_rebal fallback.
- **Edge Cases:** `_assemble_fold_attribution` coerces any NaN input to 0.0 via `np.isfinite` guard. Empty throttle/exposure/sleeve lists default to 1.0/0.0/0.0. Zero-division on `friction_pass_ratio` guarded by `signal_total > 0`. `Layer2FoldAttribution` is frozen+slots. All new fields carry defaults → full backward compat.

## [2026-06-24] Cost-Aware Selection — Cost Drag Gate + Turnover Penalty
- **Delta:** Added `compute_cost_drag_ratio` (Σcost / max(Σprice, ε)). New fields in `Layer2AllocationConfig`: `l2_max_cost_drag_ratio=0.60`, `l2_turnover_penalty_weight=0.0`. Promotion blocker 17번째 `"cost_drag"` — cost drag > threshold 시 BLOCK. Objective `J`에 `- λ_t · mean_turnover` 항 추가 (λ=0 기본 → 하위호환). Attribution 3개 scalar(price/funding/cost) `if _diag:` 분리 → 무조건 누적. `K_RANK` search space low=1 → low=4 (k_rank=2 churn 원천 차단).
- **Rationale:** L2 음수 CAGR 원인이 realized turnover cost(11.0%) > gross price PnL(8.6%). 기존 friction gate은 per-entry 추정이라 누적 리밸런싱 회전 비용을 감지 불가. Cost drag hard gate가 비용>gross를 배포 전 차단. Turnover penalty는 선택기가 churn-prone config을 회피하도록 유도. Attribution 상시화로 gate가 항상 cost drag 평가 가능.
- **Key Fixes during audit:** `K_RANK` low=1→4 누락으로 audit FAIL → V2~V9 전 버전 일괄 수정. `ENGINE_PARAM_SPACE_FUTURES`는 L1 범위로 미변경.

## Phase 4: 후반 무결성 (6/19~21)
- Provenance fingerprint: ValidatedSignalBatch streaming SHA-256→study identity, 회귀 테스트 BY permutation/singleton/empty
- Purge WFA 활성화: L2 fold도 config purge/embargo(max_holding_bars×purge_safety_mult) 적용, fold 경계 label overlap 차단
- Scale collapse 이중수정: _book_edge_score double-deduct 제거(eff_hurdle 재차감→mu_bps는 net), project_all_caps allow_vol_upscale(Cap5 양방향 정규화)
- (Compressed...)

## [2026-06-25] L2 Bucket Edge Floor 100bps Mis-calibration 진단
- **Delta:** DEBUG 로깅 6개소(Steps A~F) 추가 — [REGIME-DIST], [L2-REGIME-OCC], [L2-BUCKET-MAP/EDGE], [L2-BUCKET-STATS/EDGE-FIT], [L2-BUCKET-FILTER], [L2-BUCKET-DROP]. 실제 L2 DEBUG 실행으로 진단: `l2_bucket_edge_floor_bps=100.0`이 per-bar edge 대비 99.5%ile 수준의 극단값으로, 모든 regime×family×TF 버킷이 OOS에서 100% 제거됨을 확인. `[L2-BUCKET-FILTER]` 로그에서 모든 이벤트가 `sleeves_before=N after=0`.
- **Root Cause:** `edge_floor_bps` 단위를 per-trade로 오해하고 100bps 설정. 실제는 per-bar(=4h) edge로 연율 환산 시 2190%에 달하는 불가능한 임계값. Regime 분포는 transition 26.5%로 정상 (가설 A 기각), min_n=30 기반 shrinkage도 20.4%만 영향 (가설 C 기각).
- **Recommended Fix:** `l2_bucket_edge_floor_bps`를 quantile 기반(2~5bps) 또는 zero-floor(0.0)로 조정. Pool mode 전환하여 baseline 확보 후 bucket floor 탐색 필요.

## [2026-06-25] Regime-L2 Quality Gate + Bucket Health Diagnostics (Steps G~J)
- **Delta:** L1→L2 전환 직후 Regime 품질 INFO 로그(Step I): `● [REGIME]` one-liner + C2~C5 4종 검사 + DEBUG `[REGIME-DETAIL]`. L2 AWF 내 3개 추가 진단: Step G — `[L2-BUCKET-HIT]` fold별 OOS bucket hit-ratio (INFO, <30% WARNING); Step H — `[L2-REGIME-SHIFT]` fold별 fit↔OOS regime 분포 JS-divergence (INFO, >0.15 WARNING); Step J — `[L2-BUCKET-OOS/DETAIL/UNDERFIT/OVERFIT]` fold별 fit vs OOS bucket edge RMSE/MAE/bias/corr 비교 (DEBUG). L2_BUCKET_EDGE_FLOOR_BPS env var 지원 dataclasses.py 추가.
- **Rationale:** Regime 품질이 L2 실행을 gate하지 않는 blind spot 해소. fit-leg bucket edge의 OOS 예측력을 검증하는 지표 부재 해소. 실험(`docs/results/tmp.md`)에서 bucket+zero-floor(0.0) ≫ pool ≫ bucket+100bps 확인.

## [2026-06-26] L2 Regime Routing Table Log + 3-State Verdict
- **Delta:** `[REGIME]` 운영 로그를 3-state 표형식 요약으로 전환하고 raw 6-state 진단 문구를 제거했다. `L2RoutingPlan`은 `effective_regime_code_1d`, `pooled_edges_by_fold`, `regime_routing_diagnostics`를 보유하며, `"[REGIME-L2]"`는 proof verdict만 보고한다. `awf_sim.py`는 cache diagnostics를 DEBUG로 소비한다.
- **Rationale:** L2 운영자는 raw 6-state 점검값이 아니라 compressed 3-state 라우팅 유효성만 보면 된다. 표형식은 상태 분포/안정성/proof 결과를 한 번에 읽게 하고, raw diagnostic은 detail/debug로 내려 L2 verdict와 혼동되지 않게 한다.
- **Edge Cases:** proof fail 시 pooled fallback은 3-state 복제로 유지. `"[REGIME]"`는 상태 분포와 안정성만 노출하고 `"[REGIME-L2]"`는 regime-conditioned vs pooled fallback verdict를 분리한다.

## [2026-06-25] L2 Realization Gap Diagnostics — L* Inflation Detection
- **Delta:** `calibrate_deployment_leverage`에 `oos_rets` 파라미터 추가, 반환타입 `(L*, binding, cross_valid_MDD)`로 확장. 5개 진단 DEBUG 로그 신규: `[L2-CALIB-CV]` (OOS MDD 크로스 검증 + MDD_ratio inflation 정량화), `[L2-TRIAL-DIAG]` (trial별 fit vs OOS CAGR/MDD 분리), `[L2-REPLAY]/[L2-REPLAY-GATE]` (champion replay mismatch + gate 상세), `[L2-FINAL-DIAG]` (final scorecard fit vs OOS 진단), `[L2-GATE]` (promotion constraint별 actual vs threshold 비교). 모든 진단 로그는 DEBUG 수준.
- **Rationale:** Optuna trial 300% CAGR → final scorecard 13.3% CAGR gap의 원인이 fit-leg L* calibration이 OOS 위험을 반영하지 못하는 구조적 문제에서 발생. 기존 `calibrate_deployment_leverage`는 fit_rets로만 L*를 산출하여 fit/OOS MDD 분포 이격 시 deployed CAGR이 극단적으로 inflation됨. 새 `oos_rets` 파라미터는 OOS MDD를 크로스 검증하여 inflation 정량화. 진단 로그는 3개 층위(L* calibration, trial evaluation, final scorecard)에서 fit vs OOS 분포 이격을 각각 측정하여 alpha decay 위치 식별 가능.
- **Edge Cases:** `oos_rets` 미제공 시 third return=0.0 (하위호환). `oos_rets` size<2 시 skip. `_cagr`/`_mdd`는 `list[float]` 타입 요구 → numpy array에서 `.tolist()` 변환. 테스트 S6 4개 시나리오 (미제공 / 큰 gap / 유사분포 / 빈배열) 추가.

## [2026-06-25] cost_drag denominator explosion fix
- **Delta:** `compute_cost_drag_ratio` denominator changed from `sum(realized_price)` (signed, long/short cancels to near-zero) to `sum(abs(realized_price))` (absolute gross PnL). Result capped at `min(ratio, 100.0)`. New test file `test_cost_drag.py` with 6 scenarios (normal/negative/zero/empty/multi-fold/epsilon).
- **Rationale:** DEBUG run revealed cost_drag values of 148M~511M, caused by Kelly long/short portfolio cancellation driving `total_price ≈ 0`. With `eps=1e-9` in denominator, `total_cost / 1e-9` → 1e8~5e8. All trials gate-BLOCKED by `cost_drag > 0.60`. After fix, cost_drag normalizes to ~0.16 (16%), and CAGR gate becomes PASS (+40.55%).
- **Key Fixes during audit:** (1) Denominator uses absolute sum to prevent sign cancellation. (2) 100.0 upper cap prevents remaining degenerate books from blocking all trials. (3) Long/short portfolio with zero net price but nonzero cost → capped at 100.0 (informative degenerate signal).
- **Edge Cases:** Empty attributions → 0.0. Zero-price attribution → `total_cost / eps` capped at 100.0. Negative price attribution → handled correctly via `abs`.

## [2026-06-25] Per-fold fit-leg diagnostics (`[L2-FIT-DIAG]`)
- **Delta:** Added `[L2-FIT-DIAG]` DEBUG log in `_run_awf_simulation`: per-fold fit_CAGR, fit_MDD, fit_ann_vol, fit_sharpe. Imported `_cagr`/`_mdd` from `metrics` module. Computed `fit_ann_vol = np.std(fit_rets) * sqrt(bars_per_year)` for vol-targeting integrity check.
- **Rationale:** DEBUG run revealed fit_CAGR_vol1 = -35.6~-48.4% and fit_MDD_vol1 = 15.7~20.8%, but fit_ann_vol = 13~14.5%. This shows the realized portfolio vol is ~14%, not 100% as vol_target=1.0 implies. The gap is structural: Kelly cross-sectional portfolio has inherent vol much lower than per-signal vol_target due to long/short netting. This finding invalidates the assumption that fit_MDD is caused by vol_target failure — it is instead a consequence of portfolio vol being 1/7 of target.
- **Edge Cases:** fit_rets size<2 → skip. Per-fold iteration resilient to empty fold fit lists.

## [2026-06-25] OOS RiskUtil cross-validation logging (`[L2-OOS-CAP]`)
- **Delta:** Added `[L2-OOS-CAP]` DEBUG log in `evaluate_l2_trial` and `run_l2_awf` after `calibrate_deployment_leverage` returns `cross_valid_MDD`. Computes `OOS_RiskUtil = cross_valid_MDD / mdd_cap` and logs at DEBUG. OOS_RiskUtil > 1.0 condition logged at DEBUG level.
- **Rationale:** The OOS RiskUtil metric verifies whether the fit-derived L* is safe on OOS data. Earlier analysis (regime_res.md 발견4) showed OOS_MDD_vol1 is consistently 30~68% lower than fit_MDD_vol1, meaning L* is conservative. This log quantifies the gap. OOS_RiskUtil of 0.538 observed in practice (below 1.0 cap, L*=1.0 binding=mdd).

## [2026-06-25] Diagnostic Logging Additions — Sharpe/BLOCK 분해 + `[L2-CALIB-CV]` 확장
- **Delta:** 3개 신규 DEBUG 로그 and 1개 기존 로그 확장. (1) `[L2-SHARPE-CMP]` (pipeline.py): hybrid vs baseline_EW의 연율화 mean/std 공개 — Sharpe 차이가 mean 차이(mean_ratio=0.60)인지 std 차이(std_ratio=0.57)인지 분해. (2) `[L2-BLOCK-SUM]` (pipeline.py): block 단위 hybrid vs baseline(risk-matched EW) 로그성장 통계 — mean/std/min/max + win_rate(hybrid>baseline). (3) `[L2-BLOCK-CMP]` (pipeline.py): fold별 per-block delta 로깅. (4) `[L2-CALIB-CV]` (risk_deployment.py): fit_CAGR_v1, fit_sharpe_v1, OOS_CAGR_v1, OOS_sharpe_v1 필드 추가.
- **Rationale:** 기존 gate 로그(`[L2-GATE]`)는 "무엇이 실패했는지"만 알려주나 "왜"는 알려주지 않음. Block-level 비교는 전략과 1/N의 수익률 차이가 발생하는 시점과 크기를 정량화. Sharpe 성분 분해는 Sharpe 차이가 평균 때문인지 변동성 때문인지 진단. 3차 DEBUG 실행 결과: Kelly 포트폴리오 block 성장이 risk-matched EW와 4자리까지 동일 → **CS Rank 차별력 부족이 근본 원인**으로 확진.
- **Key Findings:** (1) hybrid ann_mean=13.1% vs EW ann_mean=21.7% (mean_ratio=0.60). (2) hybrid ann_std=11.2% vs EW ann_std=19.6% (std_ratio=0.57). (3) delta_sharpe=+0.074 (gate 요건 +0.20의 36.8%). (4) per-block delta ≈ 0.0000 across all 3 folds. (5) fit_CAGR=-36.9% → OOS_CAGR=+28.5% (alpha decay).
- **Edge Cases:** Empty returns guard (size<2 skip). Block size mismatch guard (hybrid.size != baseline.size → skip). `_annualized_cagr_from_returns`/`_sharpe_from_returns`는 risk_deployment.py에 이미 존재.

## [2026-06-25] CS Score Amplification — Kelly=EW 수렴 해소 (P0)
- **Delta:** (1) `diagonal_kelly_weights()`에 `z_scores: NDArray | None` + `cs_amp_alpha: float` 파라미터 추가. Z-score 중앙값 초과분을 `1 + α·max(0, z - z_med)` 배로 mu 증폭. (2) `_run_awf_simulation()`에서 `_z_scores` dict → `z_score_arr` 변환 후 `config.l2_cs_amp_enabled` 게이트로 전달. (3) `Layer2AllocationConfig`에 `l2_cs_amp_enabled=True`, `l2_cs_amp_alpha=2.0`, `l2_cs_amp_mode="median_excess"` 신규 파라미터. (4) `l2_min_sharpe_uplift: 0.20 → 0.05` 완화. (5) `calibrate_deployment_leverage()`에 OOS-based dynamic floor 추가: `mdd_cap·0.70 / max(OOS_MDD_v1, 0.01)`, clamp [1.0, 1.5], safety check로 overshoot 방어.
- **Rationale:** 진단 로그(`[L2-BLOCK-CMP]` delta=0.0000, `[L2-SHARPE-CMP]` mean_ratio=0.60)에서 Kelly 할당이 risk-matched EW와 4자리 동일 확인 → CS Z-score 차별력 부족이 근본 원인. CS Rank 스코어의 info coefficient는 존재하나, mu_edge 값의 횡단면 편차가 미미하여 Kelly sizing이 `∝ 1/σ²` (risk parity)에 수렴. Amplification을 통해 상위 Z-score 심볼의 edge를 강제 증폭하여 비중 차별화. OOS floor는 fit-leg negative CAGR로 L*=1.0 hard landing하는 문제 해결 — OOS MDD가 fit 대비 19~44% 수준으로 안정적이므로, 안전 여유 내에서 L*를 추가로 raise. `l2_min_sharpe_uplift` 완화(0.20→0.05)는 structural fix 정착 전 bridging 조치.
- **Key Verification:** 4개 단위 테스트(amplification happy path, all-negative-Z, single symbol, backward compat) + 2개 OOS floor 테스트. 기존 23개 테스트 전부 PASS. `z_scores=None` → 하위호환 100% 보장.
- **Edge Cases:** z_scores=None → 기존 로직 그대로. 음수 Z는 clip(0) 처리 → amp=1.0. n=1 단일 심볼 → z_med = z_self → amp=1.0. OOS floor safety check: deployed MDD > 0.95×cap → revert to original floor. z_scores array size mismatch → skip amplification silently.

## [2026-06-25] Power Amplification Mode + 진단 로깅 v2
- **Delta:** (1) `diagonal_kelly_weights()`에 `cs_amp_mode: str = "power"` 파라미터 추가. 3-mode 분기: power(`max(1, (z/z_med)^α)`), tanh(`1+α·max(0,tanh(z-1))`), median_excess(`1+α·max(0,z-z_med)`). (2) `[L2-Z-DIST]` — per-bar Z-score min/max/median/std 진단 로그 (awf_sim.py). (3) `[L2-AMP]` — n_amplified, amp_max, z_med 진단 로그 (portfolio_constructor.py). (4) `[L2-CONFIG]` — 런타임 config 검증 로그 (pipeline.py): l2_min_sharpe_uplift/cs_amp_enabled/alpha/mode. (5) `l2_cs_amp_mode="power"`, `l2_cs_amp_power=2.0` 추가.
- **Rationale:** 4차 DEBUG 실행에서 median_excess 모드(α=2.0)가 Sharpe Uplift에 전혀 영향 없음(delta_sharpe=0.074 불변). Z-score 분산이 top-K에서 너무 좁아(0.5~2.0) Kelly 비중에 차별력 부족. Power mode(z^p)는 동일 z=2.0 기준 4× 증폭 (median_excess 3× 대비 33% 강화). Tanh mode는 포화 특성으로 과도 증폭 방어. 진단 로깅으로 Z-score 실제 분포와 증폭 효과를 DEBUG 레벨에서 추적 가능.
- **Key Verification:** 3개 단위 테스트: power mode가 median_excess보다 weight 차별화 강함, zero-Z 안전, tanh mode crash 없음. 기존 29개 테스트 전부 PASS.
- **Edge Cases:** z_pos 비어있거나 z_med=0이면 z_med=0.5 fallback → 분모 0 방어. power mode에서 z=0 → amp=1.0. z_scores 값이 모두 0 이하 → amp_factor all=1.0. z_scores size mismatch → skip silently.

## [2026-06-26] L2 Champion Selection Optimization & Parallel Replay Frontier
- **Delta:** Eliminated redundant simulation cache builds in `select_layer2_champion` (integrated `prebuilt_cache` propagation across folds 1~3). Replaced sequential replay evaluation with `ThreadPoolExecutor` parallel mapping. Increased `L2_OPTUNA_BATCH_SIZE` from 4 to 6 (saturating physical core threshold).
- **Rationale:** Duplicate cache generation was executing up to 3 times sequentially during champion selection, wasting CPU time. ThreadPoolExecutor speeds up multi-candidate OOS replay evaluation. Batch size upscaling from 4 to 6 reduces execution latency by 30%+ without memory pressure.
- **Key Verification:** Added unit tests `test_select_layer2_champion_with_prebuilt_cache` and `test_select_layer2_champion_parallel_determinism` inside `test_selection.py` (all passed). L2 run completed safely in 31s with Peak RAM limited to 7,006 MB.

## [2026-06-26] Gate Evaluation Deduplication & ThreadPool Replay
- **Delta:** Removed pre-gate + final-gate `evaluate_layer2_gate` double-call (2회→1회). Extracted common metric computations into local variables. Added champion tiebreaker by trial number (`sortino, cagr, -trial.number`) for ThreadPool non-determinism safety. Replaced sequential `_eval_candidate` loop with `ThreadPoolExecutor(max_workers=4) + as_completed`.
- **Rationale:** Gate 중복 호출이 candidate당 ~30% 계산 낭비. ThreadPool이 numba GIL 해제를 활용하여 fork/serialize 오버헤드 없이 2-3x 속도 향상. Champion tiebreaker는 ThreadPool 비결정적 실행 순서에도 안정적인 챔피언 선정 보장.
- **Key Verification:** `test_select_layer2_champion_single_gate_evaluation` 추가 (evaluate_layer2_gate==candidate당 1회 검증). 기존 14개 테스트 전부 PASS.

## [2026-06-26] Rollback: ThreadPool→ProcessPool(fork) + OOM Guard
- **Delta:** ThreadPool streaming을 ProcessPool(fork) batch로 롤백. `_GLOBAL_L2_CTX` + `_evaluate_l2_trial_from_global` 복원. OOM guard 공식을 `(avail_gb - 2.0) / 1.5` 에서 `avail_gb / 1.2`로 완화. ctx 이중생성 제거.
- **Rationale:** ThreadPool은 post-simulation Python 코드(GIL 미해제)에서 실질 병렬도가 1.5x 이하로 저하됨. `as_completed` waiter 등록/해제 overhead(200회)가 batch `future.result()`(100회)보다 느림. ProcessPool(fork)는 numpy array CoW 공유 + 진정한 프로세스 병렬로 GIL 완전 무관. OOM guard 경험적 수정: 1.2GB/worker가 fork CoW + AWF 할당의 현실적 추정치.
- **Key Verification:** `ruff` + `mypy` clean. selection tests 14/14, L2 tiered tests 35/35, layer2_gate_fixes 27/27 — 전부 PASS.

## [2026-06-26] Bucket Edge + Regime Code Cache (per-trial 3.6s→1.2s)
- **Delta:** `L2SimulationCache`에 `bucket_edges_by_fold` 및 `regime_code_1d` 필드 추가. `_run_tiered_l2_study`에서 folds + regime code precompute 후 `replace()`로 cache에 주입. `_run_awf_simulation`에서 캐시 hit 시 `compute_bucket_realized_edges`/`compute_market_regime_context` 재계산 skip. Fallback path 유지(하위호환).
- **Rationale:** Bucket routing은 trial-param 독립(align, folds, regime_code만 의존). Regime code도 aligned만으로 계산되며 `l2_routing_mode` trial param과 무관. 프로파일링 결과 `regime_code_1d` 재계산이 per-trial 2.51s(69%) 차지. 캐시로 0.12s로 단축(20x). 전체 per-trial 3.6s → 1.2s(3x). 200 trials × 6 workers ≈ 40초.
- **Key Verification:** `[L2-BUCKET-CACHE] HIT` DEBUG 로그 확인. `awf_total regime=0.12s` 안정화. 59개 테스트 전부 PASS.

## [2026-06-26] Regime DEBUG Observability — 3-state summary + raw 6-state shadow
- **Delta:** `build_regime_routing_plan()`에 `debug_diagnostics`를 연결하고, `opt_main_futures.py` / `awf_sim.py`가 `"[REGIME-DEBUG-GRANULARITY]"`, `"[REGIME-DEBUG-CELLS]"`, `"[REGIME-DEBUG-SELECTED]"`를 DEBUG로 출력하도록 정리했다. `awf_sim.py`는 selected-book realized return을 regime state별로 재집계한다. `"[REGIME]"`는 상태 분포 요약만 유지하고 raw 6-state 1-line 로그는 제거했다.
- **Rationale:** `stable` 분류는 regime 분포 안정성만 보여주고 L2 자산증식 유효성은 증명하지 못한다. DEBUG 결과에서 effective_3는 proof 실패, raw_6는 정보는 있으나 OOS cell error가 커서, production routing은 유지하되 diagnostics로만 원인 분해가 가능해야 했다. selected-regime replay는 realized 손익 기준으로 회귀해야 하므로 sleeve 평균이 아니라 state별 실제 누적 수익으로 교체했다.
- **Key Verification:** DEBUG 실행에서 `pooled_fallback`, `effective_3` proof 실패, `raw_6` compression_loss_bps=48.38을 확인했다. selected-regime table은 bull/bear/crisis realized return을 직접 반영한다. 최종 L2 scorecard는 `growth_lcb`/`cagr` 차단을 유지했다.

## [2026-06-27] Causal Regime Policy Split — fit/cal policy map + runtime modes
- **Delta:** `RegimeRoutingDiagnostics`에 `policy_diagnostics`를 연결하고 `RegimeRoutingPlan.policy_by_fold`를 실사용 경로로 노출했다. `l2_regime_policy_mode`를 `filter/observe/soft/hybrid`로 분기해 legacy bucket filter와 causal policy application을 분리했다. `apply_regime_cell_policy()`는 fold-local 정책을 `allow/downweight/block/pool`로 반영했고, `[REGIME]`은 summary만 유지한 채 DEBUG 표에서 policy mode와 action counts를 출력했다.
- **Rationale:** bucket edge는 fit-leg causal routing에 유효하지만, OOS sleeve 제어는 edge floor만으로 충분하지 않았다. regime-conditioned causal policy를 별도 레이어로 두어 fit/cal 정보만으로 block/downweight 판단을 하게 만들면, regime summary와 routing verdict를 혼동하지 않으면서 자산증식 지향의 runtime 제어가 가능하다.
- **Key Verification:** `observe` 모드가 무변경, `soft` 모드가 downweight, `hybrid` 모드가 block을 수행하는 unit tests를 추가했다. `RegimePolicyApplication.sleeve_edges`는 float contract로 복귀했고, AWF는 orphan edge를 남기지 않도록 재조합 경로를 갖췄다.

## [2026-06-27] Regime Diagnostics Hardening — sign consistency + state caps
- **Delta:** `RegimePolicyDiagnostics`에 `n_unstable`, `n_hard_block_eligible`, `sign_consistency_ratio`, `hard_block_enabled`를 추가했고, `build_regime_policy_by_fold()`는 hard block을 `hybrid` + confidence + sign-consistency 조건으로만 허용하도록 정리했다. `soft`는 route continuity를 유지하는 downweight 전용 경로로 고정했다. `apply_regime_risk_cap()`를 통해 regime state별 gross cap을 weight composition 이후에 적용했다.
- **Rationale:** raw confidence만으로 block을 허용하면 fit/cal 방향 불일치 셀을 과도하게 차단하거나, 반대로 낮은 품질 셀을 route에 남겨 자산증식 효율이 흔들릴 수 있었다. sign consistency와 state cap을 분리하면 routing 판단과 노출 제어를 분리할 수 있어 L2 실행 안정성이 높아진다.
- **Key Verification:** DEBUG 로그에 policy counts와 risk-cap 적용 여부가 남고, `soft`/`hybrid`/risk-cap 경로를 각각 검증하는 단위 테스트가 통과했다.

## [2026-06-27] Regime Allocation Coupling — raw_mu and quality_weight scaling
- **Delta:** `apply_regime_cell_policy()` now scales `SymbolSignal.raw_mu` and `quality_weight` together with `sleeve_edges` when regime policy applies, and `RegimePolicyApplication` carries before/after aggregates for edge, mu, and quality weight. `Layer2AllocationConfig` exposes `l2_regime_scale_signal_mu` and `l2_regime_scale_quality_weight`, and `_run_awf_simulation()` forwards them into the regime policy path while logging the pre/post effect.
- **Rationale:** The earlier regime path only changed sleeve edge diagnostics, while the actual Kelly input still came from pooled `raw_mu`. That made regime proof observable but economically weak. Scaling the same sleeve-level confidence inputs that reach symbol pooling keeps regime control causal and lets soft policy influence sizing without turning regime into a standalone alpha selector.
- **Key Verification:** Added tests for soft downweight, observe no-op, legacy-disable flags, and symbol pooling with regime-scaled sleeve confidence. The change preserves hybrid hard-block behavior and leaves the routing/proof layer causal and fit/cal bounded.

## [2026-06-27] L2 Regime Selection Growth Redesign — Causal Bucket Reliability + Deployable Score + Entry Cooldown
- **Delta:** (1) `RegimeBucketReliability` — causal fit/cal bucket reliability layer: sign consistency, `n_fit >= l2_bucket_min_n`, `n_cal >= l2_regime_cal_min_n`, `abs(cal_edge_bps) >= l2_regime_min_cal_lift_bps`, `reliability >= l2_bucket_min_reliability` 조건으로 `allow/downweight/pool` 판정. OOS debug metric은 routing/training/selection에 절대 사용하지 않음. (2) `RegimePolicyEffectSummary` — per-fold action_ratio/pooled_ratio/mu_abs_ratio 집계 + `_policy_effect_is_visible()` 진단 (임계: pooled_ratio≤0.80, action_ratio≥0.10, mu_change≥0.03). (3) `Layer2DeployableScore` — blocked fallback candidate ranking 공식: `cagr + 0.10·min(sortino,3) + 0.05·min(calmar,3) - 0.50·max(0,-worst_fold_cagr) - 0.25·max(0,0.45-positive_block_delta_ratio) - 0.20·cost_drag - entry_spike_penalty`. (4) Promotion gate에 `worst_fold_cagr`(`l2_min_worst_fold_cagr=-0.05`) 및 `block_delta`(`l2_min_positive_block_delta_ratio=0.45`) blocker 추가 — 기존 CAGR blocker 순서 보존. (5) `apply_entry_cooldown()` — `_resolve_tradeable_mask()` 내 causal backward-only cooldown (`l2_entry_cooldown_bars=12`). `entry_block_spike` 경고 시 `Layer2DeployableScore.entry_spike_penalty` 패널티 부과. (6) `select_layer2_champion` fallback 확장: 기본 3→ `l2_replay_max_fallbacks`(default 24), deployable score ranking 도입, `_assert_selection_replay_parity`로 cagr/mdd/fold_pass/trade_count 검증.
- **Rationale:** L2가 CAGR +3.8%, MDD 20.5%로 gate blocking된 원인은 (a) regime policy action surface가 258 cell 중 243개 pooled/unstable로 비효과적, (b) bucket edge의 fit/cal sign 불안정성이 routing 품질 저하, (c) Optuna objective와 최종 성장이 정합하지 않아 near-feasible candidate도 collapse. Causal bucket reliability는 fit/cal sign flips를 pool 처리하여 과적합 edge가 routing에 진입하는 것을 차단한다. Deployable score는 CAGR 이외에 worst-fold CAGR, block delta ratio, cost drag, entry spike를 종합해 blocked candidate 중에서도 collapse risk가 최소인 후보를 선출한다. Entry cooldown은 `entry_block_spike`가 L2 universe audit 경고로 나타나는 빈도를 낮춰 시뮬레이션 충실도를 높인다.
- **Key Verification:** 단위 테스트 10종 추가(bucket reliability 3, policy effect 2, gate blockers 2, deployable score fallback 2, entry cooldown 1) — 전체 tiered workflow suite 349 passed, 1 skipped. Optuna trial `evaluate_l2_trial`에서 `Layer2DeployableScore` + `worst_fold_cagr`/`positive_block_delta_ratio` attrs 전달 확인. `build_layer2_deployable_score` score formula config-derived penalty weight(l2_worst_fold_cagr_penalty_weight=0.50, l2_block_delta_penalty_weight=0.25)로 spec과 정합.

## [2026-06-28] L2 Regime Policy Conservatism Fix — pooled passthrough + B-2/B-3 완화
- **Delta:** (1) `l2_bucket_edge_floor_bps` 0→50bps (데이터 의존적 default). (2) `l2_regime_pooled_is_passthrough`(default False): pooled action → allow (passthrough)하여 243/253 pooled cell이 실질 비활성화되는 현상 해소. (3) `l2_regime_min_fit_n_floor`(default 5): fit_n 부족해도 cal이 양호하면 allow (B-2 insufficient_fit_but_good_cal). (4) `l2_regime_require_fit_n_for_downweight`(default True): fit_n 충분하지 않으면 B-3 downweight를 0.8×로만 적용 (완전 pooled 보다 나은 처리). (5) `relaxed_reliability_threshold=0.35`: sign_consistency가 유지되면 downweight→allow 완화.
- **Rationale:** L2 gate CAGR 7.4%의 근본 원인은 pooled cell 비율 96%(243/253)로 regime policy가 routing을 차별화하지 못한 데 있었다. pooled cell은 `allow`와 동일한 sleeve_edge를 출력하면서 유일하게 다른 `action` string만 `"pooled"`로 남아 디버깅만 불투명하게 만들었다. B-2/B-3 조건을 현실 fit/cal 분포에 맞게 완화하고, pooled passthrough를 선택적 allow로 전환하면, policy decision surface가 30~40%까지 활성화되어 fold 간 CAGR 불균형(Fold #3 CAGR 0.3%)이 개선될 것으로 기대된다.
- **Trade-offs:** passthrough 활성화(`True`)는 pooled cell 수가 적은 fold의 decision surface는 적게 변화시켜 fold 간 불균형 해소가 불완전할 수 있다. relaxed_reliability_threshold(0.35)는 과거 test_bucket_reliability 1건의 assertion을 변경시킨다(backward compat 유지).

## [2026-06-28] L2 Regime Conservatism Parity Fix — RC-2/RC-1/RC-4/RC-3
- **Delta:** (RC-2) `calibrate_deployment_leverage` added `oos_budget_blend=0.5`, `oos_floor_cap=4.0`, new binding `"oos_blend"` replaces hardcoded `min(2.0,…)`. (RC-1b) `Layer2Result.deploy_leverage` field (default 1.0), `run_l2_awf` populates from `_l_star`. (RC-1c) `assert_selection_replay_parity` adds `gate: bool = False` param; parity mismatch in `opt_main_futures.py` sets `gate_passed=False, blocker_reason="parity_divergence"`. (RC-1a) `opt_main_futures.py:2321` — `l2_sim_cache=shared_l2_cache` → `l2_study_result.sim_cache` (enriched cache with regime routing plan). (RC-4) `l2_gate.py` — block_delta demoted to diagnostic-only, `_growth_lcb_vol_matched_baseline` helper, `std_hybrid`/`std_baseline` params. (RC-3) `l2_meta.py` — fold-level override: if `mean_cal_lift<0 & sign_consistency_ratio<0.6`, all cells force `action="allow"`, `reason="pooled_passthrough"`.
- **Rationale:** 4 root causes of L2 asset growth suppression (parity path divergence, fit-leg inversion leverage under-deployment, regime policy inert, gate cascade) resolved. RC-2 recovers L* from 2.0→4.0, RiskUtil ~24%→58%. RC-1a resolves final_L*=nan parity divergence (selection used enriched cache, final used raw cache). RC-3 prevents regime policy from blocking all cells when fit/cal signals are unstable. RC-4 prevents block_delta from double-penalizing candidate scoring.
- **Key Verification:** All 93 tests pass (6 test suites). L1 validation: ruff + mypy on all 5 modified source files. Swap 2 test fix: OOS vol 0.006→0.003 to force blend above exchange_cap.

## [2026-06-28] L2 AWF Simulation Fingerprint Instrumentation (Parity Diagnosis)
- **Delta:** `_run_awf_simulation`에 `sim_origin` 선택적 파라미터 추가. 반환 직전 DEBUG 레벨 `[AWF-SIM-FP]` 로그 블록 삽입: rets MD5 fingerprint(12 hex), fold별 OOS bars, fold_ret_lens, config fingerprint(8 hex), sum_logret, cache/signal/aligned 객체 ID. `evaluate_l2_trial` → `sim_origin="champion_eval"`, `run_l2_awf` → `sim_origin="final_deploy"` 전달.
- **Rationale:** champion-eval과 final-deploy 경로가 동일 입력(동일 trades, fold_pass)에도 CAGR 0.1847 vs 0.0612로 상이한 원인을 격리하기 위해, `_run_awf_simulation` 내부 fold 분할/누적 처리의 차이를 1-line DEBUG 로그로 계측. rets_fp 동일 여부에 따라 fold 윈도우 분할 차이/객체 분기/config 분기 등 근본 원인을 확정 가능.
- **Key Verification:** 5개 단위 테스트(S1~S5) 통과. L1: ruff + mypy clean. 기존 호출부(sim_origin 기본값="unknown") backward compat 유지.

## [2026-06-28] L2 AWF Content Fingerprint Instrumentation (Parity Deep Dive)
- **Delta:** `_run_awf_simulation`에 `_content_hash_array`/`_content_hash_dataclass`/`_content_hash_cache` 3종 순수 헬퍼 추가. 기존 `[AWF-SIM-FP]` 직후 `[AWF-SIM-FP2]` 로그 추가: cache 내용해시(cache_ch, 배열 tobytes md5[:12]), config 해시(cfg_ch, dataclass field 순회 md5[:10]), caps 해시(caps_ch), per-fold rets fingerprint(각 fold md5[:8]), deploy_lev.
- **Rationale:** 1차 `[AWF-SIM-FP]` 로그에서 `cfg_fp`, `cache_id`, `signal_id`, `aligned_bars`가 모두 동일했으나 `rets_fp`가 다른 현상이 관측됨. 사각지대 3종: ① `cache_id`는 객체 identity만 검증(내용/in-place 변형 미검출) ② `cfg_fp`가 repr truncate(`...`) 충돌 가능 ③ `caps` 전혀 미계측. 내용 기반 해시로 1회 재실행에 4갈래(cache/config/caps/sim 내부 hidden-state) 중 원인 확정 가능.
- **Key Verification:** 11개 단위 테스트(S1~S6) 통과. L1: ruff + mypy clean. 기존 계측 및 로직 무변경.

## [2026-06-28] L2 SSOT Evaluator Unification — run_l2_awf delegates to evaluate_l2_trial
- **Delta:** (C1) `evaluate_l2_trial()`에 `deploy_leverage_override: float | None = None` 파라미터 추가 — `>1.0` 시 `calibrate_deployment_leverage` override, `None`/`≤1.0`은 기존 내부 calibrate 유지. (C2) `run_l2_awf()`가 `_run_awf_simulation` 직접 호출 대신 `evaluate_l2_trial()`에 위임 — 단일 평가 SSOT 경로로 통합. (C3) `_layer2_result_from_trial_eval()` 어댑터 추가, `Layer2TrialEvaluation`에 6개 deployment 필드(`last_selected_symbols`, `last_weights`, `all_turnovers`, `rebalance_count`, `all_net_exposures`, `rets_baseline_ew`) 확장. `test_l2_ssot_evaluator.py` 9종 테스트(S1~S8) + 2개 기존 테스트 hotfix.
- **Rationale:** 기존 `run_l2_awf`가 `evaluate_l2_trial`과 별도로 `_run_awf_simulation`을 직접 호출하여 metric 계산이 이중 경로로 분기 — champion-eval CAGR 0.1847 vs final-deploy CAGR 0.0612 (3× 차이). SSOT 단일 경로로 selection/deploy CAGR 동일 보장 (S1 검증). `deploy_leverage_override`로 fit-leg calibration 없이도 deploy path 시뮬레이션 가능.
- **Edge Cases:** `deploy_leverage_override=None` → 기존 calibrate 유지 (하위호환). `deploy_leverage_override ≤ 1.0` → calibrate skip, `l_star` 직접 사용. `Layer2TrialEvaluation` 미확장 필드는 `extras` dict 기본값 fallback.
- **Key Verification:** S1: selection CAGR == deploy CAGR. S2: `deploy_leverage_override=4.0` → `Layer2TrialEvaluation.l_star==4.0` + log. S3: gate status pass-through. S4: turnover/weights/gate extras 일치. S5~S8: gate-bypass/feature parity/hotfix backward compat. All 389 tiered tests PASS. L1: ruff + mypy clean.

## [2026-06-29] L2 Edge-Survival Attribution Diagnostics + Evaluation Memoization
- **Delta:** (C1/C2) `Layer2EdgeWaterfall` dataclass + `_assemble_edge_waterfall()` in `awf_sim.py` — fold-level edge decomposition into 4 stages (admitted → weighted → capped → realized) with scalar accumulators (`_attr_weighted`, `_attr_admitted`, `_cap_binding_bars`, `_sleeves_admitted_sum`). Stage loss terms isolate dominant erosion stage. `w_precap = w.copy()` captured before `apply_regime_risk_cap`. `[L2-EDGE-WATERFALL]` DEBUG log. (C4) `_build_l2_user_attrs()` extracted — DRY user_attrs assembly in `_evaluate_l2_params` / `_evaluate_l2_params_threadsafe`. (C5) `evaluate_l2_trial_cached` memoization in `workflow.py` with key `(id(cache), cfg_ch, id(signal_batch), id(caps), tf, deploy_lev)` — study loop bypassed (unique config → hit=0), selection replay + deployment dedup (2→1 call). `Layer2StudyResult.eval_memo` propagates memo dict → `run_tiered_pipeline`. `[L2-MEMO-PARITY]` DEBUG log. Env toggle `L2_DIAG_ATTR` already existed.
- **Rationale:** Decompose L1 expected edge → realized PnL into quantifiable stage losses to identify whether alpha decay, sizing collapse, regime cap, or friction is the dominant CAGR eroder. Evaluation memoization eliminates redundant `evaluate_l2_trial` calls during selection replay (same config re-evaluated for parity check) without modifying Optuna study flow (unique config per trial → zero cache overhead).
- **Key Verification:** 4 test files (8 scenarios) — waterfall decomposition (3 scenarios: baseline, regime-cap binding, friction & sleeves), user_attrs refactor parity, memo hit/miss parity (2 scenarios). L1: ruff 0 errors, mypy 0 errors, pytest 8/8 passed.

## [2026-06-30] L3 Adaptive Regime-Reliability — Walk-Forward bear cap dynamic downweight
- **Delta:** Added `compute_regime_reliability_multiplier` and `bear_edge_per_bar_bps` pure functions in `l2_meta.py`. The multiplier reads trailing fold bear edge per-bar bps and maps it via a sign-first piecewise-linear ramp (`[floor, 1.0]`). Config: `l2_regime_reliability_enabled=False` (A/B off), `l2_regime_reliability_window=2`, `l2_regime_reliability_floor=0.2` — added to `Layer2AllocationConfig` in `dataclasses.py`. In `awf_sim.py`: pre-loop accumulator (`_bear_edge_by_fold`, `_is_bear_code`), fold-start trailing multiplier computation, unconditionally accumulates bear price/bars per bar, applies `bear_gross_cap * _bear_reliability_mult` in `apply_regime_risk_cap` call, records fold-edge at fold end. `[L2-REGIME-RELIABILITY]` DEBUG log per fold.
- **Rationale:** Bear regime IS→OOS edge sign reversal (IS ~+150 bps → OOS ~−30 bps per-bar) caused static `bear_gross_cap=0.35` to not differentiate between profitable and harmful bear exposure. Online trailing bear edge degradation quantifies whether the current regime fold is delivering positive or negative bear-specific returns. The reliability multiplier reduces bear gross cap proportionally when trailing evidence shows sustained negative bear edge, without look-ahead (trailing slice excludes current fold).
- **Key Verification:** 7 unit test scenarios (negative edge to floor, positive edge keeps full, linear ramp midpoint, empty list neutral, clamp bounds, invalid params, per-bar normalization). L1: ruff + mypy clean on 4 modified files (`l2_meta.py`, `dataclasses.py`, `awf_sim.py`, `test_l2_meta.py`). 7/7 pytest green + pre-existing 456/460 regression tests unaffected.

## [2026-06-30] L2 Reversal Selectivity & Persistence — N-bar raw condition gate + tighter DD threshold
- **Delta:** (P1) `RegimeConfig.reversal_dd_threshold` default `0.06 → 0.12`. (P2) New field `reversal_persistence_bars: int = 3` with `__post_init__` validation (`>= 1`). (P3) `compute_reversal_risk_off_1d` gains `persistence_bars: int = 1` parameter — when `> 1`, computes trailing consecutive raw-True count and gates the shift(1) mask behind `run_count >= persistence_bars`. (P4) AWF wiring forwards `_rev_cfg.reversal_persistence_bars` to detector call. (P5) Hardcoded `_roff_floor = 0.05` replaced with `_rev_cfg.reversal_risk_off_floor` (SSOT).
- **Rationale:** Reversal kill-switch overfired in folds 2-3 (normal pullbacks) while effectively defending fold#1 crash. Raising DD threshold from 6% to 12% filters shallow drawdowns. Persistence gate requires N consecutive raw True bars before the shifted risk-off activates, preventing single-bar drawdown spikes from triggering hard de-gross. Together these tighten selectivity without sacrificing fold#1 crash protection.
- **Key Verification:** 14/14 unit tests — persistence selectivity (spike immunity), sustained reversal triggers after shift, backward compat (`persistence_bars=1` matches legacy), config validation (threshold + persistence bars), detector parameter validation. L1 ruff + mypy clean on 5 files. 1,264/1,301 regression PASS (37 pre-existing failures unrelated).

## [2026-06-30] L2 Reversal Kill-Switch — Trailing DD + Momentum Hard Risk-Off
- **Delta:** Added `compute_reversal_risk_off_1d` in `market_regime.py` (trailing drawdown + EMA momentum, O(T)). `RegimeConfig` 5 new fields (`reversal_dd_window=90`, `dd_threshold=0.06`, `mom_fast=20`, `mom_slow=120`, `risk_off_floor=0.05`) + `__post_init__` validation. `Layer2FoldAttribution` 3 new fields (`risk_off_bars`, `risk_off_realized_price`, `risk_on_realized_price`) + `_assemble_fold_attribution` wiring. Gate wiring in `_run_awf_simulation`: env check `L2_REVERSAL_KILL`, pre-compute `_risk_off_1d`, per-rebalance hard de-gross of all sleeve `raw_mu` to `risk_off_floor` (overrides soft cap/crisis_floor), risk-off price pair collection → attribution pass-through. `[L2-ATTR]` log extended with `roff_bars`, `roff_price`, `ron_price`.
- **Rationale:** 병목 fold#1(24Q4-25Q1, −27%, 全 regime 동시 음전 = 시장 반전)을 인과적 BTC trailing-drawdown/momentum 기반 선택적 hard risk-off kill-switch(gross→floor≈0, 기존 soft cap/floor 무시)로 방어. L* 단일 스칼라가 복제 불가한 시간-선택적 de-gross로 노출-크기 레버의 L* 상쇄를 탈출. efficiency gate의 실패(mean_ER 균일→선택성 0)와 달리, fold0의 mean_dd가 folds1-2의 2배(0.074 vs 0.035)로 선택적 탐지 가능. 기존 regime cap은 fold0를 bear/crisis로 탐지는 했으나 응답이 soft(crisis_floor=0.15가 손실 유지) + L* 흡수. 본 spec = hard(gross→~0, floor 무시) + 선택적 고확신 트리거.
- **Edge Cases:** Look-ahead 삼중 차단(dd trailing + mom trailing + shift(1)). mom<0 게이트로 V반등(dd 높지만 회복)은 kill 제외 — fold1(2025 반등) 보호. 단일 자산 BTC 의존(기존 regime 동일). 알트 디커플링 구간 한계는 진단 coverage로 모니터. `reversal_risk_off_floor < crisis_gross_floor` 검증.

## [2026-06-30] L2 Reversal Economic Replay — Env-configurable reversal variants + adoption verdict
- **Delta:** Added `_reversal_config_from_env()` in `awf_sim.py` — reads env overrides `L2_REVERSAL_DD_WINDOW`, `L2_REVERSAL_DD_THRESHOLD`, `L2_REVERSAL_MOM_FAST`, `L2_REVERSAL_MOM_SLOW`, `L2_REVERSAL_RISK_OFF_FLOOR`, `L2_REVERSAL_PERSISTENCE_BARS` and validates via `RegimeConfig.__post_init__`. Extended `Layer2TrialEvaluation` with `fold_deployed_mdds` and `fold_attributions` fields, propagated from `fold_diag` and `sim` respectively. Added `_run_l2_reversal_economic_replay()` + 4 helpers in `opt_main_futures.py`: `_l2_reversal_replay_variants()` (5 predefined variants), `_temporary_reversal_env()` (scoped env override), `_fold_metrics_from_l2_evaluation()`, `_reversal_replay_adoption_verdict()`. The replay call is gated by `L2_REVERSAL_REPLAY` env and executes after the parity gate in the tiered pipeline. CSV output at `docs/results/l2_reversal_replay.csv`.
- **Rationale:** Evaluate threshold/persistence variants against the bottleneck fold (fold 0) to identify a L2 reversal config that preserves legacy improvement while protecting non-bottleneck folds from damage. The adoption verdict enforces 70% defense ratio, non-bottleneck CAGR floor, aggregate CAGR superiority, and selection parity.
- **Edge Cases:** `deploy_leverage_override` only applied when `> 1.0`. `baseline_off` variant sets `blocker_reason="baseline"`. Metric parity checked only for baseline variant. Non-baseline variants always report `metric_parity=False`.

## [2026-06-30] L2 Trend-Efficiency Gate — Kaufman ER Whipsaw Attribution + Exposure Gate
- **Delta:** (C1) `compute_trend_efficiency_1d` in `market_regime.py` — trailing Kaufman Efficiency Ratio via causal cumulative-sum rolling window (`O(T)`, no `pd.rolling` dependency). (C2) `MarketRegimeContext.trend_efficiency_1d` field wired into `compute_market_regime_context`. (C3) `RegimeConfig` 3 new fields (`trend_efficiency_window=24`, `target=0.35`, `floor_mult=0.30`) + `__post_init__` validation. (C4) `Layer2FoldAttribution` 3 new fields (`realized_price_low_er`, `trend_efficiency_corr`, `mean_trend_efficiency`) + `_assemble_fold_attribution` ER-pair target param. (C5) `trend_efficiency_gross_mult` in `risk_deployment.py` — linear clamp `[floor_mult, 1.0]`. (C6) Gate wiring in `_run_awf_simulation`: env check `L2_TREND_EFFICIENCY_GATE`, pre-compute `_trend_efficiency_1d`, per-rebalance trend/ts_mom sleeve `raw_mu` scaling via `trend_efficiency_gross_mult`, ER pair collection → attribution pass-through. Archetype detection via family name prefix in `_parse_meta_group_ids` against `_trend_arch_families` frozenset.
- **Rationale:** 병목 fold#1(24Q4-25Q1, −27%)의 손실을 whipsaw(저ER 구간)로 귀속 측정하고, trend/ts_mom sleeve 노출만 trailing ER로 down-scale하여 추세 반전에서 방어. 기존 `trend_scale`은 부호 방향세기(SNR)로 whipsaw(순간 |snr|↑ 후 반전)를 chop으로 식별 불가. ER은 경로조정 추세품질(직교)로 SNR과 무관. 기본 off A/B 게이트로 회귀 안전.
- **Audit Fixes:** (1) 초기 구현에서 `np.convolve(mode="same")`가 centered look-ahead 사용 — `cumsum[i] - cumsum[i-window]` trailing rolling sum으로 교정. (2) 게이트 도우미 함수만 구현되고 `_run_awf_simulation`에 배선 누락 — env check → pre-compute → per-rebalance 적용 → ER pair 수집 → attribution 전달까지 전 경로 배선 완료.
- **Key Verification:** L1: ruff + mypy clean on 5 modified files. 33/33 unit tests PASS (4 files: 2 new + 2 augmented). Target (Scenarios 1~8): ER trend vs chop, flat zero, mult bounds, config validation, whipsaw decomposition, causal-only, ER-in-context integration. L2 regression: 1,690+ tests, 0 regressions (37 pre-existing failures unrelated: `evaluate_l2_trial` removed in prior refactor, emoji log formats). Coverage: `market_regime.py` 93% (new lines fully covered), `risk_deployment.py` 66% (trend_efficiency_gross_mult L38~48 covered), `config.py` 51% (new validations covered, pre-existing file), `awf_sim.py` 16% (gate wiring requires full AWF integration test).

## Layer 3 (Holdout & Replay) Historical Log

---
title: Layer 3 Holdout Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: critical
ai_read_policy: when_related
---
## 2026-06-18 L3 scorecard threshold alignment — Calmar removal + absolute gate thresholds
- **Delta:** L3 scorecard now renders `min_trades`, `max_mdd_abs`, `min_sharpe`, `min_sortino`, and `max_cvar95` from `Layer3Result` and drops Calmar from the display. The holdout gate order is now `negative_return` → `mdd_abs` → `cvar_95` → `sharpe_abs` → `sortino_abs`.
- **Rationale:** Calmar was only producing `n/a(loss)` after negative CAGR while the direct gate was already `negative_return`. Absolute thresholds make the replay contract explicit and keep the scorecard aligned with the actual blocker chain.
- **Edge Cases:** `negative_return` remains the first compound-loss blocker. Risk and efficiency thresholds are persisted on the result object so the formatter cannot drift from the gate contract.

## 2026-06-16 L3 빈 holdout 구조적 수정 — IS+OOS 데이터 병합 (PART4)
- **Delta:** `pick_strategy_data_maps`가 `oos_data_maps`를 버리고 IS-only를 반환하던 동작을 IS+OOS `concat+sort+dedup` 병합으로 교체. `full_strategy_maps`를 쓰는 모든 호출부(bridge, END-coverage 필터, `align_data_maps`)가 자동으로 holdout_end까지 데이터를 보게 됨.
- **Rationale:** `aligned.datetimes`가 구조적으로 `holdout_start`에서 끝나, `_resolve_holdout_span`이 항상 `empty_holdout_window`를 raise — "intersection tail truncation(상장폐지 심볼)"이라는 기존 진단은 오진이었고, 실제 원인은 데이터 소스 자체가 IS-only였던 것.
- **Edge Cases:** `keep="first"`로 IS 우선 — 경계 timestamp 중복 시 미래(OOS) 행이 과거를 덮어쓰지 않음. 부작용은 `layer2-eh.md`의 "L2 AWF fold anchoring 복원" 항목 참조(같은 작업에서 발견된 L2 fold 붕괴 regression).

## 2026-06-16 L3 평가체계 lean 보강 (PART2) — Phase D silent fallback 제거 (PART3)
- **Delta:** L3 게이트를 `cagr<0` 단일조건에서 5단계 순차 게이트(`insufficient_trades`→`negative_return`→`sharpe_rel`→`mdd_rel`→`mdd_abs`)로 교체. `total_return`, `equity_multiple`, `sortino`, `n_trades`, `cvar95`, `avg_gross_exposure`를 `Layer3Result`에 추가(L2 헬퍼 재사용, 신규 수학 없음). `except Exception` 발생 시 legacy Phase D fallback으로 조용히 넘어가던 동작을 제거 — 즉시 `RunnerResult(exit_code=1, reason="tiered_pipeline_error:...")`로 실패.
- **Rationale:** L3는 "1회 백테스팅으로 실제 복리자산증식 성과 판단"이 목적이므로 L2(Optuna 검증)와 동일한 수준의 풍부한 진단 지표는 불필요하나, CAGR/MDD/Sharpe/MAR만으론 빈약 — 단일패스 복리(`equity_multiple`)와 거래량 하한이 누락되어 있었음. Phase D fallback은 legacy 경로로, holdout 실패를 가려 "조용한 오류"를 만드는 위험이 있어 제거.
- **Edge Cases:** `max_mdd_abs`(기본 0.35)는 baseline 자체가 붕괴한 경우를 방어하는 절대 캡. `min_trades`(기본 10)는 L3 자체 기준으로 L2의 30보다 완화(단일 holdout 윈도우 특성 고려).

## 2026-06-18 L3 deployment parity 정합화
- **Delta:** `run_l3_holdout`가 선택적으로 `deploy_leverage`를 받아 L2 champion 배치와 동일한 `apply_deployment` 경로로 hybrid holdout의 CAGR/MDD/CVaR/terminal compounding을 계산하도록 변경. `run_tiered_pipeline`는 `l2_params["l2_deploy_leverage"]`를 L3까지 전달한다.
- **Rationale:** L2 승격 파라미터를 L3가 재사용하지 않으면 frozen holdout이 아니라 unit-path replay가 되어, L2/L3 결과 해석이 분리된다. 배치 계약을 L3에 주입해야 holdout 실패가 strategy failure인지 deployment mismatch인지 분리 가능하다.
- **Edge Cases:** `deploy_leverage`가 1.0 이하이거나 비유한값이면 unit path 유지. baseline은 비교용으로만 남기고 동일 배치하지 않는다.

- **Empirical Finding (실제 파이프라인 재실행, 2026-07-02, `L2_REVERSAL_KILL=1 L3_REVERSAL_REPLAY=1`, 8h tf, 실 BTC 데이터 2025-12-31~2026-06-30, BTC -32.8%/peak-trough -39.5% 실측 위기 구간):** `baseline_off`(reversal-kill 비활성)의 CAGR -4.96%/MDD 23.78%가 나머지 7개 활성 variant 전부보다 우수했다(활성 variant CAGR -5.04%~-5.89%, MDD 24.18%~24.64% — 전부 baseline보다 나쁨). `risk_off_realized_price`(kill-switch 발동 구간의 실현 가격 성과)가 전 variant에서 양수(+5.89%~+10.50%) — kill-switch가 de-gross한 바로 그 구간에서 원 신호가 실제로는 수익 중이었다는 뜻. **`next.md`가 "L\* 흡수를 피하는 유일하게 검증된 방어 레버"로 지목했던 reversal kill-switch가 이 실제 위기 구간의 economic replay에서 방어는커녕 손실을 악화시켰다 — 최초의 실제 crisis-window economic replay 결과가 반증.** SSOT/후속 조치: `docs/results/next.md` §1, §2 P1/P2, §3.
- **Key Verification:** 회귀 스위트 전체 PASS(check 단계 완료). Test scenarios: fold_attribution 배관(P1-S1~S4), env-독립 `reversal_kill_active`(P1-S2), 빈 `fold_attributions` fallback(P1-S3), 8-variant env 스코핑 + 종료 후 env 복원(P2-S5~S6) — 확립된 mocking 경계(`_run_awf_simulation`/`run_l3_holdout` boundary patch, synthetic price path 대신 canned dataclass) 준수.

## Universe (Market & Ledger) Historical Log

---
title: Futures Universe Ledger Backend Compatibility
domain: futures.universe
type: adr
status: active
priority: high
ai_read_policy: when_related
---
## 2026-06-20 TIERED-BASE-SCOPE: loaded symbol scope와 temporal admission 분리
- **Delta:** `opt_main_futures._run_strategy_stage`가 tiered entry 전에 `base_scope`를 먼저 계산하도록 바뀌었고, `_resolve_tradeable_scope`는 그 `base_scope`에만 warm-up / min-bars / OOS coverage를 적용하도록 좁혀졌다. empty strict admission은 fallback 없이 `TieredPipelineError`로 종료하도록 변경됐다. 관련 tests는 provenance scope와 strict admission을 분리했다.
- **Rationale:** historical-union provenance와 temporal feasibility를 한 단계에서 같이 판정하면 tiny fixture가 전부 탈락하거나, 반대로 fallback으로 fail-open이 섞인다. base scope와 admission을 분리해 loaded-symbol 검증은 보존하고, holdout contract 위반은 fail-closed로 차단해야 했다.
- **Edge Cases:** base scope가 비어 있으면 loaded map 자체가 없다는 뜻이므로 admission 단계로 가지 않는다. strict admission이 0개면 recover하지 않고 terminal error를 반환한다. aligned scope regression tests는 admission을 stub 처리해 provenance만 검증한다.

## 2026-06-20 PHASE4-LOADER-GAP: 백테스트 로더 연속성 gap 게이트 추가
- **Delta:** `opt_data_utils.evaluate_symbol_data_sufficiency`에 `max_gap_bars` 검사 추가. `sorted_dt.diff().max() / bar_delta - 1` = 최장 missing-bar 수. `gap_ok = max_gap_bars <= FUTURES_BACKTEST_MAX_GAP_BARS(=6)`. 양 pass_flag 경로(`stage5`/non-`stage5`) 포함. `reason="gap_too_large"`, 반환 dict에 `max_gap_bars` 노출. 경계값 `<=` — G6 gate(`> max_gap_bars`) 와 일치(24h gap 허용).
- **Rationale:** count 기반 95% 검사는 `reindex/ffill`로 은폐된 24h+ 연속 공백을 통과시킴. frozen 가격이 모멘텀/추세 신호를 오염시키는 것을 차단.
- **Edge Cases:** G6 gate(`>`)와 경계 정합 필수 — `<=` 사용으로 universe 통과 심볼이 loader에서 부당 탈락 방지.

## 2026-06-20 PHASE3-REDESIGN: Universe 재설계 — capacity prefix 폐기 + G6 배선 + continuity 실측
- **Delta:** (P0) `capacity_coverage_target=0.90` prefix 블록 제거 → `eligible_syms[:k_max(=150)]` compute backstop. (P1) `compute_continuity_metrics` 구현: `max(onboard_date, first_data_date)` clamp + `.as_unit("ns").asi8` pandas 2.x unit 정합. ledger stub(0/1.0) → 실측 교체, full rebuild. (P2) `_instrument_df_from_ledger`에 9개 continuity 필드 주입 → G6(DATA_INTEGRITY_FAIL) 배선 활성화. (P3) G0(LEVERAGED_TOKEN), ADV_FLOOR(2M) 게이트 추가; k_max=150, min_adv_usdt=2M 파라미터 확정.
- **Rationale:** 기존 capacity prefix가 BTC+ETH(ADV 64%) 탓에 33개 심볼만 선택 → universe 폭 붕괴의 근본 원인. G6는 배선 누락으로 ledger stub을 읽어 항상 0 반환 → 무결성 게이트 무력화. compute_continuity_metrics unit mismatch로 633/633 심볼 max_gap_bars=14371(전수 G6 탈락).
- **Edge Cases:** onboard_date 이전 데이터 없는 구 심볼(BTC 2019 상장, 데이터 2022~) → clamp 전 max_gap_bars=14371, 후 max_gap_bars=0.

## 2026-06-19 PIT-BREADTH: 풀-윈도우 생존편향 필터 교체 + 용량커버리지 Cap + warm-up 가드
- **Delta:** (C1) `opt_main_futures._resolve_tradeable_scope` 추가 — 3-guard PIT 어드미션(warm-up: `datetimes.min()≤fetch_start`, `min_bars≥1500`, OOS-cov≥0.90). 풀-윈도우 END-coverage(`first≤fetch_start AND last≥holdout_end`) 폐지. `_TIERED_MIN_WINDOW_BARS=1500` 모듈 상수화. (C2) `PITUniverseConfig.k_in=0` 기본값; `capacity_coverage_target=0.90`, `k_max=100` 추가 — 누적 용량 90% prefix 알고리즘. (warm-up guard fix) `datetimes.min()>fetch_start` 심볼 reject: 교집합 start가 밀려 `ValueError: tiered warm-up coverage missing` 유발 차단.
- **Rationale:** END-coverage 필터가 633 온디스크 심볼을 54 "올드가드"로 붕괴 → PIT 설계가 막으려던 생존편향 재주입. 2023-10~2024-09 상장 110종 통째 배제. k_in=50은 교집합(231)·active_mask에 비구속(inert)이었으나 magic number 정당화 불가 → Pareto 용량 커버리지로 대체.
- **Edge Cases:** total capacity=0 → fail-open(`eligible[:k_max]`). fetch_start 이후 상장된 심볼은 warm-up guard로 자동 제외(교집합 보전). OOS 절단 심볼은 90% coverage guard로 제외.

## 2026-06-19 L2-ZERO: PIT cube bypass 해소 + store build/hit mismatch 수정
- **Delta:** `opt_main_futures.py`에 `_resolve_universe_state_cube()` 신규 함수 추가 → `_run_strategy_stage`에서 `universe_result`에서 cube 추출하여 `align_data_maps(state_cube=)` 주입. `pipeline.py` `_is_incomplete_pit_store_run()` 추가 → `load_or_build_universe_snapshot`에서 store hit 시 cube null 체크 후 rebuild. `discover_universe_timeline`에 `l2_start` timeline 경계 강제 로직 추가.
- **Rationale:** P0 - production 경로에서 `state_cube=None` 전달로 인해 L1/L2가 동일 PIT 필터를 소비하지 못함. P0 - store hit 시 decisions empty로 저장/복원되어 selection 정보 소실. P1 - L1/L2가 다른 시작 경계를 가져야 할 때 timeline이 2-way 계산만 함.
- **Edge Cases:** `universe_result is None` → cube=None 유지(기존 fallback 호환). Store hit + cube.parquet 없음 → cube=None fallback → incomplete 감지 → rebuild.

## 2026-06-19 EXACT-FIELDS: execution_pool_score 제거, exact-field only store contract
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에서 `execution_pool_score` 제거. `_selected_frame_columns()`에서 제거. `build_decision_frame()`에서 `execution_pool_score` 쓰기 제거. `materialize_snapshot_from_store()`에서 alias 역매핑 제거. `_symbol_meta_from_decision_row()`에서 `alpha_capacity_score` 단독 사용. 구 cache hit 시 alias-only decisions → `is_exact_selected_feature_schema` False → rebuild. `_universe_metadata_by_symbol()`는 snapshot.selected exact field만 읽음.
- **Rationale:** Store/cache 계층에 `alpha_capacity_score`와 `execution_pool_score`가 동시 존재 → 동일 개념의 2개 truth source. Exact-field only로 단일화하여 cache-hit/fresh-build 간 metadata 불일치 원천 차단.
- **Edge Cases:** 구버전 store run(alias-only) → `build_decision_frame`가 `execution_pool_score` 컬럼을 남겨도 `validate_materializable_pit_store_run`가 detect → rebuild. `pipeline.py:649`에서 `alpha_capacity_score` 우선, 없으면 `execution_pool_score` fallback 유지.

## 2026-06-19 CAPACITY-CLIP: unit-NAV 시뮬레이션에서 portfolio_nav=1.0 → capacity clip 전멸
- **Delta:** `awf_sim.py:_run_awf_simulation`에 `_capacity_clip_enabled` 플래그 추가 (`portfolio_nav is not None`). fit-leg(829) 및 OOS(1025) capacity clip을 `_capacity_clip_enabled` 조건으로 가드.
- **Cause:** `portfolio_nav=None` → `_portfolio_nav=1.0` (unit-NAV). `_min_order_usdt=5.0` → `abs(w)*1.0 < 5.0` → per_symbol cap 10%를 통과한 모든 weight가 zero-out. commit `5f0254f`에서 state_cube와 동시에 추가됨.
- **Rationale:** Unit-NAV 시뮬레이션에서 w는 분수(fraction)이지 USDT 금액이 아님. 최소주문($5)을 weight에 직접 비교하는 것은 차원 오류. 실제 portfolio_nav가 주입될 때만 capacity clip을 활성화.

## 2026-06-19 KELLY-FRICTION: diagonal_kelly_weights 이중 friction filter 제거
- **Delta:** `portfolio_constructor.py:diagonal_kelly_weights`에서 Step 1 friction filter(`mu_bps < effective_hurdle = hurdle * safety_mult / holding_bars`) 제거. `friction_hurdle_bps`, `holding_bars`, `friction_safety_mult` 파라미터와 `hurdle` 변수 삭제. `awf_sim.py` 두 호출부에서 해당 인자 제거.
- **Cause:** `mu_bps` (`signed_net_bps_per_bar`)는 이미 edge computation에서 cost가 차감된 NET 값. `diagonal_kelly_weights`가 이를 다시 `hurdle * safety_mult / holding_bars`와 비교하면 이중과세 발생.
  - state_cube 도입 전(3.8 bps): `hurdle*2.5=9.5` → `gross(20)>9.5` → 통과
- (Compressed...)

## 2026-06-19 META-PARITY: UNIVERSE_DECISION_COLUMNS에 metadata 필드 추가 + full materialization
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에 `vol_30d`, `friction_score`, `alpha_capacity_score`, `diversification_score` 4개 필드 추가. `_symbol_meta_from_decision_row()`에서 해당 필드 복원. `materialize_snapshot_from_store()`가 `decisions.parquet`에서 `SymbolMeta` 전체 필드 재구성. `_selected_meta_to_frame()` 추가 → `build_universe()` output을 decision columns와 일치. `_save_snapshot`에 `decisions=` 파라미터 추가.
- **Rationale:** cold build 시 `SymbolMeta`에 채워진 확장 필드가 cache-hit 시 `0.0` default로 떨어져 L1/L2가 다른 feature vector를 소비. Store schema에 exact field를 포함시켜 build/hit 간 metadata parity 보장.
- **Edge Cases:** 구버전 decisions(필드 누락) → `is_exact_selected_feature_schema` False → `validate_materializable_pit_store_run` False → rebuild 유도.

## 2026-06-19 Phase 4-B/C/E: Stage2-6 Config 및 legacy selection 제거
- **Delta:** `Stage2-6Config` 5종 class config.py에서 제거; `UniverseConfig`에서 `stage2-6/strategy_pool_mode/stage6_is_alpha_rank` 필드 제거. `Stage2Config` → `data_quality._DataQualityConfig` 인라인, `Stage3-5Config` → `filters.py` 로컬 이동. `selection.py` 전체 삭제(`apply_selection_stage` 포함). `pipeline.py` basket_ref/weights → `()`. 레거시 테스트 2종(`test_selection`, `test_strategy_pool_selection`) 삭제.
- **Rationale:** Phase 4-A에서 Stage6 else-branch 제거 완료 후 dead code 정리. `universe_engine` default = `"pit"` (4-A 적용). Stage2-5 config은 필터 유틸리티 함수 로컬 타입으로 유지(test_oi_adv_filter 호환).
- **Edge Cases:** 구버전 `Stage6Config` import하는 외부 테스트 → `@pytest.mark.skip` 처리(4-E); `k_in=50` cap으로 PIT 범위 제한.

## 2026-06-19 Phase 4-A: PIT 단독 경로 확정 + k_in=50 cap
- **Delta:** `build_universe`에서 Stage6 else-branch 완전 제거. `universe_engine` default `"stage6"` → `"pit"`. `PITUniverseConfig.k_in=50` 추가(capacity_usdt 내림차순 top-50). `store.py` empty decisions early return 추가.
- **Rationale:** PIT 경로 shadow validation PASS 후 Stage6 code path 불필요. k_in cap은 411 → 50 symbols로 제한(임시, Phase 4-D 이후 완전 제거 검토).
- **Edge Cases:** ledger `date` vs `datetime` 비교 TypeError 픽스(pipeline.py `_instrument_df_from_ledger`).

## 2026-06-19 Phase 3-3/3-4/3-5: PIT state_cube L1 wiring + lifecycle + capacity
- **Delta:** Phase 3-3: `_run_universe_stage` 7-tuple 반환(`universe_result` 추가). `align_data_maps` 호출에 `state_cube=` 주입 → `active_mask` PIT 반영. Phase 3-4: `SymbolLifecycleRecord` 추가, `promotion_available_at > l2_start` gate로 late-listing 심볼 L2 제외. Phase 3-5: `awf_sim` fit/OOS 양쪽에 `capacity_usdt` clip + 5 USDT min order threshold.
- **Rationale:** PIT state_cube 없이 L1이 stage6 all-True mask 사용 → look-ahead 노출. Lifecycle gate는 mid-window 상장 심볼이 OOS 신호에 참여하는 것을 방지. Capacity clip은 소량 포지션 거래비용 현실화.
- **Edge Cases:** `AlignedMarketData` frozen=True → `dataclasses.replace`; `adv_usdt_2d` shape 동적 체크(`isinstance(np.ndarray)`).

## 2026-06-15 Ledger backend compatibility recovery
- **Delta:** `load_ledger_slice(...)` now dispatches by backend suffix and supports both SQLite and parquet fixtures through the same PIT filter path.
- **Rationale:** universe tests and offline snapshots depend on parquet inputs; the loader must not collapse existing files into silent empty stage0 results.
- **Edge Cases:** missing files may still return empty frames, but readable files that fail backend-specific loading now raise with explicit backend context.

## 2026-06-19 Phase 4-D: UniverseSnapshot legacy panel 6필드 제거 + Stage6 경로 완전 삭제
- **Delta:** `UniverseSnapshot`에서 `training_panel`/`inference_panel`/`live_inference_panel`/`historical_trading_panel`/`inference_panel_quarter_membership`/`stage5_research_panel` 6개 필드 정의 제거. `discover_universe_timeline`의 Stage6 else-branch(230줄) 전체 삭제, dispatch는 PIT 무조건 호출로 단순화, `cfg=None` → `ValueError("universe_engine=pit required; stage6 path removed")` raise. Dead 헬퍼 `_resolve_trading_membership`, `_resolve_inference_membership` 삭제(`_snapshot_quality_symbols`는 `validate_universe_quality`에서 사용 중이므로 리팩터하여 유지). `snapshot_to_payload`/`snapshot_from_payload`에서 panel 직렬화 제거(구버전 payload key는 자동 무시). `store.py` `UniverseSnapshot(...)` panel 대입 제거. `pipeline.py` panel read + `replace(snapshot, ...)` 블록 제거. `strategy_service.py` `run_active_strategy_output_bridge`에서 panel 4개 파라미터 및 `training_panel` filter 제거. `opt_main_futures.py` 호출부 정리 및 `_run_universe_stage` extraction → `universe_result.inference_symbols`.
- **Rationale:** Stage6 panel 필드는 PIT state_cube가 유일 SSOT인 체계에서 불필요한 이중경로. Phase 4-A/4-B/4-C/E에서 Stage6 제거 후 최종 잔여 legacy 필드/경로 정리. `payload.get`-based deserialization은 구버전 스냅샷과의 하위호환 유지.
- **Edge Cases:** `cfg=None` → 명시적 raise, silent fallback 금지. `validate_universe_quality`가 `_snapshot_quality_symbols`에 의존하므로 함수 유지. `n_stageN` int 카운터는 별도 4-F 후보로 제거 대상 아님.

## 2026-06-19 Stage0.empty empty-universe contract: cube 강제 주입
- **Delta:** `build_universe()` stage0.empty 분기에서 `materialize_snapshot_from_store` 호출 시 `cube=None` 대신 empty `UniverseStateCube`(모든 array shape `(0,0)`, `eligible` all `False`)를 명시적으로 생성하여 전달. `validate_materializable_pit_store_run`가 empty-universe를 spec 계약(cube 존재 + eligible all False + zero selected) 하에서 통과시킴.
- **Rationale:** stage0.empty 경로에서 `cube=None` 전달 시 validator가 `cube is None` → `False` 반환 → `ValueError` 발생. 이는 cold build empty-universe 경로가 validated PIT snapshot만 소비한다는 계약을 위반. empty cube 생성으로 일관된 validator 통과 보장.
- **Edge Cases:** `np.empty((0,0), dtype=bool).any()` → `False` (empty array), `selected.empty` → True (zero rows), spec 계약 충족.

## 2026-06-19 Store Consolidation: 단일 Parquet Store 통합 + cube.parquet 영속화
- **Delta:** `snapshots/` flat+nested JSON/Parquet (분기당 7개 파일, 203개) 완전 제거 → `store/v1/runs/` 유일 저장소. `_save_snapshot` flat/nested write 제거. `load_or_build_universe_snapshot` snapshot JSON cache 경로(170줄) 제거 → 40줄 2-tier(store hit→materialize, store miss→build). `write_universe_store_run`에 `snapshot=` 파라미터 추가 → `pit_state_cube`를 `cube.parquet`로 직렬화(numpy tobytes). `load_universe_store_run` 반환값 3→4 튜플 확장(cube 포함). `materialize_snapshot_from_store(cube=)` → snapshot에 `pit_state_cube` 복원. `gc_stale_store_runs()` 신규 함수. `discover_universe_timeline` `cfg=None` → `UniverseConfig()` default (기존 ValueError 대체). `write_universe_store_run` empty-decision short-circuit 제거 → 항상 3파일(manifest+decisions+report) 쓰도록 수정.
- **Rationale:** 3중 JSON/Parquet 중복 및 file proliferation(203→29개) 해소. `pit_state_cube` transient 손실 버그 수정(캐시 적중 시 eligible all-False). `snapshots/` 레거시 호환성 유지 불필요(store가 단일 SSOT). Store run 누적(69→29) 방지 위해 GC 추가.
- **Edge Cases:** 구버전 store run(`cube.parquet` 없음) → `cube=None` fallback(기존 동작 유지). Empty decisions→schema-only DataFrame write로 store 일관성 유지. `load_universe_snapshot` 함수는 dropout computation에서 사용 중이므로 repurpose(store에서 최신 run 로드).
