# Active Decisions Log (Sliding Window)

## [2026-07-14] [TASK_L1_ZERO_SIGNAL_REGRESSION] [ADR_20260714_L1_ZERO_SIGNAL_REGRESSION]
- **Context/Why:** ADR_20260714_L1_MEMORY_EXECUTION 이후 6개 TF 전부 labeled delivery 없음으로 gate 차단, L0는 57건 통과했으나 L1 도달 신호 0건.
- **Resolution/What:** (1) assemble_l0_strategy_delivery_manifest: floor 붕괴 시 final_selected_recipe_ids만 치유되고 routes는 미치유되던 불일치를 fail-open 통일로 수정. (2) bridge.py: raw_events.empty 조기 반환이 이미 계산된 _multi_tf_htf_panels를 검사 없이 폐기하던 문제를 HTF-only 라벨링 fallback으로 수정.
- **Impact:** 동일 실행(--phase l1 --timeframe 4h --sync skip) 재검증: 6개 TF 전부 delivery 정상 도달, 2h=14/4h=52/6h=163/8h=261/1d=146건 총 636건 신호 승격, exit 0.

## [2026-07-14] [TASK_L1_MEMORY_EXECUTION] [ADR_20260714_L1_MEMORY_EXECUTION]
- **Context/Why:** 실제 18GB 환경에서 multi-TF 패널 family 동시 계산과 fork worker가 peak memory를 증폭했고, handoff 출력 계약도 aligned 누락으로 L1을 중단시켰다.
- **Resolution/What:** rule family와 native TF 패널을 순차 생성하고 L0 gate를 단일 worker로 제한한다. parent-inclusive PSS planner와 destructive CandidatePipelineOutput handoff를 연결하며 signal-only 경로도 aligned를 반환한다. dual event schemas와 LTF logging 계약을 정합화한다.
- **Impact:** 최종 실행은 125개 데이터 중 114개 admission, 6개 TF L1 루프까지 RSS peak 약 8.32GB로 완료되었으나 모든 TF는 labeled delivery 없음으로 gate 차단되었다. 실행 시간 3분24초, exit 0.

## [2026-07-13] [TASK_L1_HYBRID_MEMORY_AUDIT] [ADR_20260713_L1_HYBRID_MEMORY_AUDIT]
- **Context/Why:** 1m 데이터를 on-demand hybrid 방식으로 전환했지만 전체 L1 RSS 절감 목표가 달성되지 않았고, multi-TF 패널과 nested worker가 실제 병목인지 실측 결과를 SSOT에 기록할 필요가 있다.
- **Resolution/What:** core loader의 1m 전수 적재 제거는 유지한다. 6개 TF 패널 동시 보유와 L1 worker fork를 전체 메모리 병목으로 확정하고, 후속 개선 대상으로 panel 수명 단축·TF 순차 해제·worker/IPC 상한 조정을 등록한다.
- **Impact:** 1m 저장 효율 개선과 전체 L1 메모리 안정성은 별도 문제로 관리한다. 2026-07-13 기준 정상 L1 peak 약 11.10GB이며, 충분한 메모리 환경에서는 6개 TF gate 완료가 가능하다.

## [2026-07-13] [TASK_L1_1M_COVERAGE_WARMUP] [ADR_20260713_L1_1M_COVERAGE_WARMUP]
- **Context/Why:** 1m files are intentionally sparse for storage efficiency, so execution readiness must be evaluated against the admitted L1 scope and the actual warmup-to-holdout interval rather than the full universe.
- **Resolution/What:** Keep core loader 1m-free; LTF streaming receives only admitted symbols, computes coverage over the aligned interval, and plans only symbols meeting the configured 0.80 coverage floor and memory cap. For 2026-07-13, 52 of 114 admitted symbols are LTF-covered.
- **Impact:** The current coarse L1 run is not blocked by 1m absence; intrabar/LTF breadth is limited to 52 symbols. Full 1m backfill is required only when broader LTF coverage is explicitly desired.

## [2026-07-13] [TASK_L1_DEPLOYMENT_PASS_CONTRACT] [ADR_20260713_L1_DEPLOYMENT_PASS_CONTRACT]
- **Context/Why:** l1_structural_gate_only=True could build a non-empty deployment registry while Layer1Result.gate_passed still exposed legacy strict advisory failure; multi-TF aggregation also separated PASS selection from the registry delivered to L2.
- **Resolution/What:** Define deployment PASS as configured policy gate plus a non-empty registry, restrict automatic master-TF candidates to deployable results, and make aggregate gate and registry originate from the same selected TF. Preserve strict Layer1GateReport.passed and all economic gates.
- **Impact:** 4h conditional deployment can be represented truthfully without relaxing economics; empty or blocked registries fail closed; diagnostics expose strict, structural, and advisory status. Targeted contract tests and lean check pass.

## [2026-07-13] [TASK_L1_FAMILY_ADMISSION_INVESTIGATION] [ADR_20260713_L1_FAMILY_ADMISSION_INVESTIGATION]
- **Context/Why:** 직전 ADR에서 4h 실패를 '순수 비정상성'으로 결론지었으나, 신규 계측(l1_registry_overlap_diag)이 이를 반증 — 동일 family가 4개 폴드 전부에서 일관되게 배제되는 구조적 패턴 발견. eff_n 계산이 TF 세밀도에 반비례해 작동하는 버그 가설 수립.
- **Resolution/What:** l1_family_admission_diag 신규 계측(family별 eff_n/n_obs 비율 + structural_reasons 분포) 추가·실행. 결과: eff_n/n_obs=0.83~0.94로 전 TF 유사(계산 버그 가설 반증). 실제 탈락 사유는 no_incremental_edge/negative_gross_edge(순수 경제성) — 4h/6h/8h는 228쌍 중 130~170건 탈락, 12h는 33~74건으로 실제 개선. 진짜 경제적 성과 차이 확정.
- **Impact:** 가설 2회 연속 반증(activation_context 불일치 → eff_n 계산버그) 후 최종 확정: dual_momentum/taker_imbalance_momentum은 일중 시간단위(4h/6h/8h)에서 진짜로 초과수익 없음, 12h부터 개선. 추가 코드 수정 없음(과적합 방지) — 4h는 현재 상태(l1_structural_gate_only=True 부분배포)가 정직한 최종선. check 9/9 PASS, 회귀 없음. SSOT: docs/results/result.md.

## [2026-07-13] [TASK_L1_4H_FOLD_COLLAPSE_REMEDIATION] [ADR_20260713_L1_4H_FOLD_COLLAPSE_REMEDIATION]
- **Context/Why:** 4h만 L1 실패(fold_ratio 1/4)하는 게 TF별 편향인지 점검 필요. per-fold 신규 진단(l1_per_fold_diag) 실측 결과 4h는 4개 outer-fold 중 2개(fold2/3)가 registry_empty(예측 0건)로 완전공백 — 6h는 1개, 2h/8h/12h/1d는 0~1개. 동일 판정함수가 전 TF에 적용되며 임계값 조작 없음 확인, 진짜 시장 비정상성.
- **Resolution/What:** l1_structural_gate_only 기본값을 False→True로 전환(코드 1줄, 이미 검증된 opt-in 메커니즘 활성화). override 신설 등 원인 아닌 것을 고치는 변경은 반려.
- **Impact:** 실측 재실행 결과 4h n_ready 0→3(부분 배포), 2h/6h/8h/12h/1d n_ready 완전 동일(17/13/34/84/111, 무변화) — 안전성 재확인. gate_report.passed(엄격판정)는 여전히 False, fold_ratio 근본원인(fold2/3 완전공백)은 미해결(과적합 없이는 해소 불가, 후속 조사 필요). check 통과, 회귀 없음. SSOT: docs/architecture/layer1.md.

## [2026-07-13] [TASK_L1_READINESS_GATE_REDESIGN] [ADR_20260713_L1_READINESS_GATE_REDESIGN]
- **Context/Why:** match_ratio가 실제로는 (decision_idx,symbol,strategy_id,activation_context) 4키 정확조인 성공률로, 성과지표가 아닌 조인 아티팩트였음. fold_ratio는 n=4 고정폴드라 5개 이산값뿐인데 TF별 임계값(0.40~0.60)으로 비교해 통계적으로 무의미. 전체-TF AND게이트가 이미 존재하는 per-strategy 세밀평가(build_qualified_signal_registry)를 통째로 봉쇄.
- **Resolution/What:** align_outer_opportunities_with_realized에 3키 재병합으로 label_drift/true_unmatched 분리. match_ratio를 pooled count + Wilson LCB로 재계산(probe_lcb_bps와 동일 패턴). Layer1GateReport에 structural_passed(fold_cov/sym_count/probe_lcb_bps)/advisory_checks(match_ratio/fold_ratio) 분리, l1_structural_gate_only(기본 False) 플래그로 opt-in 배포.
- **Impact:** 실측(2026-07-13 18:xx) 기본값(flag off)만으로 6h 완전 해제(n_ready 0→13, blockers none) — match_ratio가 진짜 false negative였음을 증명. flag on 시 4h도 부분 해제(0→3, fold_ratio만 잔존, 진짜 불안정성). check 89/89+14 PASS, 회귀 1건(레거시 compat 픽스처, structural_passed로 정정). SSOT: docs/architecture/layer1.md, docs/decisions/decisions.md.

## [2026-07-13] [TASK_L0_L1_ASSET_GROWTH_RESTRUCTURE] [ADR_20260713_L0_L1_ASSET_GROWTH_RESTRUCTURE]
- **Context/Why:** L0 준비도 62/100(직전 ADR) — 6개 TF 중 3개만 배포, 28개 family 중 다수가 여러 세션째 통과율 0%(cross_sectional/carry/flow/mean_reversion), cross_tf_pruning이 audit와 AND로 묶여 배치최적화 미적용(130s 낭비).
- **Resolution/What:** `DEFAULT_L1_TFS`에서 `1h` 제거(구조적 붕괴, 회생 불가)·네이티브인데 미사용이던 `1d` 승격. `family_lifecycle.RETIRED_FAMILIES`(14종) 신설, `resolve_tf_signal_pool`과 `build_alpha_recipe_catalog`(base+htf 4개 호출부) 양쪽에 배선 — check 단계에서 후자 배선 누락을 재발견·수정(config.py만 고쳐서는 evidence에 그대로 남아있었음). `bridge_helpers.py` shared-context 게이트를 AND→OR로 완화.
- **Impact:** 실측(`4h_1783927361`) — 배포 가능 TF 3/6(50%)→**4/5(80%)**, `1d` n_ready=111(최고 성과, master_tf가 12h→1d로 전환), 평가 family 28→14(durable-zero 완전 제거, 배포결과 무변), `l0_cross_tf_pruning` 130s→11.3s, 전체 wall-clock 337.84s→261.17s. 4h/6h는 근소 미달(match_ratio 0.50/0.75)로 제거 보류, Phase 2 대상. SSOT: `docs/architecture/layer0.md`, `docs/results/result.md`.

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
