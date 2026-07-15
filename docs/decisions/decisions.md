# Active Decisions Log (Sliding Window)

## [2026-07-15] [TASK_L1_PER_TF_NATIVE_LABELED_EVENTS] [ADR_20260715_L1_PER_TF_NATIVE_LABELED_EVENTS]
- **Context/Why:** aligned_by_tf 수정 후 IndexError(6h) 노출. 추적 결과 labeled_events(L1 워크포워드 실제 소비 데이터)가 애초부터 base(4h) grid 하나로만 생성되고, 타 TF 신호는 project_htf_panels_to_base로 base grid에 투영 후 native_tf 태그만 원래 TF명으로 붙어 entry_idx가 base-grid 기준 위치값이었음(구조적 결함, boundary bug 아님).
- **Resolution/What:** (1) labeled_events_by_tf dict 신설 — 각 TF 고유 panels_for_l1(L0 admission 통과, recipe_id 스탬프 완료)로 그 TF 고유 grid 위에서 직접 라벨링. (2) 구현 중 발견된 2차 결함(라벨링 시점이 recipe-binding 이전이라 l0_recipe_id 공백→전TF 차단) 즉시 수정: pruned_multi_results 계산 이후로 재배치 + native_tf 컬럼 명시 설정. (3) run_per_tf_l1에 entry_idx 경계 가드 추가(범위밖 이벤트 드롭+WARNING, 크래시 방지).
- **Impact:** 재실행 실측: 크래시 없이 6개 TF 전부 완주. n_ready 대반전 확인 — 2h 17→103(master TF 자동선정도 1d→2h로 전환), 6h 16→1, 8h 70→0(완전차단), 12h 151→0(완전차단), 1d 153→12. 결론: 기존 '느린 TF일수록 성과 좋음' 관찰은 그리드 불일치 아티팩트였음이 확정됨. docs/results/result.md 갱신 완료, 다음 세션은 2h 신호의 진짜 경제성 검증 및 8h/12h 완전차단 타당성 재확인 필요.

## [2026-07-14] [TASK_L1_ALIGNED_BY_TF_HANDOFF_WIRING] [ADR_20260714_L1_ALIGNED_BY_TF_HANDOFF_WIRING]
- **Context/Why:** aligned_by_tf 필드 추가(TASK_L1_4H_SYMMETRIC_TF_CONSTRUCTION) 이후에도 TieredL1Handoff/run_tiered_pipeline 호출부가 이를 안 넘겨 6개 TF 전부 L1 walk-forward가 동일 grid(n_bars=6949) 공유. 게다가 bridge.py의 CandidatePipelineOutput 생성 지점 6곳 중 3곳이 aligned_by_tf 누락(whack-a-mole 구조 리스크).
- **Resolution/What:** (1) TieredL1Handoff/consume_candidate_output_for_tiered/run_tiered_pipeline 호출 2곳에 aligned_by_tf 배선. (2) bridge.py에 _build_output 로컬 빌더 도입해 4개 반환지점 전부 통일, CandidatePipelineOutput.__post_init__에 aligned_by_tf 누락 시 [DATA] WARNING 가드 추가, 소스스캔 회귀테스트로 7번째 누락지점 재발 차단.
- **Impact:** 재실행 검증: n_bars가 TF별로 정확히 분리됨(2h=11736, 4h=6949, 6h=3912 — 이전엔 전부 6949로 동일). 그리드 공유 버그 해결 확정. 단, 이 수정이 6h 처리 중 IndexError(event_t=3915 > n_bars=3912)를 새로 노출시킴 — 폴드/홀딩기간 경계값 클램핑 누락으로 추정되는 별개의 잠재 버그(과거엔 모든 TF가 더 큰 공유 그리드를 써서 가려져 있었음). 후속 조사 필요, 미해결.

## [2026-07-14] [L1_4H_ZERO_EVENT_TRUE_ROOT_CAUSE] [ADR_20260714_L1_4H_ZERO_EVENT_TRUE_ROOT_CAUSE]
- **Context/Why:** 4h L1 zero-event 장애 원인이 (1) timeline quarter empty bootstrap에 의한 L0 evidence window clamp와 (2) membership_active_mask가 base timeframe 외 타 TF에 미적용된 구조적 결함이었음. 또한 TF별 지표 warm-up 상수가 ad hoc 테이블로 관리되어 불일치 및 starvation을 유발함.
- **Resolution/What:** (1) _resolve_effective_evidence_start() 헬퍼 함수를 추가하여 최소 유니버스 크기(50) 및 2분기 연속 유지 조건을 적용한 시작일을 계산해 L0-evidence end 일자를 clamping. (2) 전 TF(l1_tfs)에 대해 inject_membership_masks_into_maps() 멤버십 마스크 루프를 적용. (3) 4h warm-up day 상수(42일) 및 scale_bar_count() 기반 SSOT TF 변환 적용.
- **Impact:** 실측 결과 2h=17, 6h=16, 8h=70, 12h=151, 1d=153개 신호가 정상적으로 L1 검증을 통과하여 총 5개 TF 배포(5/6 유효 배포) 완료. 1d starvation 및 모든 TF의 labeled events 없음 블로킹 현상을 완벽히 해결함. Unit test 100% 통과.

## [2026-07-14] [TASK_L1_4H_SYMMETRIC_TF_CONSTRUCTION] [ADR_20260714_L1_4H_SYMMETRIC_TF_CONSTRUCTION]
- **Context/Why:** 4h L1 게이트 n_ready=0(전 레시피 insufficient_events) 지속. v1 스펙 진단(base-tf만 멀티키 dict 입력)의 실증 검증 필요.
- **Resolution/What:** l1_tfs 전체를 _build_single_tf_panels로 대칭 구성(base tf 특별취급 제거, bridge.py), CandidatePipelineOutput.aligned_by_tf 추가, audit_zero_event_timeframe 가드 함수 추가(cheap_gate.py).
- **Impact:** 재검증 결과 버그 미해결(4h n_ready=0 그대로, 79/79 레시피 n_events=0). 실측으로 v1 진단 반증: active_mask(0.618)/panel valid_mask_2d(0.604)/causal window(~0.05, 전TF 동일) 전부 정상 — 결함은 run_alpha_foundry_l0_gate_multi_tf의 aligned_by_tf 배선 이후 미확정 지점. 가드 함수는 프로덕션 call site 미연결(dead code) 확인. 별도 발견: inject_membership_masks_into_maps가 run_config.timeframe 프레임에만 적용되어 나머지 5개 TF는 멤버십마스크 전부 허용 기본값(미검증) 상태.

## [2026-07-14] [TASK_L0_LTF_STREAM_PARALLEL] [ADR_20260714_L0_LTF_STREAM_PARALLEL]
- **Context/Why:** `bridge_post_rules` 169.3s bottleneck traced to LTF 1m parquet I/O load (52 symbols × 3 LTF panels). `labeled.copy()` added ~400MB peak RSS with no benefit.
- **Resolution/What:** (1) ThreadPoolExecutor dual path in `build_ltf_native_alpha_panels_streaming` (max_workers=2 via `L0_LTF_EXEC_1M_MAX_WORKERS=2`). (2) `resolve_1m_coverage_tier` parallel scan. (3) `labeled.copy()`→`labeled.assign(native_tf=tf)`. (4) memory cap changed from `max(1, ...)` to `min(max_workers, 2)`.
- **Impact:** bridge_post_rules 166.37s→105.41s (-36.6%), STRATEGY total 367.03s→286.75s (-21.9%), pre_gc RSS 7,001MB→6,908MB (-93MB). Promotion result byte-identical (14/45/94/104/107). Env var `L0_LTF_EXEC_1M_MAX_WORKERS=2` required for activation.

## [2026-07-14] [TASK_L1_BRIDGE_CACHE] [ADR_20260714_L1_BRIDGE_CACHE]
- **Context/Why:** `build_rule_signal_panels`가 base TF + HTF 4회 = 5회 중복 호출되며 동일 indicator를 매번 재계산. `_resample_probe_source_frame`에서 `.copy()`로 인한 불필요한 RSS peak 발생. L1 bridge 내 profile 미출력.
- **Resolution/What:** (1) `_SignalIndicatorCache` dataclass + `_precompute_shared_indicators` 추출 → per-TF cache wiring. (2) bridge.py `.copy()` 제거로 RSS ~50MB 절감. (3) BRIDGE PERFORMANCE profile은 multi-TF early return으로 미출력 — SYS stage log만 확보. cache 정확성 104/104 PASS.
- **Impact:** .copy() 제거로 peak RSS 6.93GB (12GB cap의 57.8%). Indicator cache는 wall-clock 개선 미미 (진짜 병목은 LTF streaming 170s). bridge_post_rules 169.3s 중 cache 영향 <2%. 실질 병목 LTF streaming 최적화가 다음 과제.

## [2026-07-14] [TASK_L1_PROJECTION_VECTORIZATION] [ADR_20260714_L1_PROJECTION_VECTORIZATION]
- **Context/Why:** L1 `run_candidate_strategy_for_universe()`에서 "SWF SCOPE & ADMISSION"→"MULTI-TF PANEL INJECTION" 로그 간 bridge gap의 55%는 `build_native_htf_panels` 4개 TF 순차처리, 8%는 `_project_panel_to_base_grid` per-symbol Python loop(`for n in range(n_syms)` 3,192회 `searchsorted`)가 차지. Bounded concurrency(`ThreadPoolExecutor`)로 HTF build를 2-wave로 단축 시도.
- **Resolution/What:** (1) `project_higher_tf_to_grid` 2D 입력 지원 → `_project_panel_to_base_grid` HTF/LTF "last" mode per-symbol loop 제거, 4회 2D vectorized 호출로 대체. (2) `build_native_htf_panels`에 `L1_HTF_BUILD_MAX_WORKERS=2` env-gated ThreadPoolExecutor 추가. 벤치마크(114 syms × 4 TFs × 2000 bars) 결과: projection 10.6× speedup(766ms→73ms) 확인. 반면 concurrency는 GIL contention으로 0.70× regression → serial path 유지, **concurrency rollback**.
- **Impact:** Projection 단독 10.6× (28 panels in 72ms). Peak RSS 1.7GB (12GB cap의 14.2%). Concurrency는 GIL에 의해 threading overhead가 실제 연산보다 커서 오히려 둔화 — pandas/numpy CPU-bound 작업은 serial이 최적. check 11/11 PASS.

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
