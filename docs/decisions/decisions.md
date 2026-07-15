# Active Decisions Log (Sliding Window)

## [2026-07-15] [TASK_L0_L1_NET_EVIDENCE_REPLAY] [ADR_20260715_L0_L1_NET_EVIDENCE_REPLAY]
- **Context/Why:** 최신 control replay에서 실제 데이터는 정상 로드되었지만 2h만 L1을 통과했고, 느린 TF는 registry 공백·fold coverage·음의 net edge로 차단되었다. gross-only pooled LCB는 데이터 부족과 경제성 실패를 혼동할 수 있었다.
- **Resolution/What:** L1 pooled LCB에 execution cost를 반영하고 음의 경제 fold를 보존하며, support blocker만 제외하도록 evidence policy를 연결했다. L0에는 cutoff 검증을 포함한 L1 causal feedback multiplier 경로를 연결하고 동일 실행 결과 재사용은 금지했다.
- **Impact:** 2h는 4/4 fold와 55.999bps로 PASS, 4h는 18.990bps이나 coverage 부족으로 BLOCKED, 8h는 -35.095bps로 BLOCKED, 6h/12h/1d는 registry_empty로 BLOCKED. capacity 관측과 동적 funding/slippage 및 prior-period feedback replay는 후속 과제로 남는다.

## [2026-07-15] [L0_TF_PROBE_DEFAULT_DISABLED] [ADR_20260715_L0_TF_PROBE_DEFAULT_DISABLED]
- **Context/Why:** 정식 L0 multi-TF gate와 L1 검증이 tf-probe와 독립적으로 동작하지만, probe가 phase=l1에서 자동 실행되어 0 winning 결과와 L1 지표를 혼동시키고 불필요한 계산을 유발했다.
- **Resolution/What:** OPT_FUTURES_CONFIG와 AlphaFoundryRuntimeConfig의 tf-probe 기본값을 False로 통일하고 active pipeline의 기본 실행 조건을 opt-in으로 변경했다. 명시적 활성화 없이는 telemetry probe를 실행하지 않으며 L0/L1 admission은 canonical multi-TF 결과만 사용한다.
- **Impact:** 실측에서 tf-probe 0 winning과 무관하게 canonical L0 43개 recipe, L1 1h/2h PASS가 확인됐다. 기본 실행은 probe 없이 L0/L1 계산량과 결과 해석을 안정화하며, legacy probe 테스트는 명시적 opt-in 경로에서만 유효하다.

## [2026-07-15] [L1_PAIR_GATE_TF_DENSITY_CALIBRATION] [ADR_20260715_L1_PAIR_GATE_TF_DENSITY_CALIBRATION]
- **Context/Why:** registry_empty(4h/6h/8h/12h/1d fold#1 이후 붕괴)의 진짜 원인을 실행 계측으로 추적한 결과, config.py _DEFAULT_PER_TF_GATE_OVERRIDES의 l1_pair_min_effective_obs가 TF 속도와 반대 방향(1h=3.0→2h=4.0→4h=5.0(누락폴백)→6h=5.0→8h=5.0→12h=6.0→1d=7.0)으로 설정되어 있었음. 실측(fold별 evidence/registry 프로브, bootstrap probability_positive 분포)으로 raw evidence row 수는 TF간 비슷한데 pair당 관측치(n_obs)는 느린 TF일수록 자연히 적음(2h fold3 중앙값 100 vs 12h 21)에도 불구하고 문턱값은 오히려 높게 요구되어 4h~1d가 구조적으로 거의 통과 불가능했음이 확인됨.
- **Resolution/What:** scripts/calibrate_l1_pair_gate.py 신규 작성 — 측정(control replay + effective_n_sink 훅)과 채택(config.py 수동 반영)을 분리해 quant.md anti-overfitting 원칙 준수. signal_selection.py의 compute_symbol_strategy_evidence에 선택적 effective_n_sink 파라미터 추가(기본 None, 기존 동작 불변). 최초 구현에 크래시 2건 발견 및 수정: (1) run_once(trace={})가 STAGE_ORDER 미시딩으로 크래시 — trace 사전 시딩으로 수정. (2) pipeline.py가 compute_symbol_strategy_evidence를 자체 이름으로 재import하므로 signal_selection 모듈 패치는 무효였음(측정치 전부 빈 값) — pipeline 모듈 자신의 바인딩을 패치하도록 수정. 두 회귀 모두 재현 테스트로 고정. 실제 control replay 재실행으로 6-TF 전체 effective_n p10 실측 후 config.py의 4h/6h/8h/12h/1d l1_pair_min_effective_obs를 전부 4.0(2h 기존값=ceiling)으로 갱신.
- **Impact:** 실측: 6개 TF 전부 effective_n p10(4.9~28.4)이 ceiling(4.0)을 초과해 전부 4.0으로 수렴 — 어떤 TF도 2h보다 엄격한 문턱값을 요구받지 않게 되어 역방향 스케일링 버그 제거. lean_check 전 파일 PASS(calibrate_l1_pair_gate.py 격리 커버리지 96%). L0 TF-probe 별도 조사에서 _TF_PROBE_FALLBACK_SYMBOLS(BTC/ETH/BNB)가 실제로는 BNBUSDT 단일 심볼로 조용히 축소되는 별개 이슈 발견(BTC/ETH가 유니버스 선정 정책상 거래 유니버스에서 원천 제외됨) — 이번 task 범위 밖으로 별도 기록, 후속 조사 필요.

## [2026-07-15] [L0_L1_DIAGNOSTIC_PIPELINE_INTEGRITY] [ADR_20260715_L0_L1_DIAGNOSTIC_PIPELINE_INTEGRITY]
- **Context/Why:** run_l1_cross_tf_replay.py가 RunnerResult를 폐기(exit 항상 0)하고 canonical 10-stage 중 terminal_event_audit/outer_folds 2개를 캡처하지 않았음. cross_tf_diagnostics.py의 diagnose_snapshots/write_cross_tf_diagnosis 정식 계약은 유닛테스트 fixture 외 어디서도 호출되지 않는 고아 코드였고, run_tiered_pipeline_outcome의 diagnostic_sink 파라미터는 정의만 되고 미사용, RunnerResult는 models.py/active_pipeline.py에 이중 정의되어 매 호출마다 상호 변환되고 있었음.
- **Resolution/What:** run_once()가 caller-owned trace dict를 참조로 받아 RunnerResult와 STAGE_ORDER 10개 전체(신규: outer_folds, terminal_event_audit)를 기록하도록 수정, main()이 RunnerResult.exit_code를 프로세스 exit code로 반영하고 예외 시에도 partial trace를 보존하도록 변경. cross_tf_diagnostics._STAGE_ORDER를 공개 STAGE_ORDER로 승격하고 snapshot_from_raw_stage_entry() 어댑터를 신설. scripts/run_l1_cross_tf_diagnosis.py 신규 작성 — control/control_repeat/treatment/fusion_ablation 4-run을 순차 supervisor로 실행하고 diagnose_snapshots()/write_cross_tf_diagnosis()에 실제로 연결. run_tiered_pipeline_outcome의 미사용 diagnostic_sink 파라미터 제거. RunnerResult 이중정의를 models.py 단일 클래스로 통합(active_pipeline.py는 이를 import), runner/pipeline.py의 불필요한 재포장 제거. L0 probe(probe_manifest)는 L2 master TF 선정에만 쓰이고 L1 admission을 게이트하지 않는 기존 설계를 그대로 유지(범위 밖으로 명시).
- **Impact:** control replay 재실행으로 실측 검증: artifact에 runner_result + 10/10 stage 전부 기록됨(이전엔 8/10 + RunnerResult 없음), 2h n_valid=74/fold edge 수치가 result.md와 완전 일치(회귀 없음). 4h/6h/8h/12h/1d는 여전히 registry_empty로 BLOCKED — 이 fold 판정 로직 자체는 이번 범위 밖(과거 조사에서 진짜 시장 비정상성으로 판정됨). L0 probe 0 winning cells vs 2h L1 독립 PASS가 동일 실행에서 재확인되어 L0/L1 분리 설계를 실측 뒷받침. lean_check 전 파일 PASS(신규 진단 파일 커버리지 90~92%).

## [2026-07-15] [L0_L1_CONTROL_REPLAY_RESULT_20260715] [ADR_20260715_L0_L1_CONTROL_REPLAY_RESULT_20260715]
- **Context/Why:** 최신 control 단일 순차 실행은 2h만 L1 PASS이고 나머지 TF는 fold-level registry/경제성 게이트로 BLOCKED였다. 그러나 replay artifact에 terminal_event_audit와 outer_folds가 없고 RunnerResult가 process 성공으로 변환되어 결과 완전성과 종료 상태를 신뢰할 수 없었다.
- **Resolution/What:** docs/results/result.md를 최신 control 측정값으로 전면 교체하고, 불완전 artifact는 cross-TF 인과 결론에 사용하지 않는다. 다음 실행 전 RunnerResult 전달, terminal/outer checkpoint, signal/RSS/last-stage 보존을 요구한다.
- **Impact:** 2h 결과만 현재 유효 측정으로 기록한다. 1h가 6h/12h에 미친 영향, OOM 여부, cross-TF 최초 divergence는 미판정으로 유지하며 계측 보강 후 네 run을 순차 재실행한다.

## [2026-07-15] [L0_L1_RUNTIME_TERMINAL_OBSERVABILITY] [ADR_20260715_L0_L1_RUNTIME_TERMINAL_OBSERVABILITY]
- **Context/Why:** After the policy refactor, the CLI passed alpha_foundry=None as the string 'None', the replay utility imported a removed builder, and a None strategy-stage return could be interpreted as successful L1 completion. A sequential rerun reached 106 loaded symbols and 241/241 TF readiness but terminated before L0/L1 artifacts were emitted.
- **Resolution/What:** Normalize omitted runtime flags at the canonical policy boundary, migrate the replay utility to build_effective_run_config, and convert zero-delivery, blocked-tiered, contract, and missing strategy-stage paths into explicit RunnerResult failures. Preserve single-process execution and do not promote incomplete measurements.
- **Impact:** The configuration/import defects are removed and targeted tests pass. The latest run provides only readiness evidence (2023-07-31~2026-03-31, OOS 2025-10-01); L0 candidate counts, terminal-event audit, L1 folds, and treatment comparisons remain unavailable because execution ended before strategy/L1 completion.

## [2026-07-15] [L0_L1_NATIVE_CONTRACT] [ADR_20260715_L0_L1_NATIVE_CONTRACT]
- **Context/Why:** The corrected sequential replay activated L0, while the control stopped on six terminal 2h boundary events; the earlier zero-L0 result came from an inactive runtime configuration. Native event identity and failure visibility must be recorded before treatment conclusions.
- **Resolution/What:** Established a canonical FuturesRunConfig, enforced active L0 gate mode for L0/L1, added native event-grid validation and explicit cross-TF diagnostic artifacts, and retained single-process replay as the memory-safe default. The current control remains incomplete until terminal-maturity handling is wired into the L1 consumer.
- **Impact:** Observed L0 candidate counts are now real and route-consistent; 2h L1 delivery reached 133740 native events before the terminal-boundary contract stopped the run. No treatment comparison, deployment threshold, or cross-TF causal conclusion is promoted.

## [2026-07-15] [L1_TF_COVERAGE_1H_REINTRO] [ADR_20260715_L1_TF_COVERAGE_1H_REINTRO]
- **Context/Why:** L0→L1 병목 재진단으로 l1_min_matched_events_per_fold=20 TF-불변 플랫 상수를 8h/12h 붕괴 용의자로 지목, 1h 재도입의 전제조건(LIMIT-05) 충족 확인 및 LIMIT-06 밀도 정규화 세이프가드 구현. **[check 단계 반증, 중요]**: 1h 추가 전/후 격리 재실행 결과 matched_events 스케일링은 **단독으로는 효과 없음**(1h 없이 재실행 시 6h/8h/12h 전부 스케일링 이전 원값과 완전 일치) — 12h sym_count 개선(1.0→3.0)과 6h 회귀(PASSED→BLOCKED)는 전부 **1h를 l1_tfs에 추가한 것 자체의 부작용**(정확한 인과 경로 미규명, seed는 tf_idx 무관 확인됨 — cross-TF 공유 연산 의심되나 미확정)이었음. "확정"이라는 원 서술은 부정확했음.
- **Resolution/What:** DEFAULT_L1_TFS에 '1h' 추가, Layer1FoldReadiness에 bars_per_fold_native/decision_points_per_calendar_year 진단 필드 추가, _TF_SCALE_NAME_PATTERNS에 _per_fold/_events_per_fold 가드 패턴 확장, l1_min_matched_events_per_fold에 tf_scale_base 메타데이터 태깅(효과 미확인이나 회귀 리스크 없어 유지). _bars_per_year_for_tf 중복 정의(365.25일 기준) 발견해 기존 SSOT(tiered_workflow.metrics, 365일 기준)로 통합. candidate_contracts.py 1:1 테스트 파일 누락 보완.
- **Impact:** 1h L0/L1 배선 기존 per-TF 설정 재사용(신규 설정 불필요). 1h 자체는 probe_lcb_bps=4.58bps(<7.5 breakeven)로 구조 게이트 최종 실패 — Symbol-Breadth(20.8)는 밀도 덕에 쉽게 통과하나 경제성 게이트가 독립적으로 걸러냄(LIMIT-06 세이프가드 설계 의도 검증됨). 진단 필드는 게이트 판정 무영향(순수 리포팅). **미해결 후속 과제**: 1h 추가가 6h/12h 결과를 바꾸는 정확한 cross-TF 인과 경로 규명 필요(별도 조사 필요, 이번 세션 범위 밖).

## [2026-07-15] [TF_SCALED_CONFIG_FIELD_GOVERNANCE] [ADR_20260715_TF_SCALED_CONFIG_FIELD_GOVERNANCE]
- **Context/Why:** max_holding_bars(4h 기준 36bar 상수)를 _resolve_block_bars_eff에서 미스케일 재사용해 1d 부트스트랩 block이 6배(72bar) 폭증, n_ready 12→0 회귀 재현(3/3). 전수 감사 결과 config.py 전역에 동일 패턴(base-TF 캘리브레이션 상수 vs TF-네이티브 값 구분 컨벤션 부재)이 15개+ 필드에 퍼져있음 확인(RegimeConfig 클러스터, channel_bars/lookback_bars 등).
- **Resolution/What:** dataclasses.field(metadata={"tf_scale_base": "4h"|None})로 5개 dataclass 전체 bar-duration 필드 명시 분류. apply_tf_gate_overrides(config.py)에 스케일링 로직 통합(기존 2개 호출부 자동 수혜, 하위 함수 시그니처 무변경). run_tiered_pipeline의 build_l1_nested_swf_folds 호출부(fold 경계/purge 계산)에 신규 apply_tf_gate_overrides 호출 추가. 신규 필드 미분류 시 실패하는 구조 테스트 추가. min_listing_age_days(달력일수, bar-count 아님)의 오분류도 리뷰 중 발견해 tf_scale_base=None으로 정정.
- **Impact:** 재실행 실측: 1d n_ready 0→12(완전 복원, 버그 이전 베이스라인과 Symbol-Breadth/probe_lcb_bps 정확히 일치). 2h/4h/6h/8h/12h 무변화. 회귀 47/47 통과. 백로그 6개 필드(l1_evidence_lookback_bars, score_pct_variant_hist_window_bars, RegimeConfig 클러스터 등)는 메타데이터 태깅만 완료, 소비부 마이그레이션은 후속 스펙 필요.

## [2026-07-15] [L1_TF_BIAS_GATE_CALIBRATION] [ADR_20260715_L1_TF_BIAS_GATE_CALIBRATION]
- **Context/Why:** per-TF native grid 수정 이후 2h만 압도적(n_ready=103, probe_lcb_bps=108.2) 결과 관측. 코드 감사 결과 l1_bootstrap_block_bars(6, bar-count 고정)가 TF/보유기간 미스케일, l1_sym_count_mode=effective_n이 TF별 sym_count 오버라이드를 우회(전 TF 공통 3.0 적용), probe_lcb_bps 구조 게이트가 breakeven(round-trip cost) 미반영(>0.00)임을 확인.
- **Resolution/What:** signal_selection.py 5개 moving_block_bootstrap_mean 호출부에 _resolve_block_bars_eff 도입, config.py _DEFAULT_PER_TF_GATE_OVERRIDES에 l1_min_effective_sym_n(1h/2h=5.0) 추가, evaluate_layer1_readiness의 probe_lcb_bps 임계값을 max(l1_min_probe_bps, l1_breakeven_floor_bps)로 교정.
- **Impact:** 재실행 실측(--phase l1 --timeframe 4h): 2h n_ready 103→101(소폭 감소, 여전히 마스터 TF), 새 임계값(Symbol-Breadth≥5.00, probe_lcb>7.5bps)에서도 여유 통과(19.6/82.6bps). 4h/6h/8h/12h 무변화(오버라이드 대상 아님, 예측대로). 1d는 max_holding_bars TF-미스케일 2차 버그로 12→0 회귀 발견(후속 ADR 참조).

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
