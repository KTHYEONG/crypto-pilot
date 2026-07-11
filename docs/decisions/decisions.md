# Active Decisions Log (Sliding Window)

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
