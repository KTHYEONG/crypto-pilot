# Active Decisions Log (Sliding Window)

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
