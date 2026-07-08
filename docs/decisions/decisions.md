# Active Decisions Log (Sliding Window)

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


