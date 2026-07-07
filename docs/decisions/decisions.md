# Active Decisions Log (Sliding Window)

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
