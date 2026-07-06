# Active Decisions Log (Sliding Window)

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
