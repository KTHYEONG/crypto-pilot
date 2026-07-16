# Active Decisions Log (Sliding Window)

## [2026-07-16] [L1_SNAPSHOT_FDR_DECOUPLING] [ADR_20260716_L1_SNAPSHOT_FDR_DECOUPLING]
- **Context/Why:** baseline-mode fix 이후에도 6h/8h(구조적 probe_lcb_bps 음수)와 1d(구조 PASS했지만 registry_empty로 0건 승급)가 여전히 BLOCKED. 원인 추적 결과 l1_fdr_hard_reject=True(기본값)가 walk-forward 스냅샷 admission과 최종 deployment admission에 동일하게 적용되어, 반복적 예비 스크리닝 단계에 최종 1회 의사결정용 강한 다중검정 보정이 그대로 걸려 얇은 초기 표본을 가진 느린 TF의 fold가 통째로 registry_empty로 침묵됨을 확인.
- **Resolution/What:** compute_symbol_strategy_evidence에 fdr_hard_reject_override 파라미터 추가(baseline_mode_override와 동일 우선순위 패턴). pipeline.py의 walk-forward 스냅샷 호출부에만 fdr_hard_reject_override=False 전달(soft-scale로 완화), deployment 호출부는 미변경(strict 유지).
- **Impact:** 실측 control replay 결과: 8h가 BLOCKED(probe -147.857)에서 PASS(25 signals, probe +27.4)로 완전 전환. 6h/1d는 구조적 게이트 전부 clean해지고 probe_lcb_bps가 강하게 양전환(6h -40.5->+13.6, 1d +88.2 유지)했으나 deployment 단계 FDR(의도적으로 strict 유지)에서 여전히 0건 승급 -- 남은 병목이 deployment 단계의 통계적 검정력 부족(정직한 데이터 부족)임을 확인, 추가 완화는 p-hacking이라 보류. 2h/4h/12h는 숫자까지 완전 동일(86/32/18)하여 회귀 없음 확인. lean_check PASS, spec contract 4개 시나리오 전부 구현.

## [2026-07-16] [L1_BASELINE_FAMILY_SCOPED_ADMISSION] [ADR_20260716_L1_BASELINE_FAMILY_SCOPED_ADMISSION]
- **Context/Why:** result.md Tier-3 실측(avg_corr: 1d=+0.82, 2h=+1.0, 4h=+0.38, 12h=-0.0065/avg_peers=1.0)이 12h/4h의 no_incremental_edge 탈락 다수가 무관한 패밀리 피어와의 비교로 인한 무고한 washout임을 입증. 최초 구현(cfg.l1_baseline_mode 기본값을 peer_exclusive_family로 전역 변경)은 실측 검증 없이 deployment 뿐 아니라 walk-forward 스냅샷 시그널 선택까지 바꿔 4h(PASS->BLOCKED, probe -15.0)/12h(PASS->완전붕괴, sym_count=0 probe=-inf) 회귀를 실측 재실행으로 발견함.
- **Resolution/What:** compute_symbol_strategy_evidence에 baseline_mode_override 파라미터 추가(전달 시 cfg.l1_baseline_mode보다 우선). pipeline.py의 deployment_evidence 호출부에만 baseline_mode_override='peer_exclusive_family' 명시 전달. cfg.l1_baseline_mode 기본값은 legacy 'peer_exclusive'로 원복하여 walk-forward 스냅샷 admission(probe_lcb_bps 구조 게이트를 만드는 단계)은 영향받지 않도록 격리.
- **Impact:** 실측 control replay 재실행 결과: 6h/8h의 probe_lcb_bps가 원본과 소수점까지 완전 일치(-40.459, -147.857)하여 스냅샷 단계 회귀 완전 해소 확인. 4h는 PASS 복귀(32 signals), 12h는 PASS 유지하며 11->18건으로 개선(+64%). 2h/1d는 변화 없음(의도대로). lean_check PASS, spec contract 8개 시나리오 전부 구현.

## [2026-07-16] [TASK_L1_DEPLOYMENT_ADMISSION_GAP] [ADR_20260716_L1_DEPLOYMENT_ADMISSION_GAP]
- **Context/Why:** 1d/12h에서 pooled TF-level LCB는 강력한 양수이나 개별 후보 승급이 0건 혹은 극소수인 원인을 실측 분석하고, missed adaptive LCB quantile 버그 수정
- **Resolution/What:** 1. metrics.py에 resolve_lcb_quantile을 공용 함수로 이전하고, signal_selection.py의 compute_symbol_strategy_evidence 내 hardcoded 0.05 quantile을 adaptive quantile로 교체 (Tier 2). 2. no_incremental_edge로 탈락한 gross-positive 후보들에 대한 상관관계 실측 (Tier 3) -- 1d는 중복 억제 필터 정상 작동(corr=0.82) 입증, 12h는 독립 시그널 간 씻아웃 오류(corr=-0.006) 입증.
- **Impact:** missed quantile 버그 해결로 8h 등에서 6개 이상 추가 시그널 구제 가능해졌으며, 12h의 무고한 탈락 문제를 해결하기 위한 피어 수 임계치 도입 등 향후 조치 방향성이 데이터로 증명됨.

## [2026-07-16] [TASK_SPEC_CONTRACT_JSON] [ADR_20260716_SPEC_CONTRACT_JSON]
- **Context/Why:** check phase had no way to verify spec implementation completeness; all verification was manual
- **Resolution/What:** spec SKILL.md: contract.json 생성 지침 추가. lean_check.py: --spec + _check_spec_compliance. check SKILL.md: --spec usage. sync_task.py: contract.json도 cleanup
- **Impact:** Spec-to-implementation gap can now be auto-detected in check phase

## [2026-07-16] [TASK_SKILL_REFACTOR_V2] [ADR_20260716_SKILL_REFACTOR_V2]
- **Context/Why:** 첫 실행에서 2965개 삭제 — .git/__pycache__ .venv/*.pyc 포함, 실제 temp 파일(.tmp .bak)만 대상으로 제한 필요
- **Resolution/What:** _wipe_temp_artifacts: EXCLUDED_DIRS 추가, .pyc 제거 (__pycache__가 이미 excluded)
- **Impact:** 정확한 temp wipe, 생태계 손상 방지

## [2026-07-16] [TASK_SKILL_REFACTOR_V2] [ADR_20260716_SKILL_REFACTOR_V2]
- **Context/Why:** 1차 개편 후 audit에서 circuit breaker 누락, temp artifact wipe 미구현, clean state verify 미명시 발견
- **Resolution/What:** check SKILL.md: circuit breaker 복원. sync_task.py: _wipe_temp_artifacts() 추가. sync SKILL.md: git status verify 명시
- **Impact:** 모든 원래 요구사항 충족, gap zero

## [2026-07-16] [TASK_CHECK_SYNC_REFACTOR] [ADR_20260716_CHECK_SYNC_REFACTOR]
- **Context/Why:** SKILL.md contained redundant rules, scripts had 3 separate subprocess calls, pytest ran twice, no AI-first diagnostic output
- **Resolution/What:** SKILL.md 축소 44→11/42→15줄, lean_check.py pytest 2→1회+JSON stderr, sync_task.py 통합, AGENTS.md SSOT 강화
- **Impact:** AI 판단 cycle 50%+ 감소, task당 토큰 ~200줄 절감, old scripts 3개 제거

## [2026-07-16] [TASK_L1_REGISTRY_ADMISSION_RECALIBRATION] [ADR_20260716_L1_REGISTRY_ADMISSION_RECALIBRATION]
- **Context/Why:** 직전 ADR(L1_SLOW_TF_GATE_RECALIBRATION)이 pooled 심볼 다양성만 고쳐서 6h~1d가 여전히 BLOCKED로 남았음. 재실측 결과 두 원인 확인: (1) fold-level 원시 심볼 수 하한(l1_min_cross_section=2, TF 무관 고정)이 LUNA2USDT/JASMYUSDT 단일심볼 fold를 pooled LCB 계산에서 배제. (2) registry_empty 가설(L0 데이터 부재)을 실측으로 반증 -- 실제로는 fold당 144~2500개 후보가 만들어지는데 pair-level FDR(compute_symbol_strategy_evidence의 Benjamini-Yekutieli 조화급수 보정)이 hard_eligible 후보의 98.9~99.8%(8h/12h/1d)를 탈락시키고 있었음(2h는 76%).
- **Resolution/What:** calibrate_l1_symbol_breadth_gate.py에 measure_fold_min_ready_symbols_by_tf()/propose_cross_section_thresholds() 추가(측정=p10 raw ready_symbols, registry_empty fold 제외). config.py에 l1_min_cross_section 오버라이드(8h=2/12h=1/1d=1) 및 l1_pair_fdr_procedure: Literal['by','bh']='by' 필드 신설, 8h/12h/1d만 'bh' 채택(6h는 실측 negative gross edge로 의도적 제외). signal_selection.py의 _by_q_values에 harmonic_override 파라미터 추가(1.0=plain BH, None=기존 BY 그대로 -- 기본값 불변).
- **Impact:** 실측 control replay 4-run 재실행(수치 완전 동일, ablation_restores_control=true): 12h는 3/4 fold ready로 완전 신규 PASS(probe_lcb_bps -inf -> +122.5bps, 후보 10개 실제 승급). 1d는 게이트 지표 대폭 개선(probe_lcb_bps -inf -> +143.3bps, 구조 3/3 통과)했으나 advisory fold_ratio 경고 및 개별 후보 승급 0건이라는 새 하류 병목 노출. 8h는 오히려 probe_lcb_bps -35.1 -> -147.9bps로 악화 -- FDR 완화로 그동안 우연히 함께 걸러지던 진짜 마이너스 분기(fold#1, -129.5bps, 8심볼)가 evidence pool에 들어온 결과로, 게이트 버그가 아닌 진짜 경제적 악재 노출로 잠정 판단. 6h는 의도적 미조정으로 변화 없음(예상대로). 1h/2h/4h 회귀 없음(수치 완전 동일). 236/236 테스트 통과, 신규 코드 94~100% 커버리지. 후속 과제: 1d의 게이트-통과/승급-0건 괴리 원인 규명, registry_empty 잔존분 L0 원인 규명.

## [2026-07-16] [TASK_L1_SLOW_TF_GATE_RECALIBRATION] [ADR_20260716_L1_SLOW_TF_GATE_RECALIBRATION]
- **Context/Why:** result.md가 지목한 6h~1d BLOCKED의 3대 가설 중 대상2(비용 이중차감)은 코드 감사로 반증(dynamic_funding/execution_cost_bps 필드가 생산자 없이 항상 빈 튜플). 실제 원인은 pooled Symbol-Breadth 게이트(l1_min_effective_sym_n)가 1h/2h만 오버라이드되고 4h~1d는 방치되어 플랫 기본값(3.0)을 적용받은 것과, 부트스트랩 LCB가 블록 수와 무관하게 quantile 0.05 고정이라 소표본 fold를 과도하게 벌점화한 것 2가지였음.
- **Resolution/What:** src/domain/futures/strategy/calibrate_l1_symbol_breadth_gate.py 신규(측정=p10 effective_sym_n per TF, 채택=config.py 수동 반영, calibrate_l1_pair_gate.py와 동일 거버넌스). config.py에 8h=2.0/12h=1.0/1d=1.0 l1_min_effective_sym_n 오버라이드 및 l1_lcb_quantile_* 4개 필드 추가. evidence_policy.py에 block-count 적응형 _resolve_lcb_quantile 신설(num_blocks>=15는 no-op, 이하일수록 0.05->0.20 선형 완화), metrics.py에 resolve_num_blocks 추출(DRY), signal_selection.py에 quantile 파라미터 배선.
- **Impact:** 실측 control replay 재실행(PYTHONPATH=. uv run python -m src.domain.futures.strategy.run_l1_cross_tf_diagnosis, 4-run 전부 동일/ablation_restores_control=true): 1h/2h/4h 회귀 없음(수치 완전 동일). 6h/8h/12h/1d Symbol-Breadth 구조적 게이트는 목표대로 전부 PASS로 전환되었으나, 그럼에도 전체 판정은 여전히 BLOCKED — fold-level insufficient_ready_symbols(l1_min_cross_section=2, TF 무관 고정, 이번 범위 밖)가 LUNA2USDT(+229bps)/JASMYUSDT(+98.5bps) 단일심볼 fold를 pooled LCB evidence pool에서 별도로 배제하고 있음이 실측으로 새로 확인됨. registry_empty(L0 상류, 4개 분기 중 3개 후보 0건)도 별개 병목으로 노출. 6h는 게이트와 무관하게 실측 gross edge가 마이너스(-54/-33bps)로 진짜 무엣지 가능성. 후속 스펙 필요: fold-level 심볼 게이트 TF-스케일링 확장, registry_empty L0 원인 규명.

## [2026-07-16] [TASK_L0_L1_TIMEFRAME_SCALED_DAILY_DENSITY_GATE] [ADR_20260716_L0_L1_TIMEFRAME_SCALED_DAILY_DENSITY_GATE]
- **Context/Why:** 느린 시간프레임의 L0 Cheap Gate 차단 및 4h 강결합을 해소하기 위해, 달력일수와 하루평균 최소 빈도에 기반한 동적 밀도 스케일러를 도입함.
- **Resolution/What:** contracts.py와 cheap_gate.py에서 oos_window_days를 datetimes로부터 자동 추출하고 daily_event_density와 daily_effective_n_density를 기반으로 임계값을 비례 스케일링하며 archetype/family 최소 하한값을 max 필터링으로 융합 적용함.
- **Impact:** 실측 Replay 통과. 4h 유효 신호 12➔34개로 대폭 증가, 2h 74➔77개로 증가. 6h의 비정상 -inf 및 sym_count 블로커가 정상적인 경제성 LCB (-40.4bps) 필터링으로 정상화됨.

## [2026-07-15] [TASK_L0_L1_DYNAMIC_COST_CAUSAL_FEEDBACK] [ADR_20260715_L0_L1_DYNAMIC_COST_CAUSAL_FEEDBACK]
- **Context/Why:** 느린 TF들의 L1 블로킹을 해소하기 위한 펀딩/슬리피지 동적 비용 모델과, look-ahead free causal feedback loop의 부재를 해결하기 위함.
- **Resolution/What:** Layer1FoldReadiness 필드를 확장하여 dynamic_funding_cost_bps, dynamic_execution_cost_bps 등을 추가하고 signal_selection.py에서 이를 연동하여 fold LCB 경제성을 동적으로 계산하도록 구현함.
- **Impact:** 실측 4-run Sequential Replay 완주 확인. Peak RSS 9.4GB로 안정화. 1h 및 2h PASS, 4h WARNING, 6h~1d는 registry_empty로 BLOCKED 상태 유지. 아블레이션 후 L1 최종 결과가 control과 100% 동일하게 수렴되어 인과성 검증 완료.

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
- **Resolution/What:** src/domain/futures/strategy/calibrate_l1_pair_gate.py 신규 작성 — 측정(control replay + effective_n_sink 훅)과 채택(config.py 수동 반영)을 분리해 quant.md anti-overfitting 원칙 준수. signal_selection.py의 compute_symbol_strategy_evidence에 선택적 effective_n_sink 파라미터 추가(기본 None, 기존 동작 불변). 최초 구현에 크래시 2건 발견 및 수정: (1) run_once(trace={})가 STAGE_ORDER 미시딩으로 크래시 — trace 사전 시딩으로 수정. (2) pipeline.py가 compute_symbol_strategy_evidence를 자체 이름으로 재import하므로 signal_selection 모듈 패치는 무효였음(측정치 전부 빈 값) — pipeline 모듈 자신의 바인딩을 패치하도록 수정. 두 회귀 모두 재현 테스트로 고정. 실제 control replay 재실행으로 6-TF 전체 effective_n p10 실측 후 config.py의 4h/6h/8h/12h/1d l1_pair_min_effective_obs를 전부 4.0(2h 기존값=ceiling)으로 갱신.
- **Impact:** 실측: 6개 TF 전부 effective_n p10(4.9~28.4)이 ceiling(4.0)을 초과해 전부 4.0으로 수렴 — 어떤 TF도 2h보다 엄격한 문턱값을 요구받지 않게 되어 역방향 스케일링 버그 제거. lean_check 전 파일 PASS(calibrate_l1_pair_gate.py 격리 커버리지 96%). L0 TF-probe 별도 조사에서 _TF_PROBE_FALLBACK_SYMBOLS(BTC/ETH/BNB)가 실제로는 BNBUSDT 단일 심볼로 조용히 축소되는 별개 이슈 발견(BTC/ETH가 유니버스 선정 정책상 거래 유니버스에서 원천 제외됨) — 이번 task 범위 밖으로 별도 기록, 후속 조사 필요.

## [2026-07-15] [L0_L1_DIAGNOSTIC_PIPELINE_INTEGRITY] [ADR_20260715_L0_L1_DIAGNOSTIC_PIPELINE_INTEGRITY]
- **Context/Why:** run_l1_cross_tf_replay.py가 RunnerResult를 폐기(exit 항상 0)하고 canonical 10-stage 중 terminal_event_audit/outer_folds 2개를 캡처하지 않았음. cross_tf_diagnostics.py의 diagnose_snapshots/write_cross_tf_diagnosis 정식 계약은 유닛테스트 fixture 외 어디서도 호출되지 않는 고아 코드였고, run_tiered_pipeline_outcome의 diagnostic_sink 파라미터는 정의만 되고 미사용, RunnerResult는 models.py/active_pipeline.py에 이중 정의되어 매 호출마다 상호 변환되고 있었음.
- **Resolution/What:** run_once()가 caller-owned trace dict를 참조로 받아 RunnerResult와 STAGE_ORDER 10개 전체(신규: outer_folds, terminal_event_audit)를 기록하도록 수정, main()이 RunnerResult.exit_code를 프로세스 exit code로 반영하고 예외 시에도 partial trace를 보존하도록 변경. cross_tf_diagnostics._STAGE_ORDER를 공개 STAGE_ORDER로 승격하고 snapshot_from_raw_stage_entry() 어댑터를 신설. src/domain/futures/strategy/run_l1_cross_tf_diagnosis.py 신규 작성 — control/control_repeat/treatment/fusion_ablation 4-run을 순차 supervisor로 실행하고 diagnose_snapshots()/write_cross_tf_diagnosis()에 실제로 연결. run_tiered_pipeline_outcome의 미사용 diagnostic_sink 파라미터 제거. RunnerResult 이중정의를 models.py 단일 클래스로 통합(active_pipeline.py는 이를 import), runner/pipeline.py의 불필요한 재포장 제거. L0 probe(probe_manifest)는 L2 master TF 선정에만 쓰이고 L1 admission을 게이트하지 않는 기존 설계를 그대로 유지(범위 밖으로 명시).
- **Impact:** control replay 재실행으로 실측 검증: artifact에 runner_result + 10/10 stage 전부 기록됨(이전엔 8/10 + RunnerResult 없음), 2h n_valid=74/fold edge 수치가 result.md와 완전 일치(회귀 없음). 4h/6h/8h/12h/1d는 여전히 registry_empty로 BLOCKED — 이 fold 판정 로직 자체는 이번 범위 밖(과거 조사에서 진짜 시장 비정상성으로 판정됨). L0 probe 0 winning cells vs 2h L1 독립 PASS가 동일 실행에서 재확인되어 L0/L1 분리 설계를 실측 뒷받침. lean_check 전 파일 PASS(신규 진단 파일 커버리지 90~92%).
