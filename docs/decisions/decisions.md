# Active Decisions Log (Sliding Window)

## [2026-07-17] [TASK_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX] [ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX]
- **Context/Why:** assess_crisis_reliability가 LUNA/FTX 위기 윈도우를 로드할 때 BTCUSDT 등 timestamp_x/timestamp_y 병합-접미사 스키마 심볼(전체 ~4%)이 load_single_symbol_data의 3단 폴백에서 전부 실패해 침묵 탈락(has_btc=False 실측 확인). market_regime._btc_index()는 BTC 부재 시 예외 없이 return 0(임의 심볼 대체)해, 이미 프로덕션에 활성화된 regime-conditional 익스포저 캡(apply_regime_risk_cap, bull=1.0/bear=0.35/crisis=0.25)이 엉뚱한 심볼로 레짐을 오판정하고 있었다.
- **Resolution/What:** opt_data_utils.py에 _resolve_timestamp_column 헬퍼를 추가해 timestamp 부재 시 timestamp_x로 폴백하도록 load_single_symbol_data 두 분기를 수정. market_regime._btc_index()는 BTC 부재 시 ValueError로 fail-closed 전환. active_pipeline.py의 [CRISIS-RELIABILITY] 로그에 윈도우별 raw MDD/CAGR/CVaR detail을 추가. 사전 존재하던 테스트 버그(FUTURES_DATA_DIR를 opt_data_utils 모듈에 잘못 monkeypatch, 실제 소유자는 src.core.settings)도 함께 수정해 xfail 7건을 실통과로 전환.
- **Impact:** 실측(scratch/probe_crisis_regime*.py): BTC 데이터 정상 복구 확인(overlap_symbols 37→47, valid_symbols 35→45, has_btc False→True, events 50532→67593). 전체 L2 파이프라인 재실행 결과 위기 MDD/CAGR은 오히려 악화(29.01%→55.47%, -32.73%→-38.44%) — timestamp_x 버그가 위기 replay 경로뿐 아니라 정상 L1/L2 유니버스 로딩에도 걸쳐 있어 이번 수정으로 정상장 데이터 풀이 바뀌었고(registry_symbols 93→103), Optuna가 더 공격적인 챔피언을 선택(정상장 CAGR +61.2%→+92.8%, Uplift +0.10→+0.29)한 결과로 판단됨. 위기 게이트는 여전히 stress_tested_fail로 정상 차단 중 — 데이터 무결성 수정 자체는 검증됨, production 승격은 계속 차단.

## [2026-07-17] [TASK_L2_CRISIS_SURVIVAL_POLICY] [ADR_20260717_L2_CRISIS_SURVIVAL_POLICY]
- **Context/Why:** 정상장 L2 스코어카드는 PASS했지만 독립 LUNA/FTX replay에서 champion이 MDD 29.01%와 CAGR -32.73%를 기록했고, 기존 위기 판정은 첫 window의 MDD만 확인해 production 승격을 막지 못함.
- **Resolution/What:** CrisisWindowMetrics와 순수 evaluate_crisis_survival 정책을 도입하고, 모든 configured crisis window에 대해 데이터 충분성·MDD·CAGR·CVaR·거래 수를 함께 판정하도록 assess_crisis_reliability를 집계형으로 변경했다. 현재 L2 threshold에서 하나라도 실패하거나 데이터가 부족하면 apply_crisis_reliability_override가 gate를 monotonic fail-closed로 차단하며, 위기 결과는 Optuna selection에 재투입하지 않는다.
- **Impact:** 동일 4h/2026-07-17/seed42 replay에서 L1 7TF 및 정상 L2 수치(CAGR +61.2%, Sharpe 2.026, MDD 19.9%)는 유지됐고, crisis replay는 stress_tested_fail/verified=False로 승격을 차단했다. 35 symbols·903 bars·50,532 events를 평가했으며 기존 false PASS 경로를 제거했다. Spec compliance와 전체 check PASS(Cov 35%).

## [2026-07-17] [TASK_L2_TF_INCLUSION_GATE_NATIVE_TF_FIX] [ADR_20260717_L2_TF_INCLUSION_GATE_NATIVE_TF_FIX]
- **Context/Why:** 위기 재현성 게이트 replay(2026-07-16)에서 mdd=0.0000 cagr=+0.0000이 정확히 0으로 산출. 코드 조사 결과 crisis-stress 전용 버그가 아니라, 커밋 c2831990(L1→L2 네이티브 TF 핸드오프)이 strategy_id 포맷을 TF 접미사 포함(donchian_72_8h)에서 family:variant(TF 별도 native_tf 필드)로 바꾼 뒤, C4 게이트의 OOS 필터(awf_sim.py:3205,3208)가 옛 정규식 파서 _parse_tf_from_strategy_id로 여전히 strategy_id를 파싱해 항상 unk를 반환 → included_tfs_by_fold와 매칭 실패 → 매 OOS bar 전체 sleeve 탈락. l2_tf_inclusion_enabled(기본값 True) 전 경로에 적용되는 전역 회귀였음.
- **Resolution/What:** docs/specs/l2-tf-inclusion-gate-native-tf-fix.md 구현: cache.sleeve_to_tf(SSOT)로 (symbol, strategy_id)->native_tf 룩업(_build_sleeve_tf_lookup) 구축, OOS 필터가 파싱 대신 직접 조회하도록 변경. 죽은 파서 함수(_parse_tf_from_strategy_id)와 스텁 테스트(S4) 정리, 실제 필터 경로를 검증하는 통합 테스트로 교체.
- **Impact:** A/B 실측(git stash 전/후 동일 champion registry로 assess_crisis_reliability 직접 실행): 수정 전 mdd=0.0000/cagr=+0.0000 재현, 수정 후 mdd=0.0635/cagr=-0.0093로 정상 산출 확인. 전체 L2 phase 재실행(2026-07-17)에서도 실제 champion registry(93심볼)로 mdd=0.2901 cagr=-0.3273 정상 산출(더 이상 0 아님). L1 수치 회귀 없음, /check PASS(spec compliance + ruff/mypy/pytest, Cov 45%). 산출된 CAGR -32.7%가 위기 방어 관점에서 허용 가능한지는 별도 정책 판단 필요(result.md 다음 조치 기록).

## [2026-07-17] [TASK_L2_GATE_SCORECARD_AND_CRISIS_RELIABILITY] [ADR_20260717_L2_GATE_SCORECARD_AND_CRISIS_RELIABILITY]
- **Context/Why:** L2 스코어카드(format_layer2_table)의 개별 ✅/❌ 판정이 실제 Layer2AllocationConfig 값을 읽지 않고 하드코딩된 리터럴(uplift 0.20 등)로 재계산되어 실제 게이트(0.05)와 표시가 자기모순됐고, DSR/PSR 표시 역할이 실제 게이팅(PSR이 하드 블로커, DSR은 diagnostic)과 반대로 표시됐다. 또한 evaluation_window_bottleneck_verdict()가 계산하는 NO-CRISIS-WINDOW 여부가 텍스트 경고로만 출력되고 promotion_passed에 전혀 반영되지 않아, 위기 미검증 상태로도 PASS가 나갈 수 있었다(reversal kill-switch가 실제 위기 replay에서 손실을 악화시킨 것으로 반증된 사고가 사후 ad-hoc replay로만 발견됐던 사례가 이 공백의 실제 위험을 보여줌).
- **Resolution/What:** format_layer2_table에 config 파라미터를 추가해 모든 임계값을 config.l2_min_*/l2_max_*에서 읽도록 SSOT화하고 PSR/DSR 표시 역할을 실제 게이팅과 일치시켰다. L1/L2/L3가 전혀 보지 않은 out-of-band 역사적 붕괴장(2022 LUNA+FTX 붕괴, 2022-04-01~2023-02-15, 챔피언 registry와 실제 겹치는 legacy 심볼만 사용)에 대해 champion의 이미 확정된 전략 정체성(rule-based family/variant)을 재학습 없이 그대로 적용해 생존 테스트하는 assess_crisis_reliability()/apply_crisis_reliability_override()를 신설하고 active_pipeline.py의 기존 사후 override 패턴(master_tf mismatch와 동일)에 배선했다. evaluate_layer2_gate/Optuna 200-trial 탐색 루프는 전혀 건드리지 않았다.
- **Impact:** 실측 replay(동일 cutoff/seed)로 스코어카드 표시 수정 확인(Uplift +0.10이 실제 임계값 0.05 기준 올바르게 ✅로 표시, PSR/DSR 역할 정정). 위기 재현성 게이트는 배선 검증 과정에서 3개의 독립된 pre-existing/신규 버그를 추가 발견·수정: (1) 위기 구간이 275일로 1d 최소 300-bar 요건 미달 → 320일로 확장, (2) 8h(챔피언 master TF)는 정적 enriched 파일이 없는 파생 TF라 항상 4h로 로드 후 리샘플링하도록 수정, (3) align_data_maps()의 cache_result=False 시 UnboundLocalError(cache_key 미정의) pre-existing 버그 수정. 최종 replay에서 배선 전체가 에러 없이 완주(35개 겹치는 심볼로 stress_tested_pass 산출)했으나 mdd=0.0000/cagr=+0.0000이 정확히 0으로 나와 시뮬레이션이 실제 포지션을 잡았는지 의심스러움 — 후속 조사 필요, 이번 promotion 판단에는 미사용.

## [2026-07-16] [TASK_L1_L2_MASTER_TF_HANDOFF_WIRING] [ADR_20260716_L1_L2_MASTER_TF_HANDOFF_WIRING]
- **Context/Why:** L1 7개 TF 전부 PASS했음에도 L2가 TieredPipelineError(no deployable timeframe found for L2 master TF)로 fail-closed됨. 원인 2가지: (1) _resolve_l2_master_tf(cfg, {})가 empty per_tf_l1 dict로 호출되는 3개 프로덕션 콜사이트(pipeline.py 2곳, active_pipeline.py 1곳)가 override 유무와 무관하게 항상 실패, (2) 자동선택 분기가 assess_l1_tf_handoff의 master/auxiliary readiness(breadth+family diversity+finite positive edge)를 배선하지 않고 legacy _is_deployable_per_tf_result만 사용.
- **Resolution/What:** Layer1Result.selected_timeframe(기존 미사용 필드)를 run_tiered_pipeline의 실제 per-TF 계산 분기에서 채우고, 신규 _resolve_l2_master_tf_from_prior 헬퍼가 empty-dict 재계산 대신 이 값을 재사용하도록 3개 콜사이트를 교체. _resolve_l2_master_tf 자동선택 분기는 assess_l1_tf_handoff(min_ready_symbols=cfg.l2_master_min_ready_symbols, min_source_families=cfg.l2_master_min_source_families, 기존 config 필드 재사용) 기반 master_eligible 필터로 교체하고 rejection reason을 [ALGO] trace로 기록.
- **Impact:** 동일 cutoff/seed(2026-07-16, seed=42) replay 재실행 결과 fail-closed 재발 없이 L2 최종 시뮬레이션까지 완주 확인(master_tf=8h 자동선정, CAGR +61.2%/Sharpe 2.026/MDD 19.9%, Uplift 게이트 1개만 미달). L1 7개 TF 수치는 수정 전과 완전 동일(회귀 없음). lean_check.py 전 구간 PASS(spec-compliance 포함, Cov 17%). NO-CRISIS-WINDOW 캐비아트로 이번 replay 수치의 production 승격 근거 사용은 금지.

## [2026-07-16] [L1_L2_NATIVE_TF_SPEC_CLEANUP] [ADR_20260716_L1_L2_NATIVE_TF_SPEC_CLEANUP]
- **Context/Why:** The native-TF handoff implementation, regression checks, ADR, and replay report are complete; the working spec must not remain active.
- **Resolution/What:** Removed the completed implementation blueprint and contract JSON from docs/specs.
- **Impact:** docs/specs contains no stale active blueprint; permanent decisions and current replay status remain documented.

## [2026-07-16] [L2_NATIVE_TF_HANDOFF] [ADR_20260716_L2_NATIVE_TF_HANDOFF]
- **Context/Why:** L2 required native timeframe artifacts but its runtime policy disabled L0, causing fail-closed missing event maps.
- **Resolution/What:** Run L0 gate for multi-layer phases and normalize removed CLI defaults before the L1-to-L2 handoff.
- **Impact:** Native artifacts now reach L1; current replay advances to master selection, which remains separately fail-closed.

## [2026-07-16] [TASK_L0_SLOW_TF_XS_CHALLENGER] [ADR_20260716_L0_SLOW_TF_XS_CHALLENGER]
- **Context/Why:** 6h/1d는 구조 게이트와 pooled LCB가 양수인데도 개별 pair quality_weight_zero로 0건 승급이었고, 기존 XS residual family는 이 TF pool에 없었다.
- **Resolution/What:** slow_tf_xs_challenger_enabled opt-in 아래 6h/1d pool에 residual_momentum_xs와 xs_residual_rebalance를 중복 없이 추가하고 해당 TF effective config에만 XS factor-level admission을 활성화했다.
- **Impact:** 동일 historical replay 2회에서 6h는 BLOCKED 0에서 PASS 8, 1d는 BLOCKED 0에서 PASS 1로 전환했고 비목표 TF 최종 L1 결과는 불변이다. full upstream trace 비결정성과 독립 holdout 부재로 production promotion은 보류한다.

## [2026-07-16] [L1_FDR_HARD_ELIGIBLE_SCOPING] [ADR_20260716_L1_FDR_HARD_ELIGIBLE_SCOPING]
- **Context/Why:** FDR 디커플링 수정 이후에도 6h/1d가 구조 게이트 clean+probe_lcb_bps 양수임에도 0건 승급. 추적 결과 compute_symbol_strategy_evidence의 q-value 계산이 이미 구조적으로 탈락 확정된(hard_eligible=False) 후보까지 다중검정 보정 분모 m에 포함시켜, 실제 경쟁하지도 않는 후보 수백 개가 진짜 후보 몇 개의 q-value를 인위적으로 부풀리고 있었음을 확인(decisions.md 기존 실측: 8h/12h/1d 후보 풀 950~1515개 중 2~3개 패밀리가 대부분).
- **Resolution/What:** FDR q-value 계산을 hard_eligible 부분집합에만 적용하도록 제한(raw_p_values를 hard_eligible_idx로 필터링 후 _by_q_values 호출, non-hard-eligible은 q_value=1.0 sentinel). _compute_probe_m_eff의 groups도 동일 부분집합으로 제한. 단조적으로 안전한 수정(m 축소는 q-value를 개선만 시킴)이라 스냅샷/deployment 양쪽 호출부 모두에 call-site 분리 없이 동일 적용.
- **Impact:** 실측 treatment replay(1h 포함 전체 7TF) 결과: 모든 TF가 동일하거나 증가(1h=100 동일, 2h 86->87, 4h 32->33, 8h 25->44 큰 개선, 12h 18->21), 감소 0건으로 단조성 검증 완료. 6h/1d는 여전히 0건 승급이나 원인이 명확해짐 -- FDR 재계산 후에도 quality_weight_zero의 대부분이 probability_positive<=0.5(진짜 약한 부트스트랩 증거)로 확인되어, 이 두 TF는 게이트 결함이 아닌 순수 데이터 검정력 부족으로 최종 판단(quant.md anti-overfitting 원칙에 따라 추가 완화 보류). lean_check PASS, spec contract 4개 시나리오 전부 구현.

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
