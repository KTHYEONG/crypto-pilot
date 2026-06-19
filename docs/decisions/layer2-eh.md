---
title: Layer 2 AWF Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## 2026-06-18 L2 final promotion hardening — recent fold gate + exchange cap default restore
- **Delta:** final replay promotion now requires the latest non-empty deployed fold to pass `recent_fold` when enabled. Fold pass ratio is computed from deployed fold CAGR instead of unit-path compounding. `Layer2AllocationConfig.from_mapping` now restores the absent-key default `l2_max_exchange_leverage=10.0` and keeps explicit `None` as a deliberate cap disable.
- **Rationale:** L2 already exposed a last-fold collapse before L3. Promotion on stale aggregate fold ratio hid the deterioration, and the missing exchange cap default allowed oversizing beyond the documented contract.
- **Edge Cases:** recent fold gate is optional through config, but enabled by default. The recent fold Sharpe floor is replay-only and does not affect Optuna feasibility.

## 2026-06-18 L2 realization 정합 복구 — C4 천장주입 제거 + 수익률 직접 스케일 + exchange_cap
- **Delta:** `project_all_caps` Cap5 하향전용(`min(vol_scale,1.0)`) + per_symbol=0.15 ceiling 선행 binding 확인 → `max_ann_vol/gross *= L*`가 구조적 no-op 판정. `selection._apply_deployment_to_params`에서 천장주입 제거, 추적(l2_deploy_leverage)만 보존. `run_l2_awf(deploy_leverage=L*)` 파라미터 추가 → `apply_deployment(rets, L*)` 단일 경로로 최종 스코어카드 CAGR/MDD/CVaR 재산출. `calibrate_deployment_leverage`에 `exchange_leverage_cap`(기본 10×) 추가 — binding argmin 통일. `l2_deploy_cvar_margin` 노브 공식화. `pipeline.py` 호출부에서 `l2_params["l2_deploy_leverage"]` SSOT 전달.
- **Rationale:** 결함 #3: trial-path(apply_deployment → CAGR 133.6%) vs 최종스코어카드(C4 천장주입 무효 → CAGR 5.5%) 이중 경로 불일치. 정합 복구로 `BLOCKED(cagr)` 허상 해소. exchange_cap=10×: L*=19.5 → notional 58× notional은 Binance perp 한도 초과 → 실행불가 차단.
- **Edge Cases:** RiskUtil 정합 가드(binding=mdd 시 `|risk_util-(1-margin)|>0.15` warning) — 결함 #1/#2 재발 자동 감지. champion L* SSOT 미전달 시 내부 recalibrate fallback(결정론적, 동일값 보장).

## 2026-06-18 L2 CAGR 배치 갭 수정 — C1(fit-leg book) + C2(trial 내 L*) + C4(vol/gross 노브)
- **Delta:** (C1) `awf_sim.py` fit-leg 루프를 equal-weight 시장수익률 평균 → OOS와 동일한 체인(rank→kelly→throttle→tradeable→cost→funding) per-bar book 수익률로 교체. (C2) `evaluate_l2_trial`에서 L* = `calibrate_deployment_leverage(fit_rets_hybrid)` 호출 → `cagr/mdd/cvar`를 deployed 값으로 갱신; `sortino/sharpe/psr`는 unit-vol 유지. `Layer2TrialEvaluation`에 `deploy_leverage/deploy_binding` 필드 추가. (C4) `selection._apply_deployment_to_params`에서 kelly 배율 제거, `max_ann_vol=vol_base×L*`·`gross_exposure_cap=base_gross×L*`로 실제 배치 노브 전환. `l_hard_cap` 기본값 4.0→20.0 유지.
- **Rationale:** RC-2: fit-leg가 시장 MDD≈25%를 참조해 L*≈1 고정(전략 unit-vol MDD≈1%). RC-3: kelly 배율이 `project_all_caps(target_ann_vol)` 재정규화로 상쇄(dead code) — vol_target·gross_cap만이 실제 scale 노브. RC-4(throttle): m_t≈1.0 확인(무죄). 이론 CAGR(L*=20 적용 시)≈91%, 실측 3.3%는 순수 미배치 문제.
- **Edge Cases:** fit-leg loop에서 `risk_budget_floor` 미적용(경미, 위험 완충). fit_rets_hybrid 빈 경우 OOS proxy fallback 보존. opt_main_futures.py tqdm 변경은 scope creep(선언된 의존성, 기능 무관).

## 2026-06-18 L2 Sortino-Shape 재설계 — D1(목적함수) + D2(vol복원) + D3(fit-leg) + D4(DSR→PSR)
- **Delta:** (D1) Optuna 목적 `growth_lcb` → `Sortino_HAC_unit`(scale-invariant). 게이트 3단: Sortino≥1.5(1차) + Sharpe≥0.7(sanity floor, 기존 1.0 이완) + Calmar≥0.5(복리앵커). `metrics.py::_sortino_hac_unit` 신설(HAC downside dev). (D2) `vol_target=None→1.0` 강제 — RC-1 cascade 3종(vol-targeting·risk_budget_floor·adaptive_breadth) 재활성. (D3) AWF fit-leg 수익률 수집(`_AwfSimResult.fit_rets_hybrid`) → `_apply_deployment_to_params`에서 OOS 대리 대신 fit-leg 우선 소비. `l2_deploy_l_hard_cap` 4.0→20.0(MDD/CVaR가 실제 binding). (D4) `dsr_floor` BLOCKER 제거 → `psr_floor`/`sortino_floor`/`calmar_floor` 신설. DSR = diagnostic 잔존(계산·로깅 유지). `L2_ALLOC_SPACE_V9` = V7 구조 복원.
- **Rationale:** RC-1: V8가 `max_ann_vol` 제거 → `vol_target=None` silent cascade. RC-2: scale-dep `growth_lcb`를 고정 kelly=0.25에서 최적화 → objective 표면 평탄 → shape 노이즈 선택 → Sharpe 0.839. RC-3: DSR pool = 동일 신호셋 파라미터 섭동(독립 가설 불성립) → 자기참조 → DSR≈0.5 고정 → V8 절단을 강제. Crypto 비대칭 수익 → Sortino(하방전용)가 Sharpe보다 shape 측정 타당. Sharpe floor(≥0.7)는 하방 표본 희소 시 Sortino 인플레이션 방어.
- **Edge Cases:** `l_hard_cap=20.0` + fit-leg 저변동 → L*가 MDD binding에서 결정. fit_rets_hybrid 빈 경우 OOS fallback 유지. Calmar 게이트 불가도달 구간 없음(BLOCKER 순서: calmar_floor < mdd_abs, 수치 조합으로 테스트 확인).

## 2026-06-17 L2 DSR-First 구조 개혁 — Fix A(결정론적 배치) + Fix B(목적함수 정렬) + Fix C(DSR pool 정직화)
- **Delta:** (A) `risk_deployment.py` 신설: `calibrate_deployment_leverage(fit_rets, mdd_cap, l_hard_cap=4.0)` 이분탐색 → champion kelly/vol 사후 스케일링(L*≤4). (B) `compute_v3_score`에 `disable_risk_penalty=True` 추가 — L2 objective에서 MDD/CVaR 소프트 패널티 제거. champion 선택을 `argmax(dsr, cagr)`로 변경(이전: 첫 gate-pass break). (C) `L2_ALLOC_SPACE_V8`(V7 − kelly − max_ann_vol, 9-param). DSR pool = feasible-only → benchmark 하락 → DSR 0.270→0.502 관측.
- **Rationale:** 진단: CW1=목적함수↔게이트 불일치(MDD/CVaR penalty가 under-deployment 보상), CW2=DSR pool 오염(infeasible trials의 Sharpe가 벤치마크 인플레이션), CW3=kelly/vol이 보수적 목적함수에 공동 종속. DSR/Sharpe는 L 스케일에 불변 → CAGR과 DSR 문제 분리 가능 → Fix A로 CAGR, Fix B+C로 DSR 독립 공략.
- **Edge Cases:** `binding=hard_cap`이면 OOS-fit 분포 이격 무관(L* 결정에 데이터 미활용). fit-leg 수익률 미노출 → OOS 대리 사용; `mdd_margin=0.30`으로 완충. 향후 AWF `fold_fit_rets` 노출 시 per-fold calibration 전환.

## 2026-06-17 L2 replay gate contract 정렬 — raw Sharpe / EW baseline Sharpe 보존
- **Delta:** `Layer2TrialEvaluation`에 `sharpe_hybrid`와 `sharpe_hac_baseline_ew`를 추가해 replay 시 final gate를 원본 pipeline 입력과 동일한 계약으로 재계산하도록 정렬. `select_layer2_champion`은 DSR를 주입하기 전후 모두 같은 promotion gate helper를 사용하고, replay candidate의 `promotion_blocker`를 canonical blocker로 유지한다.
- **Rationale:** 기존 replay 경로는 `sharpe_hac_hybrid`를 raw Sharpe 대용으로 쓰고 baseline도 EW가 아닌 HAC baseline으로 대체해, pipeline과 champion selection의 gate 의미가 어긋날 수 있었다. DSR는 replay-time diagnostic으로만 추가하고, gate 정의 자체는 pipeline과 동일해야 L2/L3 불협화음을 줄일 수 있다.
- **Edge Cases/Trade-offs:** legacy trial attrs는 `l2_constraint_values` fallback으로 유지한다. replay 입력이 없는 구 trial은 기존 HAC 기반 근사값으로만 복원되므로, 새 contract가 존재하는 trial보다 blocker 정밀도가 낮을 수 있다.

## 2026-06-17 L2 최적화 무결성 강화 — 선정차단·DSR게이트·OOS보강·V6·worst-fold
- **Delta:** (1) `opt_main_futures.py`: `blocker_reason!=""||best_evaluation is None` → `layer2_blocked` 즉시 반환(infeasible 챔피언 L3 승격 차단). (2) `pipeline.py`: `dsr_floor`(≥0.60) 게이트 13번째 조건 추가. (3) `config.py`: `ml_fit_fraction` 0.60→0.55, `ml_calibration_fraction` 0.20→0.15 (OOS 20%→30%). (4) `opt_config.py`: `L2_ALLOC_SPACE_V6`(8-param, V5 14-param 보조항 6개 동결), `L2_OPTUNA_TRIALS` 120→200. (5) `workflow.py`: worst-fold Sharpe soft-penalty(`max(0,-0.30-worst)×0.005`) objective 차감.
- **Rationale:** L3 홀드아웃 −7% 원인은 `feasible=0→infeasible fallback 무단 승격` + `14-param×OOS 20%` 구조적 과적합(E_max=+3.09σ, DSR=0.000). CAGR 문제 아님. OOS 보강+탐색공간 축소로 과적합 토양 제거, 무결성 차단으로 false-positive 소멸.
- **Edge Cases/Trade-offs:** DSR 임계 0.75→0.60 완화(구조 개선 전 달성가능 보수적 하한). 게이트 임계(Sortino 1.5 등) 동결. IS 5% 단축 수용. worst-fold는 soft(feasible pool 보존). 동결 파라미터의 고정값은 `from_mapping` default로 하위호환.

## 2026-06-17 L2 Sortino 분모 표준화 + Objective 보수화
- **Delta:** (1) `metrics.py::_sortino` 분모를 `÷N_down(비표준)` → `÷N(전표본, Sortino&Price 1994 TDD)`으로 교정. (2) `l2_growth_lcb_z` 기본값 0.0→0.5 (fold 일관성 보상). (3) `l2_objective_risk_util_target` 기본값 0.35→0.50 (deployment headroom 확장).
- **Rationale:** 비표준 분모가 실효 게이트 요구치를 ~2.1로 상향시켜 0/120 feasible trials 유발. Sortino≈Sharpe(1.310≈1.297)은 분모가 full-σ로 수렴한 측정오류의 지문. 게이트 임계 1.5는 동결(이중조정 금지). z=0 point-estimate가 Fold#4 Sharpe=-1.041 붕괴 허용. RiskUtil=46.5%>target=35%로 CAGR deployment bonus 조기 포화.
- **Edge Cases:** 게이트 1.5 동결 — 분모 교정만으로 게이트 통과 여부 자연 결정. `l2_objective_risk_util_target` 상향 시 MDD/CVaR 게이트 재실행으로 동시 충족 확인 필요.

## 2026-06-16 L2 AWF fold anchoring 복원 (PART4 data-merge regression fix)
- **Delta:** `run_tiered_pipeline`에서 `build_walk_forward_folds(n_bars=...)` 호출 시 `n_bars`(전체 aligned 길이, holdout 포함)가 아니라 `ho_start_idx_l2`(holdout_start까지 bar 수)를 전달하도록 anchoring 복원.
- **Rationale:** 같은 날 적용된 IS+OOS 데이터 병합(L3 빈 holdout 수정)으로 `aligned.datetimes`가 holdout_end까지 늘어나면서, `n_bars`에 비례하던 L2 fold 생성 공식의 global OOS 구간이 holdout_start 너머로 밀려나 fold 3개→1개로 붕괴(Optuna feasible trial 0건, DSR=0.000 fallback, Trades 58→29 BLOCKED).
- **Edge Cases:** L1(`build_l1_nested_swf_folds`)은 명시적 `l1_start_idx`/`l1_end_idx`로 anchored되어 이 영향을 받지 않음(수정 전후 byte-identical 출력으로 확인). 수정 후 fold 4개/Trades 99로 회복, 정당한 Sortino 게이트 미달로 BLOCKED — 데이터 버그 아님.

## 2026-06-16 Layer 2 active deployment 재배선 — 비용 허들/스로틀/위험예산 분리, V4 탐색공간 공개
- **Delta:** (1) `fixed_cost_safety_mult`의 역할을 gross-edge 보수화로 유지하고, deployment 단계에는 `deploy_cost_safety_mult`를 분리 배선. (2) `edge_throttle_min_active_mult`로 positive edge book의 최소 활성 비중을 허용. (3) `risk_budget_floor_ratio` + `risk_budget_max_scale`로 목표 vol 대비 under-deployed book을 상향 조정하되 support는 보존. (4) `L2_ALLOC_SPACE_V4` 공개 및 `search_space_version="v4"`로 study hash를 분리. (5) `Layer2AllocationConfig.from_mapping`에 신규 필드와 범위 검증 추가.
- **Rationale:** 결과가 CAGR 14.8%, MDD 1.5%, RiskUtil 5.0%로 나타난 것은 알파 부족보다 under-deployment에 가깝다. L1 신호를 더 크게 쓰되, 비용 모델과 look-ahead 방어는 유지해야 하므로 배치 강도와 비용 허들을 분리해야 했다. 기존 게이트를 완화하는 대신 자본 배분 경로를 적극화해 복리 성장 목적에 정렬했다.
- **Edge Cases:** 새 파라미터가 없는 기존 trial params는 `from_mapping` default로 호환. `risk_budget_floor_ratio<=0` 또는 `vol_target is None`이면 floor 적용을 건너뛴다. support 밖 신규 non-zero 생성은 금지.

## 2026-06-16 Layer 2 adaptive breadth/objective shaping — trade 수와 risk utilization을 Optuna 신호로 승격
- **Delta:** (1) `adaptive_breadth_enabled`/`adaptive_k_extra`/`adaptive_expand_below_vol_ratio`로 저활용 book에서 `K_RANK`를 causal하게 확장. (2) `growth_lcb`를 유지하면서 `risk_utilization`과 `trade_count`에 작은 bounded bonus를 주는 shaped objective 추가. (3) `L2_ALLOC_SPACE_V5` 및 `search_space_version="v5"`로 study hash 분리. (4) `select_layer2_champion`은 replay mismatch 시 즉시 block하지 않고, 상위 후보를 bounded fallback으로 재검증해 replay-feasible champion을 canonical result로 선택.
- **Rationale:** 최신 L2는 PASS이지만 `Trades=58`이 정체되고 `RiskUtil=12.0%`로 MDD budget을 과소 사용하고 있었다. 단순 sizing 확대만으로는 support churn이 늘지 않으므로 breadth 자체를 조건부로 넓히고, Optuna가 deployment intensity를 직접 학습하도록 objective를 재구성해야 했다. replay canonicalization은 결과 신뢰성을 먼저 확보하기 위한 안전장치다.
- **Edge Cases:** adaptive breadth는 `vol_target`이 없거나 `expand_below_vol_ratio<=0`이면 비활성. objective bonus는 primary growth_lcb를 역전하지 않도록 bounded. replay fallback이 전부 infeasible이면 `non_deterministic_replay`를 유지.

## 2026-06-16 L2 평가체계 5측면 재편 — Sortino/CVaR완화/MDD30%/Trade수 게이트
- **Delta:** (1) `l2_max_mdd_abs` 0.20→0.30, `l2_max_cvar_95` 0.03→0.06(핵심 unlock). (2) 신규 `l2_min_sortino_abs=1.5`(efficiency), `l2_min_trades=30`(sample floor) 게이트+표시 추가. (3) 상대MDD(`mdd_rel`) 게이트 완전 제거→진단(`RelMDD` display). (4) 게이트 체인 9→12조건: Stage0(sanity+`low_trades`)→A(cagr)→B(sharpe/sortino/mar)→C(mdd_abs/cvar_95)→D(fold/active_blocks/friction)→E(growth_lcb/uplift). (5) `evaluate_l2_trial` 제약 10→12-tuple(sortino+trade_count), `layer2_constraints_from_trial` 패딩 동기화. (6) `Layer2Result`에 `sortino_hybrid/terminal_multiple/total_pnl_pct/trade_count/risk_utilization` 표시필드 추가. (7) scorecard를 Growth/Efficiency/Risk/Robust/Uplift/Diag 5그룹으로 재편.
- **Rationale:** CAGR 17.6% BLOCKED인데 Sharpe 2.747/MAR 6.297/MDD 2.8% 우수 — 알파부족이 아니라 위험예산(MDD20%) 14%만 사용하는 under-deployment. binding 제약 1순위 = per-bar CVaR 캡(3%). 사용자 결정: 복리자산증식 극대화가 목적(자동매매·심리개입 없음)이므로 MDD를 깊게 제한하지 않되 통계신뢰성(Sortino/Calmar=scale-invariant, 레버리지로 curve-fit 불가)으로 신뢰축을 분리. under-deployment는 하드 floor 대신 Soft(진단표시+캡완화로 objective가 자연유도).
- **Edge Cases:** `_sortino` dd<1e-12(무손실)→0.0 방어. `_terminal_multiple` 전손(∏≤0)→0.0. trade_count는 `prev_support→new_support` 신규진입만 카운트(리밸런스 유지는 미포함). DSR/PSR/상대MDD 모두 진단 전용 유지(과거 결정 존중). `L2_ALLOC_SPACE` 탐색공간 불변(anti-curve-fit, 게이트 임계값만 조정).

## 2026-06-16 Growth-Gate 재설계(C1-C4) — LCB z=0, 배치천장 개방, 게이트 재배선, l1-tmp 폐기
- **Delta:** (1) `l2_growth_lcb_z` 1.0→0.0(분산 패널티 제거), `l2_min_cagr` 0.15→0.30. (2) `L2_ALLOC_SPACE_V3.max_ann_vol` high 0.50→1.20(MDD≤20% 생존 제약으로 탐색 위임). (3) DSR 하드게이트(`l2_min_dsr`) 완전 제거 → 진단 전용; relative-MDD를 hard `<=baseline`에서 material floor(0.05) 면제 + 25% tolerance band로 교체. (4) `select_layer2_champion`이 DSR 컷오프 루프 없이 objective(growth_lcb) 최상위 feasible trial을 직접 챔피언으로 선정. (5) `docs/specs/l1-tmp.md`(L1 비정상성 별도 트랙) 폐기.
- **Rationale:** CAGR 12.2%/Sharpe 2.433/fold 100% pass인데도 구조적으로 under-deployed — 원인은 (a) LCB z=1.0의 과도한 분산 패널티가 objective를 왜곡, (b) `max_ann_vol<=0.50` 배치천장이 growth 최대화를 인위적으로 제한, (c) DSR이 trial-pool 상대 benchmark라 trial 수와 동반 상승하는 구조적 편향 + L3 frozen holdout과 중복 검증.
- **Edge Cases:** `l2_min_dsr` 필드는 코드에 잔존(과거 호환/진단용)하나 게이트 어디서도 read 안 됨. `mdd_h<=0.05`는 노이즈로 간주해 상대 게이트 전부 면제.

## 2026-06-16 Optuna Study 오염 수정 — 매 실행 초기화 + 영구 챔피언 레저
- **Delta:** (1) L2 study 생성을 `_optuna.create_study(load_if_exists=True)`→`get_or_create_study(resume=False)`로 전환(매 실행 `delete_study`후 재생성). (2) 별도 `l2_champion_store_{tf}` study 신설(절대 미초기화, `optimize`/`ask` 미호출 — 순수 기록용); `load_champion_params`/`update_champion_store`(`run_tracker.py`)로 run간 챔피언이 개선될 때만 갱신. (3) 다음 run의 warm-start anchor가 레저 best params를 우선 사용(현재 `L2_ALLOC_SPACE` 키만 필터).
- **Rationale:** `study_name` 해시가 search-space 내용을 반영하지 않아(`search_space_version` 리터럴) 코드 변경(bounds 조정 등) 후에도 동일 study에 이종 trial 누적 → TPESampler가 동일 파라미터의 distribution 불일치를 "dynamic search space"로 오판해 RandomSampler fallback 경고 발생 + `study.optimize(n_trials=120)`가 매 실행 기존 trial에 120개씩 누적(Trial#361 관측). 오염 study 2건 Redis에서 즉시 삭제.
- **Edge Cases:** 챔피언 레저는 `optimize` 대상이 아니므로 search space 변경에도 dynamic-search-space 경고 무관. 레저에 과거 dead-param이 있어도 anchor 병합 시 현재 space 키만 채택해 무시.

## 2026-06-16 Edge-Conditional Throttle — 시변 conviction multiplier 도입
- **Delta:** `awf_sim._run_awf_simulation` 사이징 직후 `w *= m_t`. `m_t = clip((s-floor)/(ref-floor),0,1)**γ`. `s = Σ|w_i|·max(|μ_i|−h̃_i,0)/Σ|w_i|`, `h̃_i=h_i·safety_mult/rebal_bars`. `Layer2AllocationConfig` 4필드(enabled/floor/ref/gamma) 추가. Optuna 탐색 미추가(anti-curve-fit).
- **Rationale:** Sharpe/DSR은 척도-불변(`Sharpe(c·r)=Sharpe(r)`) → kelly_fraction·max_ann_vol 튜닝은 DSR 구조적 무이동. 유일한 DSR 레버 = 수익 분포 재형성(시변 승수). Fold#1 Sharpe−2.08 구간(저-edge)을 down-weight → aggregate Sharpe↑·좌측 tail↓ → DSR 2경로 상승.
- **Edge Cases:** baseline 미적용(게이트 보수성). `safety_mult` 포함(`eff_hurdle=hurdle*safety_mult/rebal_bars`)으로 내부 friction filter와 산식 일관. NaN score→m=0. disabled→m≡1(기존 동작 비트-동일).

## 2026-06-16 DSR-Optuna 정렬 — 11번째 제약 + V3 Range + Warm-start
- **Delta:** (1) `L2_ALLOC_SPACE_V3` 신설: `kelly[0.15,0.55]·vol[0.20,0.50]` 축소(V2: 0.10–1.0/0.20–0.80) → `std(pool_SR)↓` 직접 반영. (2) `objective_l2_growth`에 running-DSR 11번째 Optuna 제약 추가(`_DSR_LOOP_MIN_POOL=8` startup 면제). (3) `L2_OPTUNA_TRIALS` 50→120, warm-start anchor trial, `search_space_version="v3"`.
- **Rationale:** DSR 0.228 차단은 신호 한계 아님 — benchmark(`E[max pool SR]≈2.46`) > champion(1.69). 원인: (a) DSR이 탐색 루프 밖 → TPE가 DSR 방향 학습 불가. (b) 넓은 range가 pool SR 분산 과부풀. 11번째 제약으로 TPE feasibility 모델이 저-DSR 영역 회피, range 축소로 benchmark 직접 하강.
- **Edge Cases:** 구버전 10-element `l2_constraint_values`는 11번째 `1.0` 자동 패딩(backward compat). Bailey-LdP 원형(전 complete trial 풀) 유지. DSR 11번째 제약은 selection.py `all(c<=0)`에 자동 포함(tuple 길이 무관).

## 2026-06-15 L2 BLOCKED 교정 — Objective↔Gate 정렬, Dual Baseline, DSR 재보정
- **Delta:** (1) `build_directional_equal_weight_baseline`(순수 1/N) 신설 — uplift 게이트/표시 전용. risk-matched는 MDD 상대 게이트 전용 유지. (2) Optuna 제약 10번째에 uplift 제약 추가 → 탐색이 게이트로 수렴. (3) `l2_min_active_blocks 4→3` (AWF fold 기반 정의 통일), `l2_min_dsr 0.95→0.75`, PSR 게이트 제거(진단 강등). (4) DSR fallback 단일원소 degenerate 경로 → PSR 정직 하한. (5) `n_startup_trials` 40%→20%.
- **Rationale:** objective(growth_lcb)와 차단 기준(DSR·uplift)이 분리된 구조적 오설계 + risk-matched baseline self-defeat(diagonal-Kelly와 수학적 동일 → uplift≡0) + active_blocks 정의 불일치(fold 수 3 < 임계 4). DSR·PSR 이중 차감도 동시 제거.
- **Edge Cases:** `rets_baseline_ew` 빈 list 방어(`_sharpe_hac` → 0.0 반환 유지), `mdd_baseline`은 risk-matched 유지(MDD 상대 게이트 불변).

## 2026-06-15 Layer 2 DSR 수식 교정 및 Scorecard 가독성 개선
- **Delta:** `metrics.py`의 `_deflated_sharpe_probability` 함수 내에서 연율화 Sharpe와 per-bar 표준오차 간의 단위 혼용을 per-bar 스케일로 통일하고 Bailey & Prado (2012) 표준오차 정밀 수식으로 수정. `tiered_logging.py`에서 성과표의 각 지표 옆에 심플한 통과 기호(예: `>=15.0%`, `<=baseline`)를 출력하도록 수정.
- **Rationale:** 단위 불일치로 DSR이 항상 0.000으로 차단되던 수학적 오류를 정정하고, 사용자가 각 지표의 패스 기준 장벽을 로그에서 직관적으로 파악할 수 있도록 하기 위함임.

## 2026-06-15 Layer 2 DSR 챔피언 선정 및 Replay 검증 도입
- **Delta:** [selection.py](file:///src/domain/futures/strategy/tiered_workflow/selection.py)를 신설하여 DSR-corrected champion selection(`select_layer2_champion`) 로직을 구축함. `opt_main_futures.py`에서 `_layer2_experiment_key`로 Study를 영속 로드하고 `override_dsr` 브릿지로 최종 pipeline에 DSR을 동기화함. 시뮬레이션 재실행 결과 지표와 stored 지표 간 오차를 비교하는 deterministic replay 검증을 추가함.
- **Rationale:** 최적화 구간의 최고 CAGR을 편향되게 선택하는 것을 배제하고, 여러 trial 간의 Sharpe 분산 및 유효 검정 횟수를 반영하여 보정된 기대 복리성장 성과 지표(DSR)를 챔피언 승격 기준으로 삼기 위함임.

## 2026-06-15 L2 AWF 평가 폴드 동기화 및 로그 중복 제거
- **Delta:** `objective_l2_sharpe` 내 AWF 평가 폴드를 L2 윈도우 `[l2_start, holdout_start)` 범위로 통일. 스터디 설정/생성 로그 억제, `FutureWarning` 경고 차단, 튜닝 완료 후의 중복 챔피언 파라미터 출력 구문을 차단했습니다.
- **Rationale:** 최적화와 최종 파이프라인 간의 폴드 불일치로 인한 수학적 수치 오차를 정정하고, 최종 단계의 `[AWF PORTFOLIO PERFORMANCE SCORECARD]` 출력이 유일한 챔피언 매개변수 결과 노출지가 되도록 정합성을 통일했습니다.

## 2026-06-15 L2 Optuna 로그 억제 및 CAGR 목적함수 최적화
- **Delta:** `run_tiered_pipeline`, `run_l2_awf` 등에 `verbose: bool` 옵션 배선 및 Trial 구동 시 `verbose=False`로 스코어카드 출력 차단. `\r`을 사용한 1줄 진행률 표시 callback 연동. `objective_l2_sharpe`를 L2 리스크 게이트 검증을 거친 CAGR(복리연수익률) 극대화 지표로 개편 및 미통과 시 `-inf` 패널티 반환 적용.
- **Rationale:** 수십 회의 최적화 Trial 동안 거대한 포맷 테이블들이 콘솔을 지저분하게 채우던 환경을 단 한 줄의 캐리지 리턴 진행률 모니터링으로 개선하고, 자산 복리증식 목적을 직접 최적화하여 튜닝 결과의 아키텍처적 정합성을 강화함.

## 2026-06-15 L2 Optuna 최적화 및 Tiered 파이프라인 연동
- **Delta:** `opt_main_futures.py`에 Step A→B→C→D 흐름 구축(`_build_l2_signal_batch`, `_run_tiered_l2_study` 추가). `run_tiered_pipeline`에 `l1_result_override` 파라미터 추가하여 L1 재실행 방지. `L2_OPTUNA_TRIALS` 설정 추가.
- **Rationale:** 기존에는 L2 최적화 파라미터가 파이프라인에 주입되지 않고 빈 딕셔너리로 고정되어 최적화 혜택을 받지 못함. L1 중복 피팅을 제거하고 L2에 특화된 Optuna Sharpe 최적화 및 holdout 검증까지의 엔드투엔드 파이프라인 완결성 확보.
- **Edge Cases:** MagicMock 객체와의 호환성을 위해 `_to_utc_timestamp` 헬퍼 함수를 통해 Timestamp 변환 오류를 방지하고, L1 import 바인딩 충돌을 config 네임스페이스 접근으로 우회.

## 2026-06-15 L2 게이트 재보정 — 보수적 임계값 + PSR/Friction 게이트 + EW Bench baseline
- **Delta:** 임계값 강화(CAGR 0→15%, Sharpe 0.5→1.0, MAR 0.5→1.0, MDD 50→20%). PSR≥0.90·Friction≥0.50 신규 게이트 배선(dead config 활성화). Baseline: valid-전체-EW→Top-K-EW. 음수MAR `n/a(loss)` 표기 가드.
- **Rationale:** 기존 PASS는 느슨한 절대 게이트 + strawman baseline(EW-of-all=-81% CAGR)으로 상대 게이트 자동통과. PSR/Friction config가 opt_config.py에 선언만 되고 L2 로직 미배선 상태. 재보정 결과 BLOCKED(friction=34.5%<50%) — 신호 2/3 비용 허들 미달 노출.
- **Edge Cases:** PSR 분모 off-by-one(fisher=True κ→(κ+2)/4, 기존 (κ+1)/4) audit에서 발견·수정. MAR CAGR<0 무의미(단조성 파괴) → 표기만 가드, 수식 불변.

## 2026-06-15 Layer2 이벤트 계약 분리 및 support-preserving projection
- **Delta:** `ValidatedSignalBatch` 기반 event schedule을 L2 입력 SSOT로 고정하고, `rank_and_select`는 `signed`/`absolute` 모드로 분리했다.
- **Rationale:** L1의 방향·holding·net edge를 bar-level 포지션 계약으로 보존해야 short symmetry와 기존 legacy rank를 동시에 유지할 수 있다.
- **Edge Cases:** `project_all_caps`는 support 밖 신규 non-zero를 만들지 않으며, malformed mock mask/funding 입력은 fail-open으로 처리한다.

## 2026-06-15 L2 AWF 정합성 강화 (P0+P1)
- **Delta:** 5개 결함 수정 — 복리 CAGR, 로그 필드명, taker 비용 차감, net edge 핸드오프, AWF 윈도우 look-ahead 제거 + 4중 게이트 도입
- **Rationale:** audit 점수 43/100 → 구조 정합성 확보. 복리 목적(사용자 요구)과 비용 현실성(quant.md §4) 직접 충돌 제거
- **Edge Cases:**
  - `fold_sharpes_h` 전체 fold 기준 계산 후 별도 `_nonempty_sharpes`로 pass_ratio 분모 산출 (zip 길이 불일치 방지)
  - taker 비용은 리밸런싱 첫 bar에만 차감 (`t2 == t` 조건); baseline(1/N)은 무비용 유지 (게이트 보수성)

## 2026-06-15 L2 게이트 재설계 (8조건 절대+상대 이중기준)
- **Delta:** 곱셈식 게이트(×1.20, 절대Sharpe0.30, fold>50%) → 8조건 AND: Stage0(sanity)+A(CAGR/MAR/Sharpe절대)+B(MDD상대+절대)+C(복리fold≥60%)+D(가산Uplift+0.20). `Layer2Result`에 cagr/mar/fold_pass_ratio/blocker_reason 필드 추가.
- **Rationale:** 음수 baseline에서 ×1.20이 임계를 역전(로그 `>=-1.02` 버그). CAGR/MAR 게이트 부재로 절대손실 전략 통과 가능. 복리자산증식 목적과 직접 충돌.
- **Edge Cases:**
  - Stage A CAGR>0: `<=` 비교 (0.0 정확히는 FAIL — 원금 유지만으론 불충분).
  - fold pass = `prod(1+r)>1.0` (변동성드래그 반영); 빈 fold nonempty 분모 분리.
  - config 6키 `l2_params.get(key,default)` 노출 — magic number 금지.

## 2026-06-15 AWF fold pass_ratio zip 버그 수정
- **Delta:** `fold_sharpes_h = [_sharpe(fr) for fr in sim.fold_rets_hybrid if fr]` → `if fr` 필터 제거, `zip(strict=True)` 길이 불일치 런타임 오류 수정
- **Rationale:** 빈 fold 존재 시 `ValueError` 발생. 전체 정렬 유지 + 분모 별도 분리로 수정

## 2026-06-18 L2 Vectorization & Logic Recovery — Object Instantiation 0% + 1:1 Parity
- **Delta:** (1) `L2SimulationCache`에서 Python `global_schedule` 삭제 → `expected_gross_bps_2d`, `signal_mask_2d` 등 2D NumPy 행렬 6종으로 신호 전체 사전 번역. (2) `_run_awf_simulation` 루프 내부의 `SymbolSignal`, `Layer2ExpectedEdge` dataclass 생성을 100% 제거하고 `np.where` 마스킹을 통한 1D 배열 직접 연산으로 교체. (3) `build_l2_simulation_cache`에서 `_event_strength` 기반의 엄격한 tie-breaking 복구 및 missing hurdle(NaN)에 대해 `3.8 bps` 기본값 fallback 복원.
- **Rationale:** 파이썬 객체 생성 오버헤드가 L2 Optuna 속도의 90% 이상을 점유함을 확인하여 엔진을 완전 벡터화로 교체(200 Trials 01:25→01:06, ~25% 단축 및 병목 제거). 초기 벡터화 시 `3.8 bps` 누락(NaN 처리)으로 인해 발생한 `no_feasible_trials` 차단을 해결하여 기존 엔진과 수학적 1:1 정합성을 확보함.
- **Edge Cases:** `event_strength_2d` 캐싱으로 동일 종목-동일 시점의 시그널 충돌 시 결정론적 선택 보장. Aligned 데이터의 `execution_cost_bps_2d`가 `None`인 경우에도 안전하게 fallback 작동 확인.

## 2026-06-18 L2→L3 deployment parity 정합화
- **Delta:** `run_tiered_pipeline`가 L2 champion의 `l2_deploy_leverage`를 L3 holdout에도 그대로 전달하고, `run_l3_holdout(deploy_leverage=L*)`가 L2 final scorecard와 동일한 `apply_deployment(rets, L*)` 경로로 deployed hybrid 지표를 재계산하도록 정렬. `Layer3Result`에 `deploy_leverage` 추적 필드 추가.
- **Rationale:** L2 champion이 배치까지 통과했더라도 L3가 unit path로 평가되면, holdout 결과가 L2 scorecard와 계약상 불일치하여 false-negative/false-positive 판정이 발생한다. 동일 배치 계약을 L3에 명시적으로 주입해 백테스트 정합성을 복구한다.
- **Edge Cases:** `deploy_leverage<=1.0` 또는 비유한값은 무효화되어 기존 unit path를 유지한다. baseline 수익률은 배치하지 않는다.

## 2026-06-18 L2 recent-fold collapse 진단 + 게이트 경화 + 유니버스 감사
- **Delta:** (1) `Layer2FoldDiagnostics` dataclass + `compute_layer2_fold_diagnostics()` 신설 — fold별 deployed CAGR/MDD, unit Sharpe, compound pass, selected symbols 산출. (2) `Layer2TrialEvaluation`에 `recent_fold_passed/sharpe/cagr/mdd`, `latest_to_median_cagr`, `fold_deployed_cagrs`, `fold_selected_symbols` 필드 추가 → Optuna attr 기록 및 `evaluate_layer2_gate` 입력. (3) `evaluate_layer2_gate` optuna_constraint_values 8→9 (recent_fold constraint index 5). (4) `calibrate_deployment_leverage`에 `cvar_margin`, `exchange_leverage_cap` 인자 추가 — `evaluate_l2_trial`에서 배치 제약 정합. (5) `select_layer2_champion` replay gate에 `recent_fold_passed/sharpe` 전달 — 최근 fold 미통과 후보 차단. (6) `LayerUniverseAudit` dataclass + `build_layer_universe_audit()` 신설 — active/entry_block/kill mask 기반 4종 경고. (7) `run_tiered_pipeline` L1/L2/L3 각 window audit 로깅. (8) `format_layer1_outer_fold_table`에 calendar date + symbol preview 추가. (9) `_AwfSimResult.fold_selected_symbols` 추가 — fold별 선택 심볼 추적. (10) `Layer2AllocationConfig.from_mapping` 기본값 조정: `l2_min_sharpe_abs=0.7`, `l2_min_sortino` passthrough (legacy `l2_min_sortino_abs` fallback), `l2_min_calmar=0.5`.
- **Rationale:** L1·L2 최신 fold 약화 원인을 결과에 끼워 맞추지 않고, 유니버스 활용 계약·데이터 정렬 감사·fold 안정성 평가·진단 로그를 분리해 L3 진입 전 인플레이션 후보를 검증해야 했다. Optuna constraint에 recent-fold 제약을 포함해야 TPE가 최근 fold 품질을 탐색에 반영할 수 있다. fold별 selected symbols 로깅은 디버깅 성능 진단의 핵심.
- **Edge Cases:** empty fold → `recent_fold_passed=None`, `recent_fold_constraint=-1.0` (non-blocking). `l2_require_recent_fold_pass=False` 시 제약 비활성. universe audit은 진단 전용으로 portfolio weight 미변경. 피팅 시 L3 holdout leak 없음.

## 2026-06-19 L2 provenance fingerprint hardening — batch identity now keys study identity
- **Delta:** `_signal_batch_fingerprint()`가 `ValidatedSignalBatch`의 start/end bounds, symbols, registry/model versions, event count, 그리고 모든 event field를 streaming SHA-256으로 해시하게 됐다. `_layer2_experiment_key()`는 event count만 보지 않고 이 fingerprint를 study identity에 포함하도록 바뀌었다. `_run_tiered_l2_study()`는 event count, unique symbols, fingerprint prefix를 provenance 로그로 남긴다. 관련 회귀 테스트는 real `ValidatedSignalBatch` fixture와 BY permutation/singleton/empty 케이스로 갱신됐다.
- **Rationale:** event-count-only study names는 같은 크기의 서로 다른 L2 입력을 충돌시킬 수 있었다. 재생성 가능성과 replay 추적성을 batch 내용 수준으로 끌어올려야 결과 SSOT와 실행 계약이 일치한다.
- **Edge Cases:** event tuple 순서를 그대로 유지해 downstream input 순서 차이를 구분한다. float는 `float.hex()` 직렬화를 사용해 `-0.0`과 작은 finite 차이를 숨기지 않는다. study name suffix는 기존처럼 12 hex prefix를 유지한다.
