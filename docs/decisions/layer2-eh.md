---
title: Layer 2 AWF Engineering History (Compressed)
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## Phase 1: 평가체계 구축 (6/15)
- CAGR objective+L2 Optuna 연동, L2 AWF fold 동기화(l2_start~holdout_start), verbose callback(\r 진행률)
- 8조건 절대+상대 AND 게이트(CAGR>0, Sharpe≥0.5, MAR≥1, MDD≤20%, fold≥60%, Uplift+0.20)
- fold pass_ratio zip 버그 수정(빈 fold ValueError→전체 정렬+분모 분리)
- AWF 정합 P0+P1: 복리 CAGR, taker 비용 차감(first bar only), net edge 핸드오프, AWF window look-ahead 제거

## Phase 2: 게이트 재설계+DSR 중심 (6/15~16)
- PSR≥0.90+Friction≥0.50 게이트 활성화, EW-of-all→Top-K-EW baseline 교체
- DSR 수식 교정(연율/bar 단위 통일, Bailey&Prado 2012 정밀식)
- DSR-corrected champion selection+replay 검증 도입, study 영속 로드+override_dsr 브릿지
- Study 오염 수정: load_if_exists→delete_study+재생성, 영구 champion 레저(별도 study, run간 갱신)
- Edge-conditional throttle(conviction multiplier clip((s-floor)/(ref-floor),0,1)^γ)
- Growth-Gate 재설계: LCB z=1.0→0.0, max_ann_vol 0.50→1.20, DSR 하드게이트 제거→진단
- 5측면 재편: MDD 20→30%, CVaR 3→6%, Sortino≥1.5 gate, trade≥30, 상대MDD 제거
- Adaptive breadth(K_RANK causal 확장) + shaped objective(risk_utilization/trade_count bounded bonus)
- Deployment 재배선: deploy_cost_safety_mult 분리, edge_throttle_min_active_mult, risk_budget_floor_ratio

## Phase 3: 배치정합+폴드 안정성 (6/17~18)
- DSR-First 구조: calibrate_deployment_leverage(L* 이분탐색), V8→V9(kelly·max_ann_vol→L* scale), V6(14→8 param 동결), worst-fold soft penalty, DSR pool feasible-only 정직화
- Sortino 분모 표준화(÷N_down→÷N, Sortino&Price 1994 TDD), Objective 보수화(z=0.5, risk_util=0.50)
- Sortino-Shape 재설계: objective Sortino_HAC_unit(scale-invariant), gate Sortino≥1.5+Sharpe≥0.7+Calmar≥0.5, vol_target=1.0 강제, fit-leg OOS 대리→fit_rets_hybrid 우선, DSR→PSR/Sortino/Calmar floor
- CAGR 배치 갭(C1~C4): fit-leg book 수익률(equal→chain per-bar), L2 trial 내 L* calibration, vol/gross 노브(kelly 배율 제거→vol×L*/gross×L*)
- 벡터화: L2SimulationCache 6종 2D 행렬, _run_awf_simulation 객체 생성 100% 제거(np.where 1D), 200 trials 1:25→1:06(+25%)
- L2→L3 deployment parity(l2_deploy_leverage 명시 전달, L3 deploy 경로 재계산)
- Recent-fold collapse 진단: Layer2FoldDiagnostics, fold별 deployed CAGR/MDD/selected symbols, Optuna constraint 9번째, calibrate_deployment_leverage cvar_margin+exchange_cap
- 선택 심볼 추적(fold_selected_symbols) + universe audit 4종 경고(LayerUniverseAudit)

## Phase 4: 후반 무결성 (6/19~21)
- Provenance fingerprint: ValidatedSignalBatch streaming SHA-256→study identity, 회귀 테스트 BY permutation/singleton/empty
- Purge WFA 활성화: L2 fold도 config purge/embargo(max_holding_bars×purge_safety_mult) 적용, fold 경계 label overlap 차단
- Scale collapse 이중수정: _book_edge_score double-deduct 제거(eff_hurdle 재차감→mu_bps는 net), project_all_caps allow_vol_upscale(Cap5 양방향 정규화)
- Parallel ProcessPoolExecutor+deterministic batching(batch-synchronous ask/tell, seed 재현성)
- Realization 정합: Cap5 하향전용→L* 직접 스케일, exchange_cap=10× 기본, deploy_leverage SSOT 전달
- 최종 promotion hardening: 최근 non-empty fold pass gate, exchange cap default 10.0 복원
