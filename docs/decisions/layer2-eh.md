---
title: Layer 2 AWF Engineering History (Compressed)
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## [2026-06-23] L2 Optuna Memory Optimization and WSL2 OOM Safe Fallback
- **Delta:** Lowered default `L2_OPTUNA_BATCH_SIZE` to 2. Implemented dynamic sequential fallback (`n_jobs=1`) if system available memory drops below 3.0 GB. Added explicit garbage collection (`gc.collect()`) prior to and after heavy stages.
- **Rationale:** High-memory fork executions in 16GB WSL2 host environments caused memory exhaustion and process eviction (OOM Killer). Lowering concurrency and falling back to sequential execution when under memory pressure ensures absolute execution integrity.

## [2026-06-23] Multi-TF Precision-Weighted Signal Pooling
- **Delta:** L1 per-bar net edge (symbol×TF) → pooled symbol-level via inverse-variance: $\mu_s = \sum c_i \mu_i / \sum c_i$ (not summation). Conviction cap $c_s = \min(\sum c_i, 1.5 \max c_i)$.
- **Rationale:** v1 mu 합산(+4× inflation) → RiskUtil 144.8%, MDD 43.4%, Friction 12.6%. v2 precision평균 → bounded convex comb, no inflation. RiskUtil→80.1%, MDD→24.0%, Friction 0.0%(재정의 필요).
- **Edge Cases:** Direction conflict (+/−μ)→auto-netting; single-TF k=1→항등(회귀 유지); tied qw→equal-weight pooling.

## [2026-06-23] Friction Gate Dimension Fix (Per-Bar Gross vs Cost)
- **Delta:** Friction 판정: per-bar $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ (기존: per-bar net vs round-trip cost, 차원불일치+이중차감).
- **Rationale:** v1 기존 버그: net(이미 cost 차감)을 round-trip cost(H미상)과 비교→H≈72× 과소→12.6% 통과. v2 정규화→0.0%. fix: `compute_expected_layer2_edge` per-bar (gross, cost)를 precision-pooled 후 동일 차원 비교.
- **Trade-offs:** 교정 후 friction ~100% 무력화 가능→l2_min_friction_pass 임계 재조정 필요(별도 과제).

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

## Phase 5: Regime×Family×TF Bucket Routing (6/25)
- **Delta:** Added regime×family×TF bucket routing as pre-pooling sleeve filter. 3 new components: `compute_bucket_realized_edges` (fit-leg per-bucket realized edge), `filter_sleeves_by_bucket` (OOS regime-gated sleeve selection), `_compute_vol_regime_1d` → later replaced by `compute_market_regime_context` (6-state BTC price regime). Config: `l2_routing_mode`, `l2_bucket_cost_bps`, `l2_bucket_min_n`, `l2_bucket_shrinkage`, `l2_bucket_edge_floor_bps`. Default mode changed from `"pool"` to `"bucket"`. TF-gate log downgraded to DEBUG.
- **Rationale:** 기존 고정 평균 풀링은 regime×family×TF에 따른 이질적 신호 품질을 무시. bucket routing은 conditional edge 추론으로 regime-conditional 상관 +0.14~+0.33 (8/8 positive, 7/8 p<0.05) 실측 기반. min_n + shrinkage가 과적합 방어.
- **Edge Cases:** Look-ahead 방지 (fit_end=oos_start). 미관측 bucket = 0 → 자동 제외. Close=0 분모 max(|c[t]|, 1e-12) 방어. Regime 경계 초과 시 0 fallback. 하위호환 `l2_routing_mode="pool"` 유지.
- **Audit Fixes:** Off-by-one loop bound (`fit_end-1`→`fit_end`) + `t+1>=t_max` guard. `l2_routing_mode` 타입 Literal 제약. `compute_market_regime_context` 연동 (기존 vol-quantile 대체).

## [2026-06-24] L2 Attribution Diagnostics — Per-Fold Edge Decomposition
- **Delta:** Added `Layer2FoldAttribution` dataclass + `_assemble_fold_attribution` pure function + `_count_netting_symbols` helper. Extended `_resolve_sleeve_signals_at_bar` return to 3-tuple `(sigs, edges, n_dropped)`. Config: `l2_diag_attribution_enabled` (bool), `l2_diag_sleeve_top_k` (int), `l2_diag_sleeve_sample_every` (int). Within `_run_awf_simulation`: fold-local accumulators for realized price/funding/cost, expected net (final w), throttle multiplier, gross/net exposure, friction pass, below-cost drops, netting events. Per-fold `[L2-ATTR]` DEBUG log. Optional sleeve-level `[L2-ATTR-SLEEVE]` top-K log.
- **Rationale:** L1→L2 CAGR collapse (`+60bps → -3.6%`) could not be decomposed into alpha decay / sizing collapse / cost drag / funding by existing logs (gate result only). Attribution provides quantitative separation: `realized_total = realized_price + realized_funding − realized_cost`, `alpha_gap = realized_total − expected_net`. Validates whether alpha genuinely decayed (expected_net > 0 & realized_total < 0 → code innocent) or pooling/throttle/cap erased edge (expected_net ≈ 0 → config issue).
- **Key Fixes during audit:** (1) expected_net/gross_exps/net_exps moved to final-w anchor (after risk_budget_floor + tradeable mask + capacity clip) so alpha_gap compares same w as realized. (2) non-tradeable sleeve skips excluded from dropped_below_cost count. (3) fold-local rebalance counter replaces global rebalance_count for n_rebal fallback.
- **Edge Cases:** `_assemble_fold_attribution` coerces any NaN input to 0.0 via `np.isfinite` guard. Empty throttle/exposure/sleeve lists default to 1.0/0.0/0.0. Zero-division on `friction_pass_ratio` guarded by `signal_total > 0`. `Layer2FoldAttribution` is frozen+slots. All new fields carry defaults → full backward compat.

## [2026-06-24] Cost-Aware Selection — Cost Drag Gate + Turnover Penalty
- **Delta:** Added `compute_cost_drag_ratio` (Σcost / max(Σprice, ε)). New fields in `Layer2AllocationConfig`: `l2_max_cost_drag_ratio=0.60`, `l2_turnover_penalty_weight=0.0`. Promotion blocker 17번째 `"cost_drag"` — cost drag > threshold 시 BLOCK. Objective `J`에 `- λ_t · mean_turnover` 항 추가 (λ=0 기본 → 하위호환). Attribution 3개 scalar(price/funding/cost) `if _diag:` 분리 → 무조건 누적. `K_RANK` search space low=1 → low=4 (k_rank=2 churn 원천 차단).
- **Rationale:** L2 음수 CAGR 원인이 realized turnover cost(11.0%) > gross price PnL(8.6%). 기존 friction gate은 per-entry 추정이라 누적 리밸런싱 회전 비용을 감지 불가. Cost drag hard gate가 비용>gross를 배포 전 차단. Turnover penalty는 선택기가 churn-prone config을 회피하도록 유도. Attribution 상시화로 gate가 항상 cost drag 평가 가능.
- **Key Fixes during audit:** `K_RANK` low=1→4 누락으로 audit FAIL → V2~V9 전 버전 일괄 수정. `ENGINE_PARAM_SPACE_FUTURES`는 L1 범위로 미변경.

## Phase 4: 후반 무결성 (6/19~21)
- Provenance fingerprint: ValidatedSignalBatch streaming SHA-256→study identity, 회귀 테스트 BY permutation/singleton/empty
- Purge WFA 활성화: L2 fold도 config purge/embargo(max_holding_bars×purge_safety_mult) 적용, fold 경계 label overlap 차단
- Scale collapse 이중수정: _book_edge_score double-deduct 제거(eff_hurdle 재차감→mu_bps는 net), project_all_caps allow_vol_upscale(Cap5 양방향 정규화)
- Parallel ProcessPoolExecutor+deterministic batching(batch-synchronous ask/tell, seed 재현성)
- Realization 정합: Cap5 하향전용→L* 직접 스케일, exchange_cap=10× 기본, deploy_leverage SSOT 전달
- 최종 promotion hardening: 최근 non-empty fold pass gate, exchange cap default 10.0 복원
