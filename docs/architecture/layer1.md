---
title: Futures Signal Architecture
domain: futures.signals
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/signals/rules.py
  - src/domain/futures/signals/diagnostics.py
  - src/domain/futures/signals/contracts.py
  - src/domain/futures/signals/workflow.py
  - src/domain/futures/signals/timeframes.py
  - src/domain/futures/signals/ensemble.py
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/optimization/metrics.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/alpha_foundry/contracts.py
  - src/domain/futures/alpha_foundry/recipes.py
  - src/domain/futures/alpha_foundry/cheap_gate.py
  - src/domain/futures/alpha_foundry/diversity.py
  - src/domain/futures/alpha_foundry/budget.py
  - src/domain/futures/alpha_foundry/posterior.py
   - src/domain/futures/alpha_foundry/pipeline.py
   - src/domain/futures/alpha_foundry/bridge_helpers.py
   - src/domain/futures/alpha_foundry/multi_tf_fusion.py
change_triggers:
  - src/domain/futures/signals/rules.py
  - src/domain/futures/signals/diagnostics.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/alpha_foundry/contracts.py
  - src/domain/futures/alpha_foundry/recipes.py
  - src/domain/futures/alpha_foundry/cheap_gate.py
  - src/domain/futures/alpha_foundry/diversity.py
  - src/domain/futures/alpha_foundry/budget.py
  - src/domain/futures/alpha_foundry/posterior.py
   - src/domain/futures/alpha_foundry/pipeline.py
   - src/domain/futures/alpha_foundry/bridge_helpers.py
   - src/domain/futures/alpha_foundry/multi_tf_fusion.py
dependencies:
  documents:
    - docs/architecture/regime.md
last_verified: 2026-07-06
---

# 1. Purpose
벡터화된 Rule Panels에 시장 Regime Context를 결합하고, L1 Breakeven Hard Gate 및 Multiplicity Control을 적용하여 유효한 Candidate Events를 생성한다. 4개의 Timeframe(4h/6h/8h/12h)에 걸쳐 Walk-forward Validation을 지원하는 Prequential Evidence Snapshots를 관리한다.

# 2. Core Logic & Math

### Signal Gating Sequence
1. **Vectorization**:
   - $S_{t} = f(\text{Data}_{1..t})$
   - Sparse Trigger: $E_{t} = 1 \text{ if } (S_{t} \neq 0 \land S_{t-1} == 0) \text{ else } 0$ (Causal)
2. **Regime Gating**:
   - `mean_rev` Archetype: `("bull_quiet", "bear_quiet", "transition")` 허용
   - `beta_neut` Archetype: `("bull_quiet",)` 허용
3. **L1 Breakeven Hard Gate**:
   - $\frac{1}{N} \sum (\text{Edge}_{i}) > 0 \land t_{\text{stat}} \geq \text{min\_rule\_ir\_t}$
4. **Profit Floor**:
   - $\mu_{\text{OOS}} \geq \text{min\_variant\_oos\_profit\_bps}$
5. **Regime-Cell Admission**:
   - Bayesian Posterior: $P(\mu > \delta | \text{data}) \ge p_{\text{admit\_min}}$
   - Newey-West variance와 cross-cell $\tau^2$ shrinkage 적용.
6. **Multiplicity Controls**:
   - **BH-FDR**: $q \le \text{l1\_pair\_fdr\_alpha}~(0.10)$
   - **SPA (Single Predictive Ability)**: Fail-closed circular bootstrap 검정.

### Ensemble Shrinkage
- Archetype cell mean과 variant-level prior에 대해 Empirical-Bayes James-Stein shrinkage 적용.
- Bayesian Prior: $\hat{x}_a = w_{prior} \cdot \bar{x}_a + (1 - w_{prior}) \cdot \mu_{prior}$
- $w_{prior} = n_{eff} / (n_{eff} + n_{prior})$

### Cross-Sectional Alpha Families (xs_alpha archetype)
- 4대 Family: `xs_momentum`, `xs_carry`, `xs_flow`, `xs_oi_skew`
- Per-bar rank score 변환을 통한 beta-neutral 구성 (Regime Gating 면제).
- `xs_momentum`: Rolling beta residual return (L=12/48)
- `xs_carry`: Funding rate z-score 역수 (`-funding_z_96/168`)
- `xs_flow`: Order flow imbalance z-score (`flow_z_24`)
- `xs_oi_skew`: Open Interest 빌드업과 LSR 스큐 결합 (`-(oi_build_z_42 * sign(lsr_log_z_42))`)

### Signal Family Registry [ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY]
- `ALL_SIGNAL_FAMILIES` (`signals/rules.py`, `strategy/rule_signals.py`, 두 모듈 각자 동일 값 유지): 27개 rule family 모듈 상수. 두 모듈은 family 연산 로직을 병행 유지하므로 값 동기화가 계약.
- `CandidateStrategyConfig.candidate_families`: native TF 빌드 시 `filter_rule_signal_panels()`가 적용하는 전역 allowlist(27종 전량 포함).
- `resolve_tf_signal_pool(cfg, tf)` / `_DEFAULT_PER_TF_FAMILIES`: HTF(6h/8h/12h) 파생 패널 전용 family allowlist. `build_multi_tf_panels()`가 `family_filter`로 직접 주입 — native TF의 `candidate_families`와는 별개 축.
- `resolve_family_registration_gap(all_families, candidate_families)`: `all_families` 중 `candidate_families`에 없는 항목을 반환하는 순수 함수(orphan 탐지).
- `src/domain/futures/strategy/family_lifecycle.py`: `FAMILY_TF_RETIREMENT`(economic replay 기각 이력의 `(family, tf)` 집합) + `is_family_tf_retired()` — 동일 조합 재검증 방지 가드.
- **L0 게이트 스코프**: `run_alpha_foundry_l0_gate()`는 native TF panel에만 적용된다(`strategy_runtime/bridge.py`, native panel 생성 직후 호출). HTF 파생 패널(`build_multi_tf_panels`)은 그 이후 별도 생성되어 L0 경제성 게이트(LCB/tstat/cost-drag)를 거치지 않고 `resolve_tf_signal_pool` family 필터만 적용된 채 L1 fold 평가로 직행한다.
- `_resolve_panel_archetype(panel)` [ADR_20260707_L1_BACKTEST_FIDELITY_FIXES]: `panel.metadata.archetype`가 없을 때 family 문자열로 archetype을 역산하는 fallback. trend 집합 = `{trend_ma, trend_donchian, vol_breakout, trend_pullback_continuation, mtf_trend_pullback, mtf_breakout_retest, vol_term_structure_gate, btc_regime_pullback}`. 미매칭 family는 전부 `mean_rev`로 폴백 — 신규 family 추가 시 이 함수 갱신 필수(누락 시 `exit_policies.build_exit_policies_for_panel()`이 엉뚱한 archetype 버킷의 손절/익절을 적용).
- `evaluate_compound_backtest()`(`candidate_evaluation.py`)/`build_candidate_target_weights()`(`candidate_portfolio.py`)의 연율화(`bars_per_year`)는 `optimization/metrics._bars_per_year_for_tf(tf)` SSOT로 통일(4h/1h/1d 하드코딩 elif 체인 제거, 그 외 TF는 4h로 암묵 폴백하던 결함 해소).

### Alpha Foundry Core [ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY][ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
- `AlphaRecipe`: `recipe_id`, `family`, `variant`, `timeframe`, `archetype`, `indicator_params`, `side_rule_id`, `exit_policy_id`, `required_fields`, `causal_lag_bars`, `max_turnover_per_year`.
- `BucketKey`: `tuple[str, str]` alias for `(family, timeframe)` — diversity selection grouping key.
- `CorroborationTier`: `Literal["corroborated", "single_tf_strict", "contradicted", "insufficient_coverage"]`.
- `CheapGateEvidence`: 경제성 지표 전용(`gate_passed`, `reject_reasons`, `n_events`(sparse entry count), `effective_n`, `block_lcb_bps`, `nw_tstat`(block moments 기반), `rank_ic`, `turnover_per_year`, `cost_drag_ratio`, `incremental_rank_ic`, `compute_cost_score`, `novelty_corr_max`, `bootstrap_lcb_bps`, `bootstrap_agree`, `mean_gross_bps`(건당 평균, `total_gross/n_events`), `total_cost_bps`(⚠️ 건당 평균이 아닌 전체 합계 — `mean_gross_bps`/`mean_net_bps`와 단위 불일치, 비교 시 `n_events`로 직접 나눠야 함)). `monotonic_bucket_score`/`regime_edges_bps`는 미사용 필드로 제거됨.
- `L0SoftFlag`에 `"weak_rank_ic"` 추가: `_rank_ic_soft_floor(n_events) = 1/√(n_events-3)`보다 `abs(rank_ic)`가 작으면 부여(표본크기 적응형, 고정 임계치 아님). `discovery_tier`의 `blocked` 판정에는 무관하고 `L0PriorityWeights.weak_rank_ic_multiplier`(기본 0.70)로 `l1_priority_score`만 감쇠.
- `audit_full_family_correlation()`(`diversity.py`): `AlphaFoundryRuntimeConfig.enable_correlation_audit`(기본 `False`) 활성 시에만 native panel 빌드 직후(게이트 이전) 전체 family 상관행렬 + `estimate_effective_test_count()`(entropy 기반 실효 독립개수)를 `{run_id}_family_correlation.parquet`로 기록하는 opt-in 진단 경로.
- `DiversitySelectionResult`: 버킷 단위 산출물(`bucket_key`, `ranked_recipe_ids`, `selected_recipe_ids`, `redundant_recipe_ids`, `redundant_reason_by_id`, `bucket_corr`, `bucket_eff_test_count`). `redundant_reason_by_id` 값은 상관 상대 recipe_id 외에 `"bh_rejected"`/`"below_conviction_floor"` sentinel도 가짐.
- `CrossBucketDiversityResult`: 교차버킷 최종 산출물(`final_selected_recipe_ids`, `demoted_recipe_ids`, `demoted_reason_by_id`, `cross_bucket_corr`, `global_eff_test_count`).
- `MultiTimeframeEvidence`: `family`, `variant`, `native_timeframe`, `native_recipe_id`, `tf_coverage_count`, `sign_agreement_ratio`, `corroboration_tier`, `fused_conviction_score`.
- `AlphaFoundryEvidenceRow`: parquet 영속화 스키마 — family/variant별 `mean_net_bps`/`block_lcb_bps`/`cost_drag_ratio`/`turnover_per_year`/`bootstrap_lcb_bps`/`bootstrap_agree`/`selected_for_l1`/`bucket_eff_test_count`/`global_eff_test_count` 비교 단위.
- `L1VerificationUnit`: fold-bounded verification unit with `prior_mu_bps`, `prior_sigma_bps`, `allocated_fold_budget`, `early_stop_state`.
- `L1PosteriorEvidence`: `posterior_mu_bps`, `posterior_sigma_bps`, `prob_mu_gt_cost`, `lcb_net_bps`, `quality_weight`, `activation_contract`.
- `evaluate_panel_cheap_gate(bars_per_year=...)` / `evaluate_alpha_cheap_gate_batch()` — `bars_per_year`는 `_bars_per_year_for_tf(recipe.timeframe)`(`optimization/metrics.py` SSOT) 주입. `n_events`는 sparse entry mask(flat→active 또는 direct 부호반전, 연속보유 bar 중복계산 방지)로 산출, `effective_n = n_events`. `block_bars_eff = max(config.block_bars, 2*holding_bars)`로 블록 크기를 보유기간에 연동, `nw_tstat = mu_block/se_block`(block moments 통일). `bootstrap_lcb_bps`/`bootstrap_agree`는 block-mean 복원추출(`config.bootstrap_samples`, `config.bootstrap_seed`)로 산출(정보성, 게이트 미반영). 다양성/novelty 계산은 이 단계에서 수행하지 않음(단일 책임).
- `apply_bucket_bh_correction()` — 버킷 내 후보의 `nw_tstat` 기반 양측 p-value에 BH step-up(`config.fdr_alpha`) 적용, 유의하지 않은 후보를 랭킹 전에 배제.
- `select_bucket_diverse_recipes()` — BH 미통과·`min_conviction_lcb_bps` 미달 후보를 우선 배제한 뒤, 버킷(`family`, `timeframe`) 내 `block_lcb_bps` 내림차순 그리디 선택. `top_k_per_family_tf` 예산과 `max_novelty_corr` 상관 임계 동시 적용, 조기종료로 O(top_k·K_b) 상관 비교.
- `resolve_cross_bucket_diversity()` — 버킷별 selected 합집합(S, 상수 상한)에 한해 `compute_panel_correlation_matrix()`/`cluster_correlated_recipes()`로 교차 중복 제거, `estimate_effective_test_count()`로 `global_eff_test_count` 산출.
- `allocate_global_l1_budget()` — 버킷 대표품질(`max(block_lcb_bps)` over selected) 기준 largest-remainder 비례배분, 품질 0인 버킷은 슬롯 0, 버킷별 `top_k_max` 상한(기존 `top_k_per_family_tf` 값 재해석). `top_k_per_family_tf` 고정 캡을 대체.
- `fuse_multi_timeframe_evidence()` (`multi_tf_fusion.py`) — 동일 run epoch의 TF별 evidence DataFrame을 `(family, variant)`(TF 접미사 정규화 후) 기준으로 조인해, 다른 TF와의 부호일치도로 `corroboration_tier` 판정. `contradicted`는 `fused_conviction_score`를 강제 음수화(사실상 거부권), `corroborated`는 15% 컨빅션 부스트.
- `shrink_l1_evidence_hierarchical()` applies family/timeframe shrinkage with `w = n_eff / (n_eff + prior_effective_n)`.

### Alpha Foundry Bridge Wiring [ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY][ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
- `PanelRecipeBinding`: binding record with `panel_index`, `recipe_id`, `family`, `variant`, `source` (`catalog_exact`/`catalog_family_variant`/`synthetic_recipe`).
- `AlphaFoundryBridgeReport`: per-TF report with `mode` (off/audit/gate), `n_panels_in`, `n_bound_panels`, `n_evidence`, `n_passed`, `n_rejected`, `reject_reason_counts`, `json_path`, `parquet_path`.
- `AlphaFoundryL0Result`: typed result from L0 gate with `panels_for_l1`, `report`, `evidences`, `bindings`.
- `run_alpha_foundry_l0_gate(panels, bindings, recipes, aligned, cost_model, runtime_config, run_id, timeframe)` — panel→recipe matching, cheap gate → BH-lite/conviction floor → 버킷 다양성 선택 → 교차버킷 중복제거 실행, audit/gate 필터링, `AlphaFoundryEvidenceRow` parquet 실기록 + JSON 집계.
- `bind_panels_to_alpha_recipes(panels, recipes, timeframe, max_recipes_per_family, include_families, exclude_families, enable_synthetic_recipes=True)` — 카탈로그 미매칭 family는 `map_signal_archetype_to_alpha_archetype()`(`recipes.py`, `SignalArchetype`→`AlphaArchetype` 8→6종 매핑)로 합성 `AlphaRecipe`를 생성해 `recipes`(MutableMapping)에 즉시 등록(`source="synthetic_recipe"`); `enable_synthetic_recipes=False`면 기존 allowlist-only 동작(미매칭 시 폐기).
- `_write_alpha_foundry_report(report, evidence_rows, report_dir, run_id)` — JSON(`{tf}_{timestamp}_report.json`, 집계)과 parquet(`{tf}_{timestamp}_evidence.parquet`, `AlphaFoundryEvidenceRow` 전건) 동시 기록.
- Mode behavior: `"audit"` preserves all bound panels with `gate_passed`/`selected_for_l1` flags; `"gate"` forwards only `final_selected_recipe_ids`(Stage3 산출물) — zero-survivor closes tiered L1 entry.

# 3. Core I/O Interfaces

### Input Data
- `AlignedMarketData`: OHLCV, indicators, flow, funding rates
- `MarketRegimeContext`: Compressed 3-state (`bull`, `bear`, `crisis`) 및 6-state regime codes

### Principal Data Structures (src/domain/futures/signals/contracts.py)
- `CandidateSignalPanel`:
  - `signed_score_2d`: NDArray[np.float64]
  - `side_hint_2d`: NDArray[np.int8]
  - `allowed_regimes`: tuple[RegimeName, ...]
  - `exit_policies`: tuple[SignalExitPolicy, ...]
- `SymbolStrategyEvidence`:
  - `mean_gross_bps`: float
  - `mean_incremental_bps`: float
  - `block_tstat_incremental`: float
  - `q_value`: float
  - `quality_weight`: float
  - `hard_eligible`: bool
  - `lcb_net_bps`: float
  - `adverse_regime_lcb_bps`: float | None — bear/crisis 구간 LCB(진단 전용, `quality_weight` 산식 미반영)
  - `adverse_regime_n_obs`: int
  - `adverse_regime_defended`: bool — `compute_adverse_regime_evidence()` 산출 [ADR_20260705_L1L2_REGIME_CONDITIONAL_WEIGHT]
- `QualifiedSignalRegistry`:
  - `by_symbol`: dict[str, tuple[SymbolStrategyEvidence, ...]]
  - `ready_symbols`: tuple[str, ...]

# 4. Architecture Flow

```mermaid
graph TD
    A[Market Data] --> B[Vectorized Indicators]
    B --> C[CandidateSignalPanel 28 Families]
    C --> C1[Multi-TF Panel Injection]
    C1 --> D[Archetype & Regime Context]
    D --> E[L1 Breakeven & Profit Floor Gate]
    E --> F[Regime-Cell OR-path Admission]
    F --> G[Multiplicity: FDR & SPA]
    G --> H[Promoted Candidate Events]
    H --> I[Per-TF L1 Pipeline 4h/6h/8h/12h]
    I --> J[L2 on Master TF]
```

# 5. Readiness & Promotion Gates

### Readiness Gate (5 Conditions)
- **Fold Coverage**: $\ge 0.80$
- **Match Ratio**: $\ge 0.90$
- **Effective $N$ ($N_{eff}$)**: $\ge 3.0$
- **Fold Ratio**: $\ge 0.50$
- **Pooled LCB**: $> 0$

### Promotion Gate (4 Conditions)
- **Hard Eligible**: Structural gates pass 통과 여부
- **LCB Net**: `lcb_net_bps > l1_breakeven_floor_bps` (~7.5 bps)
- **BH-FDR**: $q\text{-value} \le 0.10$
- **Quality Weight**: `quality_weight > 0`

# 6. Performance Optimizations
- **Adaptive Worker Cap**: 메모리 계산 및 fold 분할수에 따른 병렬 worker 수 조절.
- **Prefit Overlap (OPT-4)**: Evidence fold 수집 단계에서 Layer 1 모델을 백그라운드로 prefit하여 대기 시간 제거.
- **Prefork Cache Prime**: Process fork 이전에 shared cache를 초기화하여 CoW 메모리 효율화.
- **Numba JIT Acceleration**: Rolling z-score, cross-sectional rank, warm/ready 검사에 `@njit(cache=True)` 적용.
- **t.ppf LRU Cache**: Student-t 분포의 Percent Point Function 연산을 `@lru_cache`화하여 중복 연산 제거.
- **searchsorted Indexing (OPT-1)**: 시계열 datetime 매칭 탐색을 $O(T)$에서 $O(\log T)$로 최적화.
