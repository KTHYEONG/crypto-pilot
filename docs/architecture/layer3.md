---
title: Futures Holdout & Validation Architecture (Layer 3)
domain: futures.validation
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/validation/champion_registry.py
  - src/domain/futures/validation/gates.py
  - src/domain/futures/validation/walk_forward.py
  - src/domain/futures/optimization/candidate_selector.py
  - src/domain/futures/optimization/final_evaluator.py
  - src/domain/futures/strategy/tiered_workflow/major_symbol_registry_replay.py
  - src/domain/futures/strategy/tiered_workflow/tf_validation_repair.py
  - src/application/futures/runner/active_pipeline.py
  - src/application/futures/runner/tf_probe_scoped.py
  - src/application/futures/runner/cli.py
  - src/application/futures/runner/config.py
  - src/domain/futures/optimization/opt_config.py
  - src/domain/futures/optimization/opt_data_utils.py
  - src/domain/futures/optimization/observability/run_tracker.py
change_triggers:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/validation/champion_registry.py
  - src/domain/futures/validation/gates.py
  - src/domain/futures/optimization/final_evaluator.py
  - src/domain/futures/optimization/opt_config.py
  - src/domain/futures/optimization/opt_data_utils.py
  - src/domain/futures/optimization/observability/run_tracker.py
  - src/domain/futures/strategy/tiered_workflow/major_symbol_registry_replay.py
  - src/domain/futures/strategy/tiered_workflow/tf_validation_repair.py
  - src/application/futures/runner/active_pipeline.py
  - src/application/futures/runner/tf_probe_scoped.py
  - src/application/futures/runner/cli.py
  - src/application/futures/runner/config.py
dependencies:
  documents:
    - docs/architecture/layer1.md
    - docs/architecture/layer2.md
last_verified: 2026-07-06
---

# 1. Purpose
최종 OOS(Out-of-Sample) 구간에 대한 검증을 담당한다. Tiered Hybrid Architecture 내의 완전 격리된 "Frozen Holdout" 평가와 최종 Optuna 최적화 과정에서 다중 시드를 이용해 모사 성능의 오버피팅 여부를 판별하는 "Layer 3 Stability Check"로 구성된다.

# 2. Core Logic & Math

### Layer 3 — Frozen Holdout (Tiered Pipeline)
L2 AWF 시뮬레이션에서 결정된 최적 하이퍼파라미터(`l2_params`)와 배포 레버리지($L^*$)를 완전 동결(Frozen)하고, 미관측 홀드아웃 윈도우 `[ho_start, ho_end)`에서 단일 패스 시뮬레이션을 수행한다.

- **Deployment Parity**:
  - $L^* > 1.0$ 인 경우 `apply_deployment(rets, L*)` 수식을 적용해 평가.
  - $L^* \le 1.0$ 인 경우 unit path (leverage = 1.0) 성과를 그대로 적용.
- **Performance Formulas**:
  - CAGR, Sharpe, Sortino: Base TF의 연간 바 개수(`bars_per_year(tf)`)를 사용해 계산.
  - MAR Ratio: $\text{MAR} = \frac{\text{CAGR}}{\text{MDD} + 10^{-9}}$
  - Terminal Compounding: `equity_multiple - 1` (단일 패스 복리 종가 기준 산출)

### Diagnostic Attribution Metrics `[ADR_20260704_L3_MAJORDIAG]` `[ADR_20260704_L3_INCOHERENCE]`
- **Reversal-Kill & Regime Mix**: OOS 구간의 Regime 분포(`regime_bull_pct`, `regime_bear_pct`, `regime_crisis_pct`)와 Reversal-Kill 동작 여부를 결합하여 분석.
- **Long/Short P&L Decomposition**: 롱 비중과 숏 비중을 분리하여 각 다리의 실현 수익 및 참여율 산출.
  - $w_{long} = \max(w, 0), \quad w_{short} = \min(w, 0)$
- **Per-Symbol Long/Short Attribution**: 심볼별 P&L 기여도를 분해하여 특정 자산군의 쏠림 현상을 모니터링.
- **Regime-Mu Incoherence**: 역 regime(bear, crisis) 환경에서 매수(bullish) 시그널이 유지되는 비율 및 반전 지연 시간(Reversal Lag) 측정.

### Layer 3 — Multi-Seed Stability Check
Optuna 최적화로 최종 선별된 챔피언 전략에 대해 $N$개의 서로 다른 랜덤 시드로 AWF 시뮬레이션을 재실행하여 파라미터 안정성을 검증한다.
- **Stability Gate**: `FUTURES_TMP_LAYER3_HARD_GATE` 활성화 시, 모든 시드에서 L1 Hard Gate 조건들을 통과해야 최종 승인된다.

### Major-Symbol Registry Replay
- `MAJOR_SYMBOL_REGISTRY_REPLAY=1`이면 `run_tiered_pipeline`이 L2 결과 직후 baseline/treatment replay를 내부 실행하고 CSV를 쓴다.
- replay seed는 `FuturesRunConfig.seed` SSOT를 사용하며, 기본 경로는 `docs/results/major_symbol_registry_replay_seed_<seed>.csv`다.
- replay rows는 `baseline_parity`, `l2_cagr`, `l3_total_return`, `l3_cagr`, `l3_mdd`, `l3_sharpe`, `l3_sortino`, `l3_trade_count`, `registry_census`를 포함한다.

### Scoped TF Probe Reconnect `[ADR_20260705_TF_PROBE_SCOPED_SYNC]`
- `src/application/futures/runner/tf_probe_scoped.py`는 `full_strategy_maps`를 입력으로 받아 `probe_timeframe_alpha()`를 majors-only 기본 스코프로 실행하는 전용 wrapper다.
- `_run_strategy_stage()`는 `full_strategy_maps` 확보 직후, `data_stage.data_maps.clear()` 이전에 `_run_tf_probe_stage_scoped()`를 호출한다.
- 기본 스코프는 `("BTCUSDT", "ETHUSDT", "BNBUSDT")`, OOM 상한은 `20` symbols다.
- 반환은 `TfProbeStageResult | None`이며, 내부 gate 로직은 기존 `select_tf_family_cells()`와 `summarize_tf_probe_gate_audit()`를 그대로 사용한다.
- `probe_stage_result_to_raw_manifest()`는 scoped probe 결과를 raw rows로 변환해 clear 이후에도 `run_tiered_pipeline`이 parity capture를 복원할 수 있게 한다.

### Rolling Holdout Panel (Multi-Episode Validation) [ADR_20260705_L3_ROLLING_HOLDOUT_PANEL]
- **Windowing Duality**: 홀드아웃 실행(`LayeredWindow`, `get_layered_window`, REGIME_FLOOR 클램프 적용)과 심볼 readiness 필터링(`QuarterlyWindow`, `get_quarterly_window`, 클램프 없음)은 `opt_config.py`에 **별도 함수로 독립 계산**된다 — `run_config.date` 하나에서 두 창을 각각 파생.
- `ValidationEpisode(episode_id, reference_date, role: "promotion"|"stress_only", window: LayeredWindow)`: `build_validation_episode_panel(promotion_reference_dates, stress_reference_dates, l1_months, l2_months, holdout_months, warmup_days)`가 `get_layered_window()`를 반복 호출해 생성. `role="stress_only"`는 `regime_floor=date.min`으로 클램프를 우회.
- `EpisodeOutcome(episode_id, role, candidate_total_return, baseline_total_return)` → `evaluate_rolling_holdout_consistency()`가 `RollingConsistencyVerdict(consistent_improvement, stress_generalization_pass, n_promotion_episodes, failing_episode_ids)`를 산출 — 전 promotion episode에서 candidate≥baseline이어야 `consistent_improvement=True`.
- **ADR-레벨 Sharpe Pool**: `adr_sharpe_pool_study_name(tag)` Optuna study(승패 무관 전량 기록, `champion_store_study_name`과 별개)에 `record_adr_evaluation()`으로 적재, `compute_adr_level_deflated_sharpe()`가 기존 `optimization.metrics._deflated_sharpe_probability`(Bailey & López de Prado 2014, 신규 공식 아님, `[ADR_20260706_PRODUCTION_PIPELINE_CONSOLIDATION]`로 `allocation/`에서 이관됨)를 이 pool로 재호출.
- **오케스트레이션 미배선**: 위 3세트를 실제 여러 episode에 대해 `run_tiered_pipeline`을 반복 실행하는 통합 루프는 아직 구현되지 않음(순수 함수만 존재).
- **Data-Readiness 진단**: `_run_data_stage`(`active_pipeline.py`)의 `_build_data_not_ready_reasons()`가 `kept_symbols=()`일 때 `reason` 분포를 `RuntimeError` 메시지에 포함(`[ADR_20260706_PRODUCTION_PIPELINE_CONSOLIDATION]`). 실측 확인된 reason 값: `fetch_window_short`, `warmup_insufficient` — `QuarterlyWindow`의 `fetch_start`가 `--date`에 따라 이동하며 발생.
- **Warmup Buffer 실측 정합** `[ADR_20260706_DATA_WINDOW_FLOOR_CONSISTENCY]`: `get_layered_window`/`get_quarterly_window`의 `fetch_start` 버퍼(`warmup_days`)는 하드코딩 365일 대신 `resolve_warmup_days_for_tf(tf)`(`opt_data_utils.py`, 기존 `_resolve_warmup_bars` 재사용)로 계산 — 4h 기준 62일. `warmup_days`를 명시적으로 넘기면 그 값이 우선(하위 호환). 근거: 48개월(요구) vs ~51개월(실제 데이터 가용, 2022-04-01~) 예산에서 365일 버퍼가 여유 3개월을 전부 소진해 `--date` 이동 시 전 심볼 탈락을 유발했음을 실측 확인.

### Alpha Foundry Report Logging [ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING]
- `_run_strategy_stage()`(`active_pipeline.py`) captures `CandidatePipelineOutput.alpha_foundry_report` from the bridge stage and logs it at INFO level — present in both audit and gate modes.
- `CandidatePipelineOutput.alpha_foundry_report: AlphaFoundryBridgeReport | None` — `None` when mode=off.
- Report fields (`panels_in`, `bound`, `survivors`, `reject_breakdown`) are available for downstream diagnostics and JSON artifact at `logs/futures/alpha_foundry/`.

### Alpha Foundry Runtime Config [ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
- `application/futures/runner/config.py` builds and validates `AlphaFoundryRuntimeConfig` with `observability_mode`, `debug_top_k_rows`, `artifact_write_enabled`, and `gate_schema` in addition to the existing gate and L2 policy fields.
- `validate_alpha_foundry_runtime_config()` rejects invalid observability mode, non-unified gate schema, and `debug_top_k_rows < 1`.

### Validation Parity Report Flow [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
- `build_validation_parity_capture()`는 pre-clear probe/main/census evidence를 묶고, `finalize_validation_parity_capture()`는 이후 L2/L3 sleeve evidence로 major-gap 클래스를 확정한다.
- `validation_parity_report`는 `Layer1Result`, `Layer2Result`, `Layer3Result`를 통해 downstream으로 유지된다.

# 3. Architecture Flow

```mermaid
graph TD
    A[Layer 2 Result & Params] --> B[Define Dummy WFFold for Holdout]
    B --> C[Run AWF Simulation with Frozen Params]
    C --> D[Compute L3 Metrics: CAGR, MDD, MAR, Sharpe]
    D --> E{L3 Gate: Hybrid >= Baseline?}
    E -->|Pass| F[Emit Layer3Result]
    E -->|Fail| G[Gate Blocked / Revert to Baseline]
    
    H[Optuna Best Trial] --> I[Layer 3 Stability Check]
    I --> J[Re-run AWF across N target seeds]
    J --> K{Pass L1 Hard Gates?}
    K -->|Pass| L[Champion Promotion Evaluation]
    K -->|Fail| M[Block Promotion]
    N[full_strategy_maps ready] --> O[_run_tf_probe_stage_scoped]
    O --> P[TF Probe Manifest]
    P --> Q[Scoped gate audit]
    N --> R[data_stage.data_maps.clear()]
    O -.before clear.-> R
```

# 4. Holdout Gates (L3 Validation Seam)
L3 Holdout 검증 완료를 위해 포트폴리오는 아래 순차적 조건문(Short-circuit)을 모두 통과해야 한다.

1. **No Holdout Returns**: OOS 구간 내 거래가 전혀 없거나 수익률 배열이 비어있지 않아야 함.
2. **Non-Finite Check**: CAGR, MDD, Sharpe, Sortino 등 지표가 Finite 값이어야 함.
3. **Minimum Trades**: 총 거래 횟수 $n_{trades} \ge \text{min\_trades} \quad (10)$
4. **Positive Return**: 누적 복리 수익률 $\text{total\_return} > 0$
5. **Absolute MDD Limit**: 최대 낙폭 $\text{MDD}_{hybrid} \le \text{max\_mdd\_abs} \quad (0.35)$
6. **CVaR95 Limit**: 95% CVaR 테일 리스크 $\text{CVaR95}_{hybrid} \le \text{max\_cvar95} \quad (0.06)$
7. **Absolute Sharpe**: $\text{Sharpe}_{hybrid} \ge 0.0$
8. **Absolute Sortino**: $\text{Sortino}_{hybrid} \ge 0.0$

# 5. Core Components

| Module | Role |
|---|---|
| `tiered_workflow/pipeline.py` | `run_l3_holdout` 진입점 제공, dummy fold 생성 및 시뮬레이션 호출 |
| `validation/walk_forward.py` | frozen 파라미터를 사용한 시뮬레이션 실행 루프 및 fold diagnostics 생성 |
| `validation/champion_registry.py` | `Layer3Result` 정의 및 L3 Holdout Gate 논리 평가 |
| `optimization/candidate_selector.py`| 다중 시드 검증용 `check_stability_layer3` 구현 |
| `optimization/final_evaluator.py` | 챔피언 선출 및 최종 L3 안정성 검증 오케스트레이션 |
| `strategy/tiered_workflow/major_symbol_registry_replay.py` | major-symbol registry replay, adoption gate, CSV artifact |
| `strategy/tiered_workflow/tf_validation_repair.py` | pre-clear TF parity capture, major-gap classification, report logging |
| `runner/tf_probe_scoped.py` | majors-only TF probe wrapper, scoped gate audit, pre-clear execution |
| `application/futures/runner/cli.py` | CLI seed 전달, replay entrypoint, `--alpha-foundry` (off/audit/gate) |
| `application/futures/runner/config.py` | `FuturesRunConfig.seed` SSOT, `FuturesRunConfig.alpha_foundry` 필드, `build_alpha_foundry_runtime_config()`, `validate_alpha_foundry_runtime_config()` |
