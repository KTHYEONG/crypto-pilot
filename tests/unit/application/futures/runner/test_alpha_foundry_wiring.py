from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.budget import build_l1_verification_units
from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaRecipe,
    CheapGateConfig,
    L2PosteriorPolicyConfig,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.diversity import compute_panel_correlation_matrix
from src.domain.futures.alpha_foundry.l2_policy import (
    convert_posterior_to_l2_sleeves,
    resolve_staged_search_budget,
)
from src.domain.futures.alpha_foundry.pipeline import run_alpha_foundry_pipeline
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _aligned_4h() -> AlignedMarketData:
    # 600 bars — long enough for the 8-on/8-off entry pattern in _panel() to
    # clear min_events=40 while staying under the default turnover cap.
    t, n = 600, 2
    dt = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    # Constant per-bar log-drift so the mean 3-bar gross return stays
    # comfortably above the ~11.25bps stress round-trip cost.
    close = (100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))).reshape(-1, 1) * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full_like(close, 1000.0),
        funding_2d=np.full_like(close, 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
    )


def _panel(rid: str = "r1") -> CandidateSignalPanel:
    a = _aligned_4h()
    t, n = a.close_2d.shape
    # 8-on/8-off cycle: sparse-entry semantics need real flat/reversal
    # transitions (a constant side=1 array yields zero entries).
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=a.datetimes,
        symbols=a.symbols,
        signed_score_2d=side.astype(np.float64),
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": rid},
    )


R1 = AlphaRecipe(
    recipe_id="r1",
    family="trend_ma",
    variant="ema_12_72",
    timeframe="4h",
    archetype="trend",
    indicator_params={"fast": 12, "slow": 72},
    side_rule_id="trend_follow",
    exit_policy_id="atr_trail_2",
    required_fields=("close",),
    causal_lag_bars=1,
    max_turnover_per_year=365.0,
)


class TestAlphaFoundryWiring:
    """Verify L0→L1→L2 bridge wiring end-to-end."""

    def test_l0_cheap_gate_feeds_l1_units(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        evs = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            config=CheapGateConfig(),
        )
        assert any(e.gate_passed for e in evs)
        units = build_l1_verification_units(
            evidences=evs,
            recipes={"r1": R1},
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        assert len(units) > 0

    def test_l1_posterior_feeds_l2_sleeves(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        evs = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            config=CheapGateConfig(),
        )
        import pandas as pd

        rows = [
            {
                "symbol": "BTCUSDT",
                "recipe_id": e.recipe_id,
                "family": "trend_ma",
                "timeframe": e.timeframe,
                "activation_context": "pooled",
                "net_bps": e.mean_net_bps,
                "fold_id": 0,
                "effective_weight": 1.0,
            }
            for e in evs
        ]
        post = shrink_l1_evidence_hierarchical(
            raw_rows=pd.DataFrame(rows),
            cost_model=ExecutionCostModel(),
            config=PosteriorGateConfig(),
        )
        sleeves = convert_posterior_to_l2_sleeves(
            posterior=post,
            cost_model=ExecutionCostModel(),
            config=L2PosteriorPolicyConfig(),
        )
        assert len(sleeves) > 0

    def test_full_pipeline_end_to_end(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        cheap_ev, l1_units, *_ = run_alpha_foundry_pipeline(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(cheap_ev) > 0
        assert len(l1_units) > 0

    def test_diversity_and_search_spaces_wired(self) -> None:
        panels = [_panel("r1"), _panel("r2")]
        active_mask = np.ones_like(panels[0].valid_mask_2d, dtype=np.bool_)
        corr = compute_panel_correlation_matrix(panels, active_mask)
        assert corr.shape == (2, 2)
        budgets = resolve_staged_search_budget(
            n_dimensions={"signal": 5, "risk": 3, "regime": 4, "deployment": 2},
            requested_trials=200,
            seed_count=1,
        )
        assert len(budgets) == 4


class TestDebugSummary:
    """S1-2: DEBUG summary emits [EVAL] log, no file write when artifact_write_enabled=False."""

    def test_debug_summary_logs_eval_line(self, caplog) -> None:
        import logging

        from src.domain.futures.alpha_foundry.cheap_gate import (
            emit_alpha_generation_debug_summary as emit_alpha_foundry_debug_summary,
        )
        from src.domain.futures.alpha_foundry.cheap_gate import (
            evaluate_panel_gate,
        )
        from src.domain.futures.alpha_foundry.contracts import (
            AlphaGateConfig,
            AlphaRecipe,
        )
        from src.domain.futures.signals.contracts import CandidateSignalPanel
        from src.domain.futures.strategy.common.alignment import AlignedMarketData
        from src.domain.futures.strategy.execution_cost import ExecutionCostModel

        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-05T00"), np.timedelta64(1, "h"))[:t]
        close = np.column_stack(
            [
                np.linspace(100.0, 124.0, t, dtype=np.float64),
                np.linspace(50.0, 61.0, t, dtype=np.float64),
            ]
        )
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            open_2d=close,
            high_2d=close,
            low_2d=close,
            close_2d=close,
            volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(),
            warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
            adv_usdt_2d=np.full((t, 2), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((t, 2), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2",
            variant="bor_v2_40_4h",
            params={"lookback": 40},
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=2,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(),
            valid_mask_2d=valid,
            metadata={"recipe_id": "bor_v2_40_4h"},
        )
        recipe = AlphaRecipe(
            recipe_id="bor_v2_40_4h",
            family="sparse_breakout_retest_v2",
            variant="bor_v2_40_4h",
            timeframe="1h",
            archetype="trend",
            indicator_params={"lookback": 40},
            side_rule_id="breakout",
            exit_policy_id="sparse",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=2000.0,
        )
        evidence = evaluate_panel_gate(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=ExecutionCostModel(),
            config=AlphaGateConfig(
                min_events=10,
                min_effective_n=5.0,
                min_candidate_rank_ic_tstat=0.0,
                min_nw_tstat=0.0,
                max_cost_drag_ratio=1.0,
                max_turnover_per_year=2000.0,
            ),
            bars_per_year=365.0 * 24.0,
            run_id="test_debug",
        )
        caplog.set_level(logging.DEBUG)
        emit_alpha_foundry_debug_summary(
            run_id="test_debug",
            timeframe="1h",
            evidences=(evidence,),
            debug_top_k_rows=3,
            debug_reject_bucket_rows=3,
        )
        assert "[EVAL]" in caplog.text
        assert "stage=af_generation" in caplog.text
        assert evidence.recipe_id in caplog.text


class TestMaybeWriteAlphaFoundryReport:
    """S1-6: DEBUG mode returns (None, None), artifact disabled returns (None, None)."""

    def test_debug_mode_no_file_write(self, caplog) -> None:
        import logging

        from src.domain.futures.alpha_foundry.bridge_helpers import (
            maybe_write_alpha_foundry_report,
        )
        from src.domain.futures.alpha_foundry.contracts import (
            AlphaFoundryBridgeReport,
            AlphaFoundryRuntimeConfig,
        )

        caplog.set_level(logging.DEBUG)
        report = AlphaFoundryBridgeReport(
            run_id="test",
            mode="gate",
            timeframe="4h",
            symbols=("BTCUSDT",),
            n_bars=100,
            n_panels_in=10,
            n_bound_panels=5,
            n_evidence=5,
            n_passed=3,
            n_rejected=2,
            reject_reason_counts={},
            elapsed_sec=0.0,
            json_path="",
            parquet_path="",
        )
        config = AlphaFoundryRuntimeConfig(
            mode="gate",
            artifact_write_enabled=False,
            observability_mode="debug_log",
        )
        jp, pp = maybe_write_alpha_foundry_report(
            report=report,
            evidence_rows=(),
            runtime_config=config,
        )
        assert jp is None
        assert pp is None
        assert "ARTIFACT_JSON_BEGIN name=alpha_foundry_report run_id=test" in caplog.text
        assert "ARTIFACT_CSV_BEGIN name=alpha_foundry_evidence run_id=test" in caplog.text

    def test_debug_mode_logs_evidence_rows_without_file_write(self, caplog) -> None:
        import logging

        from src.domain.futures.alpha_foundry.bridge_helpers import maybe_write_alpha_foundry_report
        from src.domain.futures.alpha_foundry.contracts import (
            AlphaFoundryBridgeReport,
            AlphaFoundryEvidenceRow,
            AlphaFoundryRuntimeConfig,
        )

        caplog.set_level(logging.DEBUG)
        report = AlphaFoundryBridgeReport(
            run_id="test_rows",
            mode="gate",
            timeframe="4h",
            symbols=("BTCUSDT",),
            n_bars=100,
            n_panels_in=10,
            n_bound_panels=5,
            n_evidence=1,
            n_passed=1,
            n_rejected=0,
            reject_reason_counts={"weak_tstat": 1},
            elapsed_sec=0.0,
            json_path="",
            parquet_path="",
        )
        row = AlphaFoundryEvidenceRow(
            run_id="test_rows",
            timeframe="4h",
            family="trend_pullback_continuation",
            variant="tpc_fast",
            recipe_id="r1",
            archetype="trend",
            n_events=12,
            mean_gross_bps=5.0,
            mean_cost_bps=1.0,
            effective_n=8.0,
            mean_net_bps=4.0,
            gross_lcb_bps=1.2,
            net_lcb_bps=1.1,
            nw_tstat=1.5,
            rank_ic=0.02,
            rank_ic_tstat=0.5,
            incremental_rank_ic=0.01,
            cost_drag_ratio=0.2,
            turnover_per_year=12.0,
            novelty_corr_max=0.1,
            compute_cost_score=0.1,
            event_hit_rate=0.5,
            payoff_skew=0.0,
            xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=0.1,
            bootstrap_lcb_bps=0.8,
            bootstrap_agree=True,
            gate_passed=True,
            handoff_tier="candidate",
            selected_for_l1=True,
            reject_reasons="",
            soft_flags="",
            bucket_key="trend::4h",
            bucket_rank=1,
            redundant_with="",
            bucket_eff_test_count=1.0,
            global_eff_test_count=1.0,
            l1_priority_score=1.0,
            l1_budget_units=1,
            tf_coverage_count=1,
            sign_agreement_ratio=1.0,
            corroboration_tier="confirmed",
            stage_label="candidate",
            created_at_ms=1,
        )
        config = AlphaFoundryRuntimeConfig(
            mode="gate",
            artifact_write_enabled=False,
            observability_mode="debug_log",
        )

        jp, pp = maybe_write_alpha_foundry_report(
            report=report,
            evidence_rows=(row,),
            runtime_config=config,
        )

        assert jp is None
        assert pp is None
        assert "test_rows" in caplog.text
        assert "trend_pullback_continuation" in caplog.text
        assert "selected_for_l1" in caplog.text

    def test_artifact_write_disabled_returns_none(self) -> None:
        from src.domain.futures.alpha_foundry.bridge_helpers import (
            maybe_write_alpha_foundry_report,
        )
        from src.domain.futures.alpha_foundry.contracts import (
            AlphaFoundryBridgeReport,
            AlphaFoundryRuntimeConfig,
        )

        report = AlphaFoundryBridgeReport(
            run_id="test",
            mode="gate",
            timeframe="4h",
            symbols=("BTCUSDT",),
            n_bars=100,
            n_panels_in=10,
            n_bound_panels=5,
            n_evidence=5,
            n_passed=3,
            n_rejected=2,
            reject_reason_counts={},
            elapsed_sec=0.0,
            json_path="",
            parquet_path="",
        )
        config = AlphaFoundryRuntimeConfig(
            mode="gate",
            artifact_write_enabled=False,
            observability_mode="off",
        )
        jp, pp = maybe_write_alpha_foundry_report(
            report=report,
            evidence_rows=(),
            runtime_config=config,
        )
        assert jp is None
        assert pp is None


class TestRunAlphaFoundryL0Gate:
    def test_mode_off_returns_empty_evidences(self) -> None:
        from src.domain.futures.alpha_foundry.bridge_helpers import (
            run_alpha_foundry_l0_gate,
        )

        config = AlphaFoundryRuntimeConfig(mode="off")
        result = run_alpha_foundry_l0_gate(
            panels=[],
            bindings=[],
            recipes={},
            aligned=None,
            cost_model=None,
            runtime_config=config,
            run_id="test",
            timeframe="4h",
        )
        assert len(result.evidences) == 0
        assert len(result.panels_for_l1) == 0

    def test_mode_audit_produces_result(self) -> None:
        from src.domain.futures.alpha_foundry.bridge_helpers import (
            AlphaFoundryL0Result,
            run_alpha_foundry_l0_gate,
        )
        from src.domain.futures.alpha_foundry.contracts import (
            PanelRecipeBinding,
        )

        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-17T00"), np.timedelta64(4, "h"))[:t]
        close = np.column_stack(
            [
                np.linspace(100.0, 124.0, t, dtype=np.float64),
                np.linspace(50.0, 61.0, t, dtype=np.float64),
            ]
        )
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            open_2d=close,
            high_2d=close,
            low_2d=close,
            close_2d=close,
            volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(),
            warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=2,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(),
            valid_mask_2d=valid,
            metadata={"recipe_id": "", "archetype": "trend"},
        )
        binding = PanelRecipeBinding(
            panel_index=0,
            recipe_id="synth_1",
            family="trend_ma",
            variant="ema_12_72",
            source="synthetic_recipe",
        )
        config = AlphaFoundryRuntimeConfig(
            mode="audit",
            artifact_write_enabled=False,
            observability_mode="off",
        )
        result = run_alpha_foundry_l0_gate(
            panels=[panel],
            bindings=[binding],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_audit",
            timeframe="4h",
        )
        assert isinstance(result, AlphaFoundryL0Result)

    def test_l0_gate_passes_runtime_config_to_pipeline(self) -> None:
        """Gap 2 regression: bridge_helpers must pass runtime_config to pipeline."""
        from src.domain.futures.alpha_foundry.bridge_helpers import (
            run_alpha_foundry_l0_gate,
        )
        from src.domain.futures.alpha_foundry.contracts import (
            PanelRecipeBinding,
        )

        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-17T00"), np.timedelta64(4, "h"))[:t]
        close = np.column_stack(
            [
                np.linspace(100.0, 124.0, t, dtype=np.float64),
                np.linspace(50.0, 61.0, t, dtype=np.float64),
            ]
        )
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            open_2d=close,
            high_2d=close,
            low_2d=close,
            close_2d=close,
            volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(),
            warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=dt,
            symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=2,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(),
            valid_mask_2d=valid,
            metadata={"recipe_id": "", "archetype": "trend"},
        )
        binding = PanelRecipeBinding(
            panel_index=0,
            recipe_id="synth_1",
            family="trend_ma",
            variant="ema_12_72",
            source="synthetic_recipe",
        )
        config = AlphaFoundryRuntimeConfig(
            mode="audit",
            artifact_write_enabled=False,
            observability_mode="debug_log",
        )
        result = run_alpha_foundry_l0_gate(
            panels=[panel],
            bindings=[binding],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_gap2",
            timeframe="4h",
        )
        assert hasattr(result, "report")
