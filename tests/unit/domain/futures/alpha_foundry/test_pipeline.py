from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryEvidenceRow,
    AlphaGateEvidence,
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    DiversitySelectionResult,
    L0SignalCandidate,
    L2PosteriorPolicyConfig,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.pipeline import (
    build_alpha_foundry_evidence_row,
    build_l0_handoff_decisions,
    build_l2_sleeves_from_posterior,
    build_posterior_from_l1_fold_rows,
    run_alpha_foundry_l0_pipeline,
    run_alpha_foundry_pipeline,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_aligned() -> AlignedMarketData:
    # 600 bars — long enough for the 8-on/8-off entry pattern in _make_panel()
    # to clear min_events=40 while staying under the default turnover cap.
    t, n = 600, 2
    datetimes = np.arange(
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
        datetimes=datetimes,
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


def _make_panel() -> CandidateSignalPanel:
    aligned = _make_aligned()
    t, n = aligned.close_2d.shape
    # 8-on/8-off cycle: sparse-entry semantics need real flat/reversal
    # transitions (a constant side=1 array yields zero entries).
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=side.astype(np.float64),
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )


SAMPLE_RECIPE = AlphaRecipe(
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


class TestRunAlphaFoundryPipeline:
    def test_pipeline_produces_all_stage_outputs(self) -> None:
        panel = _make_panel()
        aligned = _make_aligned()
        cheap_ev, l1_units, _post, _slv = run_alpha_foundry_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(cheap_ev) > 0
        assert len(l1_units) > 0

    def test_pipeline_empty_panels_returns_empty(self) -> None:
        aligned = _make_aligned()
        cheap_ev, l1_units, _post, _slv = run_alpha_foundry_pipeline(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(cheap_ev) == 0
        assert len(l1_units) == 0
        assert len(_post) == 0
        assert len(_slv) == 0


class TestRunAlphaFoundryL0Pipeline:
    def test_l0_pipeline_returns_artifacts(self) -> None:
        panel = _make_panel()
        aligned = _make_aligned()
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
        )
        assert len(artifacts.evidences) > 0
        assert isinstance(artifacts.passed_recipe_ids, tuple)
        assert isinstance(artifacts.reject_reason_counts, dict)

    def test_l0_pipeline_empty_panels_returns_empty_artifacts(self) -> None:
        aligned = _make_aligned()
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
        )
        assert len(artifacts.evidences) == 0
        assert len(artifacts.passed_recipe_ids) == 0

    def test_l0_pipeline_with_runtime_config_creates_search_cells(self) -> None:
        panel = _make_panel()
        aligned = _make_aligned()
        from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig

        config = AlphaFoundryRuntimeConfig(
            mode="gate",
            observability_mode="debug_log",
        )
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            runtime_config=config,
        )
        assert isinstance(artifacts.search_cells, tuple)

    def test_l0_pipeline_logs_tf_fusion_stage_and_leaves_evidence_unchanged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """S2-05: [ALGO] tf_fusion log line fires with correct counts; evidence rows unaffected by logging."""
        import logging

        panel = _make_panel()
        aligned = _make_aligned()
        evidence_by_tf = {
            "4h": pd.DataFrame(
                {
                    "family": ["trend_ma"],
                    "variant": ["ema_12_72"],
                    "timeframe": ["4h"],
                    "recipe_id": ["r1"],
                    "reject_reasons": [""],
                    "mean_net_bps": [10.0],
                    "block_lcb_bps": [5.0],
                }
            ),
        }

        def _run() -> tuple[AlphaFoundryEvidenceRow, ...]:
            artifacts = run_alpha_foundry_l0_pipeline(
                panels=[panel],
                recipes={"r1": SAMPLE_RECIPE},
                aligned=aligned,
                cost_model=ExecutionCostModel(),
                cheap_gate_config=CheapGateConfig(),
                evidence_by_tf=evidence_by_tf,
            )
            return artifacts.evidence_rows

        with caplog.at_level(logging.DEBUG):
            evidence_rows = _run()

        assert "[ALGO] stage=tf_fusion" in caplog.text
        assert "n_evidence_rows_total=1" in caplog.text
        assert "n_fusion_groups=" in caplog.text
        assert "n_recipes_indexed=" in caplog.text

        # [LIMIT-08]: logging is pure observability — same call must produce
        # decision-field-identical evidence rows (created_at_ms is a wall-clock
        # timestamp, expected to differ between calls and irrelevant here).
        import dataclasses

        def _strip_timestamp(
            rows: tuple[AlphaFoundryEvidenceRow, ...],
        ) -> tuple[AlphaFoundryEvidenceRow, ...]:
            return tuple(dataclasses.replace(r, created_at_ms=0) for r in rows)

        baseline_rows = _run()
        assert _strip_timestamp(evidence_rows) == _strip_timestamp(baseline_rows)

    def test_l0_pipeline_produces_evidence_with_canonical_fields(self) -> None:
        """Gap 1 regression: evidence row must have capacity_score/regime_stability from canonical gate."""
        panel = _make_panel()
        aligned = _make_aligned()

        aligned_with_liq = AlignedMarketData(
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            open_2d=aligned.open_2d,
            high_2d=aligned.high_2d,
            low_2d=aligned.low_2d,
            close_2d=aligned.close_2d,
            volume_2d=aligned.volume_2d,
            funding_2d=aligned.funding_2d,
            active_mask=aligned.active_mask,
            warm_mask=aligned.warm_mask,
            entry_block_mask=aligned.entry_block_mask,
            kill_mask=aligned.kill_mask,
            execution_cost_bps_2d=np.full(aligned.close_2d.shape, 0.5, dtype=np.float64),
            adv_usdt_2d=np.full(aligned.close_2d.shape, 1_000_000_000.0, dtype=np.float64),
        )
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned_with_liq,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(min_events=10, min_effective_n=5.0,
                                              min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0,
                                              max_cost_drag_ratio=1.0, max_turnover_per_year=2000.0),
        )
        found_canonical = False
        for row in artifacts.evidence_rows:
            if row.gate_passed:
                assert row.regime_stability > 0.0, f"{row.recipe_id} regime_stability=0.0 but gate_passed"
                assert row.handoff_tier in {"seed", "candidate", "blocked"}
                found_canonical = True
        assert found_canonical, "no gate_passed rows with canonical fields"

    def test_evidence_handoff_tier_from_canonical_gate(self) -> None:
        """Gap 1 regression: handoff_tier must come from canonical gate, not cheap binary."""
        panel = _make_panel()
        aligned = _make_aligned()
        aligned_with_liq = AlignedMarketData(
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            open_2d=aligned.open_2d,
            high_2d=aligned.high_2d,
            low_2d=aligned.low_2d,
            close_2d=aligned.close_2d,
            volume_2d=aligned.volume_2d,
            funding_2d=aligned.funding_2d,
            active_mask=aligned.active_mask,
            warm_mask=aligned.warm_mask,
            entry_block_mask=aligned.entry_block_mask,
            kill_mask=aligned.kill_mask,
            execution_cost_bps_2d=np.full(aligned.close_2d.shape, 4.0, dtype=np.float64),
            adv_usdt_2d=np.full(aligned.close_2d.shape, 1_000_000.0, dtype=np.float64),
        )
        cfg = CheapGateConfig(min_events=10, min_effective_n=5.0,
                              min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0,
                              max_cost_drag_ratio=1.0, max_turnover_per_year=2000.0)
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned_with_liq,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=cfg,
        )
        found_seed = False
        for row in artifacts.evidence_rows:
            assert row.handoff_tier in {"seed", "candidate", "blocked"}, f"got {row.handoff_tier}"
            if row.handoff_tier == "seed":
                found_seed = True
        # At least some should be seed (weak rank IC etc.)
        assert found_seed, "no 'seed' handoff_tier found — canonical gate tier logic not wired"

    def test_diagnostic_flag_does_not_affect_l1_handoff(self) -> None:
        """[LIMIT-06] enable_failure_attribution must only append evidence_rows —
        passed_recipe_ids/handoff_decisions/stage_counts/bucket_results/
        cross_bucket_result must be byte-identical regardless of the flag."""
        from src.domain.futures.alpha_foundry.contracts import (
            AlphaFoundryRuntimeConfig,
            ConditionalCellGateConfig,
            ExecutionArmConfig,
        )

        panel = _make_panel()
        aligned = _make_aligned()
        cfg = CheapGateConfig(min_events=10, min_effective_n=5.0,
                              min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0,
                              max_cost_drag_ratio=1.0, max_turnover_per_year=2000.0)

        kwargs = {
            "panels": [panel],
            "recipes": {"r1": SAMPLE_RECIPE},
            "aligned": aligned,
            "cost_model": ExecutionCostModel(),
            "cheap_gate_config": cfg,
            "run_id": "isolation-test",
        }

        result_off = run_alpha_foundry_l0_pipeline(
            **kwargs,
            runtime_config=AlphaFoundryRuntimeConfig(enable_failure_attribution=False),
        )
        result_on = run_alpha_foundry_l0_pipeline(
            **kwargs,
            runtime_config=AlphaFoundryRuntimeConfig(
                enable_failure_attribution=True,
                enable_conditional_l0_cells=True,
                enable_execution_arms=True,
                conditional_cell=ConditionalCellGateConfig(enabled=True),
                execution_arm=ExecutionArmConfig(enabled=True, styles=("taker_now", "hybrid")),
            ),
        )

        assert result_off.passed_recipe_ids == result_on.passed_recipe_ids
        assert result_off.stage_counts == result_on.stage_counts
        assert result_off.handoff_decisions == result_on.handoff_decisions
        assert result_off.bucket_results == result_on.bucket_results
        assert result_off.cross_bucket_result == result_on.cross_bucket_result
        assert len(result_on.evidence_rows) >= len(result_off.evidence_rows)


class TestBuildPosteriorFromL1FoldRows:
    def test_empty_raw_rows_returns_empty(self) -> None:
        config = PosteriorGateConfig()
        result = build_posterior_from_l1_fold_rows(
            raw_rows=pd.DataFrame(
                columns=[
                    "symbol",
                    "recipe_id",
                    "family",
                    "timeframe",
                    "activation_context",
                    "net_bps",
                    "fold_id",
                    "effective_weight",
                ]
            ),
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) == 0

    def test_rejects_missing_net_bps_column(self) -> None:
        config = PosteriorGateConfig()
        with pytest.raises(ValueError, match="raw_rows missing required columns"):
            build_posterior_from_l1_fold_rows(
                raw_rows=pd.DataFrame({"symbol": ["SYM0USDT"]}),
                cost_model=ExecutionCostModel(),
                config=config,
            )

    def test_rejects_missing_symbol_column(self) -> None:
        config = PosteriorGateConfig()
        with pytest.raises(ValueError, match="raw_rows missing required columns"):
            build_posterior_from_l1_fold_rows(
                raw_rows=pd.DataFrame({"net_bps": [1.0]}),
                cost_model=ExecutionCostModel(),
                config=config,
            )

    def test_real_fold_rows_produces_posterior(self) -> None:
        config = PosteriorGateConfig()
        raw_rows = pd.DataFrame(
            [
                {
                    "symbol": "SYM0USDT",
                    "recipe_id": "trend_ma__ema_12_72__4h",
                    "family": "trend_ma",
                    "timeframe": "4h",
                    "activation_context": "pooled",
                    "net_bps": 4.2,
                    "fold_id": 0,
                    "effective_weight": 1.0,
                },
                {
                    "symbol": "SYM0USDT",
                    "recipe_id": "trend_ma__ema_12_72__4h",
                    "family": "trend_ma",
                    "timeframe": "4h",
                    "activation_context": "pooled",
                    "net_bps": 3.6,
                    "fold_id": 1,
                    "effective_weight": 1.0,
                },
            ]
        )
        result = build_posterior_from_l1_fold_rows(
            raw_rows=raw_rows,
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) > 0
        assert all(hasattr(r, "posterior_mu_bps") for r in result)


class TestL2SleevesFromPosterior:
    def test_empty_posterior_returns_empty(self) -> None:
        config = L2PosteriorPolicyConfig()
        result = build_l2_sleeves_from_posterior(
            posterior=(),
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) == 0

    def test_real_posterior_produces_l2_sleeves(self) -> None:
        config = L2PosteriorPolicyConfig()
        raw_rows = pd.DataFrame(
            [
                {
                    "symbol": "SYM0USDT",
                    "recipe_id": "trend_ma__ema_12_72__4h",
                    "family": "trend_ma",
                    "timeframe": "4h",
                    "activation_context": "pooled",
                    "net_bps": 4.2,
                    "fold_id": 0,
                    "effective_weight": 1.0,
                },
            ]
        )
        posterior = build_posterior_from_l1_fold_rows(
            raw_rows=raw_rows,
            cost_model=ExecutionCostModel(),
            config=PosteriorGateConfig(),
        )
        result = build_l2_sleeves_from_posterior(
            posterior=posterior,
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) > 0


class TestBuildL1VerificationUnits:
    def test_build_units(self) -> None:
        from src.domain.futures.alpha_foundry.budget import build_l1_verification_units

        ev = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=80.0,
            mean_net_bps=5.0,
            nw_tstat=2.0,
            block_lcb_bps=2.0,
            rank_ic=0.05,
            cost_drag_ratio=0.3,
            turnover_per_year=50.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=1.5,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        units = build_l1_verification_units(
            evidences=[ev],
            recipes={"r1": recipe},
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        assert len(units) == 1
        assert units[0].recipe_id == "r1"
        assert units[0].allocated_fold_budget == 3

    def test_build_units_skips_unknown_recipe(self) -> None:
        from src.domain.futures.alpha_foundry.budget import build_l1_verification_units

        ev = CheapGateEvidence(
            recipe_id="unknown",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=80.0,
            mean_net_bps=5.0,
            nw_tstat=2.0,
            block_lcb_bps=2.0,
            rank_ic=0.05,
            cost_drag_ratio=0.3,
            turnover_per_year=50.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=1.5,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        units = build_l1_verification_units(
            evidences=[ev],
            recipes={},
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        assert len(units) == 0

    def test_build_units_raises_on_budget_violation(self) -> None:
        from src.domain.futures.alpha_foundry.budget import build_l1_verification_units

        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        evs = [
            CheapGateEvidence(
                recipe_id="r1",
                timeframe="4h",
                symbol_scope="symbol",
                n_events=100,
                effective_n=80.0,
                mean_net_bps=5.0,
                nw_tstat=2.0,
                block_lcb_bps=2.0,
                rank_ic=0.05,
                cost_drag_ratio=0.3,
                turnover_per_year=50.0,
                novelty_corr_max=0.0,
                incremental_rank_ic=0.0,
                compute_cost_score=0.0,
                gate_passed=True,
                reject_reasons=(),
                bootstrap_lcb_bps=1.5,
                bootstrap_agree=True,
                mean_gross_bps=0.0,
                mean_cost_bps=0.0,
            ),
            CheapGateEvidence(
                recipe_id="r1",
                timeframe="4h",
                symbol_scope="symbol",
                n_events=100,
                effective_n=80.0,
                mean_net_bps=4.0,
                nw_tstat=2.0,
                block_lcb_bps=1.0,
                rank_ic=0.05,
                cost_drag_ratio=0.3,
                turnover_per_year=50.0,
                novelty_corr_max=0.0,
                incremental_rank_ic=0.0,
                compute_cost_score=0.0,
                gate_passed=True,
                reject_reasons=(),
                bootstrap_lcb_bps=1.0,
                bootstrap_agree=True,
                mean_gross_bps=0.0,
                mean_cost_bps=0.0,
            ),
        ]
        with pytest.raises(ValueError, match="budget violated"):
            build_l1_verification_units(
                evidences=evs,
                recipes={"r1": recipe},
                symbols=("BTCUSDT",),
                top_k_per_family_tf=1,  # only 1 allowed, but 2 same-bucket
                initial_fold_budget=3,
            )


# ── Mock boilerplate for handoff tests ─────────────────────────────


def _make_aligned_handoff(t: int = 128, n: int = 2) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    close = np.linspace(100.0, 120.0, t, dtype=np.float64).reshape(-1, 1)
    close = np.repeat(close, n, axis=1)
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=tuple(f"SYM{i}USDT" for i in range(n)),
        open_2d=close.copy(),
        high_2d=close * 1.001,
        low_2d=close * 0.999,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.full((t, n), 0.00005, dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros((t, n), dtype=np.bool_),
        kill_mask=np.zeros((t, n), dtype=np.bool_),
    )


def _make_panel_handoff(
    recipe_id: str,
    family: str = "trend_ma",
    variant: str = "ema_12_72_4h",
) -> CandidateSignalPanel:
    aligned = _make_aligned_handoff(t=128, n=2)
    t, n = aligned.close_2d.shape
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=side.astype(np.float64),
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": recipe_id},
        archetype="trend",
    )


def _make_recipe_handoff(recipe_id: str, family: str = "trend_ma", variant: str = "ema_12_72") -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id=recipe_id,
        family=family,
        variant=variant,
        timeframe="4h",
        archetype="trend",
        indicator_params={"fast": 12, "slow": 72},
        side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",),
        causal_lag_bars=1,
        max_turnover_per_year=365.0,
    )


def _make_candidate_handoff(
    recipe_id: str,
    *,
    priority: float,
    tier: str = "candidate",
    blocked: bool = False,
    corroboration_tier: str = "insufficient_coverage",
    timeframe: str = "4h",
) -> L0SignalCandidate:
    hard_reject_reasons = ("deep_negative_lcb",) if blocked else ()
    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family="trend_ma",
        variant=recipe_id,
        recipe_id=recipe_id,
        archetype="trend",
        source="catalog_exact",
        n_events=100,
        effective_n=100.0,
        mean_net_bps=priority + 1.0,
        block_lcb_bps=priority,
        nw_tstat=2.0,
        bootstrap_lcb_bps=priority,
        bootstrap_agree=True,
        cost_drag_ratio=0.3,
        turnover_per_year=100.0,
        max_abs_corr_in_bucket=0.0,
        tf_coverage_count=1,
        sign_agreement_ratio=1.0,
        corroboration_tier=corroboration_tier,
        discovery_tier="blocked" if blocked else tier,
        l1_priority_score=priority,
        l1_budget_units=0,
        hard_reject_reasons=hard_reject_reasons,
        soft_flags=(),
    )


def _make_bucket_handoff(bucket_key: tuple[str, str], recipe_ids: tuple[str, ...]) -> DiversitySelectionResult:
    return DiversitySelectionResult(
        bucket_key=bucket_key,
        ranked_recipe_ids=recipe_ids,
        selected_recipe_ids=recipe_ids,
        redundant_recipe_ids=(),
        redundant_reason_by_id={},
        bucket_corr=np.eye(len(recipe_ids), dtype=np.float64),
        bucket_eff_test_count=float(len(recipe_ids)),
    )


class TestBuildL0HandoffDecisions:
    """Scenario S1-1, S2-2, S2-4, S2-5, S2-6, S3-2"""

    def test_hands_off_only_top_n_by_priority_in_bucket(self) -> None:
        recipes = {
            "r_a": _make_recipe_handoff("r_a"),
            "r_b": _make_recipe_handoff("r_b"),
            "r_c": _make_recipe_handoff("r_c"),
        }
        candidates = [
            _make_candidate_handoff("r_a", priority=30.0),
            _make_candidate_handoff("r_b", priority=20.0),
            _make_candidate_handoff("r_c", priority=10.0),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ("r_a", "r_b", "r_c"))
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r_a", "r_b", "r_c"),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(3, dtype=np.float64),
            global_eff_test_count=3.0,
        )
        panel_by_rid = {c.recipe_id: _make_panel_handoff(c.recipe_id) for c in candidates}
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={bk: 2},
            panel_by_recipe_id=panel_by_rid,
        )
        selected = [d for d in decisions if d.selected_for_l1]
        assert len(selected) == 2
        assert selected[0].recipe_id == "r_a"
        assert selected[1].recipe_id == "r_b"
        r_c_decision = next(d for d in decisions if d.recipe_id == "r_c")
        assert r_c_decision.exclusion_reason == "budget_exhausted"

    def test_single_bucket_single_slot_does_not_promote_all(self) -> None:
        recipes = {
            "r_a": _make_recipe_handoff("r_a"),
            "r_b": _make_recipe_handoff("r_b"),
            "r_c": _make_recipe_handoff("r_c"),
        }
        candidates = [
            _make_candidate_handoff("r_a", priority=30.0),
            _make_candidate_handoff("r_b", priority=20.0),
            _make_candidate_handoff("r_c", priority=10.0),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ("r_a", "r_b", "r_c"))
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r_a", "r_b", "r_c"),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(3, dtype=np.float64),
            global_eff_test_count=3.0,
        )
        panel_by_rid = {c.recipe_id: _make_panel_handoff(c.recipe_id) for c in candidates}
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={bk: 1},
            panel_by_recipe_id=panel_by_rid,
        )
        selected = [d for d in decisions if d.selected_for_l1]
        assert len(selected) == 1
        assert selected[0].budget_units == 1

    def test_insufficient_coverage_remains_viable(self) -> None:
        recipes = {"r_a": _make_recipe_handoff("r_a")}
        candidates = [
            _make_candidate_handoff("r_a", priority=10.0, corroboration_tier="insufficient_coverage"),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ("r_a",))
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r_a",),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(1, dtype=np.float64),
            global_eff_test_count=1.0,
        )
        panel_by_rid = {c.recipe_id: _make_panel_handoff(c.recipe_id) for c in candidates}
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={bk: 1},
            panel_by_recipe_id=panel_by_rid,
        )
        d = decisions[0]
        assert d.eligible_for_diversity is True
        assert d.eligible_for_budget is True
        assert d.selected_for_l1 is True

    def test_tf_contradicted_candidate_is_fail_closed(self) -> None:
        recipes = {"r_a": _make_recipe_handoff("r_a")}
        candidates = [
            _make_candidate_handoff("r_a", priority=10.0, corroboration_tier="contradicted", blocked=True),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ())
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=(),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.empty((0, 0), dtype=np.float64),
            global_eff_test_count=0.0,
        )
        panel_by_rid = {}
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={},
            panel_by_recipe_id=panel_by_rid,
        )
        assert len(decisions) == 1
        assert decisions[0].selected_for_l1 is False
        assert decisions[0].exclusion_reason == "hard_reject"

    def test_empty_viable_pool_gives_valid_empty_decisions(self) -> None:
        decisions = build_l0_handoff_decisions(
            candidates=[],
            recipes={},
            bucket_results=[],
            cross_result=None,
            allocated_slots_by_bucket={},
            panel_by_recipe_id={},
        )
        assert len(decisions) == 0

    def test_marks_missing_panel_as_excluded(self) -> None:
        recipes = {"r_a": _make_recipe_handoff("r_a")}
        candidates = [
            _make_candidate_handoff("r_a", priority=10.0),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ("r_a",))
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r_a",),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(1, dtype=np.float64),
            global_eff_test_count=1.0,
        )
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={bk: 1},
            panel_by_recipe_id={},
        )
        d = decisions[0]
        assert d.eligible_for_diversity is True
        assert d.eligible_for_budget is False
        assert d.selected_for_l1 is False
        assert d.exclusion_reason == "missing_panel"


class TestL0HandoffInvariants:
    """Scenario S2-3: handoff invariants match across artifacts."""

    def test_handoff_invariants_match_across_artifacts(self) -> None:
        recipes = {
            "r_a": _make_recipe_handoff("r_a"),
            "r_b": _make_recipe_handoff("r_b"),
        }
        candidates = [
            _make_candidate_handoff("r_a", priority=30.0),
            _make_candidate_handoff("r_b", priority=20.0),
        ]
        bk = ("trend_ma", "4h")
        bucket_result = _make_bucket_handoff(bk, ("r_a", "r_b"))
        cross_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r_a", "r_b"),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(2, dtype=np.float64),
            global_eff_test_count=2.0,
        )
        panel_by_rid = {c.recipe_id: _make_panel_handoff(c.recipe_id) for c in candidates}
        decisions = build_l0_handoff_decisions(
            candidates=candidates,
            recipes=recipes,
            bucket_results=[bucket_result],
            cross_result=cross_result,
            allocated_slots_by_bucket={bk: 2},
            panel_by_recipe_id=panel_by_rid,
        )
        passed_recipe_ids_set = {d.recipe_id for d in decisions if d.selected_for_l1}
        budgeted_set = {d.recipe_id for d in decisions if d.budget_units > 0}
        assert passed_recipe_ids_set == budgeted_set

        evidence_rows = []
        for c in candidates:
            d = next(x for x in decisions if x.recipe_id == c.recipe_id)
            evidence_rows.append(
                AlphaFoundryEvidenceRow(
                    run_id="test",
                    timeframe="4h",
                    family="trend_ma",
                    variant=c.variant,
                    recipe_id=c.recipe_id,
                    archetype="trend",
                    n_events=100,
                    effective_n=100.0,
                    mean_gross_bps=45.0,
                    mean_cost_bps=14.0,
                    mean_net_bps=31.0,
                    gross_lcb_bps=30.0,
                    net_lcb_bps=30.0,
                    nw_tstat=2.0,
                    rank_ic=0.05,
                    rank_ic_tstat=1.5,
                    cost_drag_ratio=0.3,
                    turnover_per_year=100.0,
                    novelty_corr_max=0.0,
                    incremental_rank_ic=0.0,
                    compute_cost_score=0.0,
                    event_hit_rate=0.6,
                    payoff_skew=1.5,
                    xs_spread_lcb_bps=None,
                    liquidity_cost_stress_bps=0.0,
                    bootstrap_lcb_bps=30.0,
                    bootstrap_agree=True,
                    gate_passed=True,
                    handoff_tier="candidate",
                    selected_for_l1=d.selected_for_l1,
                    reject_reasons="",
                    soft_flags="",
                    bucket_key="trend_ma:4h",
                    bucket_rank=0,
                    redundant_with="",
                    bucket_eff_test_count=2.0,
                    global_eff_test_count=2.0,
                    l1_priority_score=0.0,
                    l1_budget_units=0,
                    tf_coverage_count=0,
                    sign_agreement_ratio=0.0,
                    corroboration_tier="",
                    stage_label="",
                    created_at_ms=1000,
                )
            )

        evidence_selected_set = {r.recipe_id for r in evidence_rows if r.selected_for_l1}
        assert passed_recipe_ids_set == evidence_selected_set


class TestL0PipelineHandoff:
    """End-to-end pipeline tests using the spec's mock boilerplate."""

    def test_l0_pipeline_hands_off_only_budgeted_viable_candidates(self) -> None:
        aligned = _make_aligned_handoff(t=128, n=2)
        recipes = {
            "r_a": _make_recipe_handoff("r_a"),
            "r_b": _make_recipe_handoff("r_b"),
            "r_c": _make_recipe_handoff("r_c"),
        }
        panels = [_make_panel_handoff(rid) for rid in ("r_a", "r_b", "r_c")]
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=panels,
            recipes=recipes,
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(min_events=10),
            run_id="test",
            top_k_per_family_tf=5,
            min_conviction_lcb_bps=0.0,
            total_l1_verification_budget=30,
        )
        assert len(artifacts.passed_recipe_ids) <= 30
        for c in artifacts.candidates:
            if c.recipe_id in artifacts.passed_recipe_ids:
                assert c.l1_budget_units > 0
            else:
                if c.discovery_tier in {"seed", "candidate", "verified"}:
                    pass  # excluded by budget/diversity — acceptable

    def test_empty_panels_empty_handoff(self) -> None:
        aligned = _make_aligned_handoff(t=128, n=2)
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            run_id="test",
        )
        assert artifacts.passed_recipe_ids == ()
        assert len(artifacts.handoff_decisions) == 0
        assert artifacts.stage_counts.viable_candidates == 0
        assert artifacts.stage_counts.l1_queued == 0

    def test_blocked_handoff_tier_not_selected_for_l1(self) -> None:
        """Gap 3: blocked handoff_tier must not leak to selected_for_l1."""
        aligned = _make_aligned_handoff(t=128, n=2)
        recipes = {"r_blocked": _make_recipe_handoff("r_blocked")}
        panels = [_make_panel_handoff("r_blocked")]
        cfg = CheapGateConfig(min_events=9999)
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=panels,
            recipes=recipes,
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=cfg,
            run_id="test",
        )
        for row in artifacts.evidence_rows:
            if row.handoff_tier == "blocked":
                assert not row.selected_for_l1, f"{row.recipe_id} blocked but selected_for_l1"


# ─── build_alpha_foundry_evidence_row tests ────────────────────────────


def _make_cheap_evidence(
    recipe_id: str = "test:r1:4h:abc",
    gate_passed: bool = True,
) -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id=recipe_id,
        timeframe="4h",
        symbol_scope="symbol",
        n_events=64,
        effective_n=64.0,
        mean_net_bps=19.0,
        nw_tstat=1.40,
        block_lcb_bps=2.0,
        rank_ic=0.05,
        cost_drag_ratio=0.32,
        turnover_per_year=48.0,
        novelty_corr_max=0.10,
        incremental_rank_ic=0.02,
        compute_cost_score=0.0,
        bootstrap_lcb_bps=1.0,
        bootstrap_agree=True,
        gate_passed=gate_passed,
        reject_reasons=(),
        mean_gross_bps=28.0,
        mean_cost_bps=9.0,
    )


def _make_canonical_evidence() -> AlphaGateEvidence:
    return AlphaGateEvidence(
        schema_version="unified",
        run_id="run-1",
        timeframe="4h",
        family="liquidity_participation_breakout",
        variant="lpb_40",
        recipe_id="test:r1:4h:abc",
        archetype="trend",
        symbol_scope="symbol",
        n_events=64,
        effective_n=64.0,
        mean_gross_bps=28.0,
        mean_cost_bps=9.0,
        mean_net_bps=19.0,
        gross_lcb_bps=11.0,
        net_lcb_bps=2.0,
        nw_tstat=1.40,
        rank_ic=0.05,
        rank_ic_tstat=2.10,
        cost_drag_ratio=0.32,
        turnover_per_year=48.0,
        novelty_corr_max=0.10,
        incremental_rank_ic=0.02,
        compute_cost_score=0.0,
        event_hit_rate=0.56,
        payoff_skew=0.30,
        xs_spread_lcb_bps=None,
        liquidity_cost_stress_bps=1.5,
        bootstrap_lcb_bps=1.0,
        bootstrap_agree=True,
        gate_passed=True,
        handoff_tier="candidate",
        selected_for_l1=False,
        reject_reasons=(),
        soft_flags=(),
        capacity_score=0.90,
        regime_stability=0.72,
        tf_corroboration=0.50,
        entry_mode="sparse",
    )


class TestBuildAlphaFoundryEvidenceRow:
    """S1-03: Terminal row uses canonical values when available."""

    def test_s1_03_terminal_evidence_uses_canonical_gate_values(self) -> None:
        cheap = _make_cheap_evidence()
        canonical = _make_canonical_evidence()
        row = build_alpha_foundry_evidence_row(
            cheap_evidence=cheap,
            canonical_evidence=canonical,
            candidate=None,
            handoff_decision=None,
            bucket_result=None,
            cross_bucket_result=None,
            created_at_ms=1000,
            source="test",
        )
        assert row.net_lcb_bps == canonical.net_lcb_bps
        assert row.gate_passed == canonical.gate_passed
        assert row.handoff_tier == canonical.handoff_tier
        assert row.mean_net_bps == canonical.mean_net_bps
        assert np.isfinite(row.mean_net_bps)

    def test_s3_03_missing_canonical_evidence_raises(self) -> None:
        cheap = _make_cheap_evidence()
        with pytest.raises(RuntimeError, match="missing canonical evidence"):
            build_alpha_foundry_evidence_row(
                cheap_evidence=cheap,
                canonical_evidence=None,
                candidate=None,
                handoff_decision=None,
                bucket_result=None,
                cross_bucket_result=None,
                created_at_ms=1000,
                source="test",
            )

    def test_s2_07_cheap_pass_canonical_block_reports_blocked(self) -> None:
        cheap = _make_cheap_evidence(gate_passed=True)
        canonical = _make_canonical_evidence()
        canonical = canonical.__class__(
            schema_version=canonical.schema_version,
            run_id=canonical.run_id,
            timeframe=canonical.timeframe,
            family=canonical.family,
            variant=canonical.variant,
            recipe_id=canonical.recipe_id,
            archetype=canonical.archetype,
            symbol_scope=canonical.symbol_scope,
            n_events=canonical.n_events,
            effective_n=canonical.effective_n,
            mean_gross_bps=canonical.mean_gross_bps,
            mean_cost_bps=canonical.mean_cost_bps,
            mean_net_bps=canonical.mean_net_bps,
            gross_lcb_bps=canonical.gross_lcb_bps,
            net_lcb_bps=canonical.net_lcb_bps,
            nw_tstat=canonical.nw_tstat,
            rank_ic=canonical.rank_ic,
            rank_ic_tstat=canonical.rank_ic_tstat,
            cost_drag_ratio=canonical.cost_drag_ratio,
            turnover_per_year=canonical.turnover_per_year,
            novelty_corr_max=canonical.novelty_corr_max,
            incremental_rank_ic=canonical.incremental_rank_ic,
            compute_cost_score=canonical.compute_cost_score,
            event_hit_rate=canonical.event_hit_rate,
            payoff_skew=canonical.payoff_skew,
            xs_spread_lcb_bps=canonical.xs_spread_lcb_bps,
            liquidity_cost_stress_bps=canonical.liquidity_cost_stress_bps,
            bootstrap_lcb_bps=canonical.bootstrap_lcb_bps,
            bootstrap_agree=False,
            gate_passed=False,
            handoff_tier="blocked",
            selected_for_l1=canonical.selected_for_l1,
            reject_reasons=("bootstrap_disagree",),
            soft_flags=canonical.soft_flags,
            capacity_score=canonical.capacity_score,
            regime_stability=canonical.regime_stability,
            tf_corroboration=canonical.tf_corroboration,
            entry_mode=canonical.entry_mode,
        )
        row = build_alpha_foundry_evidence_row(
            cheap_evidence=cheap,
            canonical_evidence=canonical,
            candidate=None,
            handoff_decision=None,
            bucket_result=None,
            cross_bucket_result=None,
            created_at_ms=1000,
            source="test",
        )
        assert row.gate_passed is False
        assert row.handoff_tier == "blocked"
        assert row.bootstrap_agree is False
        assert "bootstrap_disagree" in row.reject_reasons
