from __future__ import annotations

import concurrent.futures
from collections.abc import MutableMapping
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    _L0_TF_INPUT_CACHE,
    AlphaFoundryL0Result,
    _bind_panels_to_recipe_ids,
    _prime_l0_tf_input_cache,
    _run_l0_gate_worker,
    bind_panels_to_alpha_recipes,
    build_cheap_gate_evidence_frame,
    build_cheap_gate_evidence_frame_from_evidences,
    run_alpha_foundry_l0_gate,
    run_alpha_foundry_l0_gate_multi_tf,
)
from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaRecipe,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def make_aligned_for_tf(
    *, tf_hours: int, bars: int = 200, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(tf_hours * bars, "h"),
        np.timedelta64(tf_hours, "h"),
        dtype="datetime64[ns]",
    )
    t, n = dt.shape[0], len(symbols)
    close = 100.0 * np.exp(0.001 * np.arange(t, dtype=np.float64))[:, None] * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
        volume_2d=np.full((t, n), 1000.0), funding_2d=np.full((t, n), 0.00005),
        active_mask=mask, warm_mask=mask,
        entry_block_mask=np.zeros((t, n), dtype=np.bool_), kill_mask=np.zeros((t, n), dtype=np.bool_),
    )


def make_panel_for_tf(
    *,
    tf: str,
    family: str = "fam",
    variant: str = "var",
    bars: int = 200,
    n_symbols: int = 2,
    sign: float = 1.0,
) -> CandidateSignalPanel:
    score = np.full((bars, n_symbols), 0.0)
    side = np.zeros((bars, n_symbols), dtype=np.int8)
    for start in range(10, bars, 20):
        score[start, :] = sign * 0.8
        side[start:start + 3, :] = int(sign)
    return CandidateSignalPanel(
        family=family, variant=f"{variant}_{tf}",
        params={"lookback": 20}, datetimes=np.arange(bars, dtype=np.int64),
        symbols=("BTCUSDT", "ETHUSDT")[:n_symbols],
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=np.ones((bars, n_symbols), dtype=np.bool_),
        metadata={"recipe_id": f"{family}:{variant}:{tf}"},
    )


def make_recipe_for_tf(*, tf: str, family: str = "fam", variant: str = "var") -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id=f"{family}:{variant}:{tf}", family=family, variant=f"{variant}_{tf}",
        timeframe=tf, archetype="trend", indicator_params={"lookback": 20},
        side_rule_id="trend_follow", exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1, max_turnover_per_year=365.0,
    )


def make_gate_config() -> AlphaGateConfig:
    return AlphaGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0, min_nw_tstat=0.0,
        max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0, min_candidate_rank_ic_tstat=0.0,
        # archetype_event_floors now takes precedence over min_events
        # (resolve_family_timeframe_gate_policy wired into the real gate) — clear it
        # so this fixture's permissive min_events=1 actually applies.
        archetype_event_floors={},
    )


RUNTIME_CONFIG = AlphaFoundryRuntimeConfig(
    mode="gate",
    cheap_gate=make_gate_config(),
    max_recipes_per_family=10,
)


def _build_panels_recipes_aligned(
    tfs: tuple[str, ...],
    *,
    sign_by_tf: dict[str, float] | None = None,
) -> tuple[
    dict[str, list[CandidateSignalPanel]],
    dict[str, MutableMapping[str, AlphaRecipe]],
    dict[str, AlignedMarketData],
    dict[str, list[Any]],
]:
    """Build panels/recipes/aligned per TF AND real bindings via
    bind_panels_to_alpha_recipes() (never empty bindings — an empty binding
    list makes run_alpha_foundry_l0_gate() bind zero panels, which silently
    produces empty evidence regardless of the gate logic under test)."""
    panels_by_tf: dict[str, list[CandidateSignalPanel]] = {}
    recipes_by_tf: dict[str, MutableMapping[str, AlphaRecipe]] = {}
    aligned_by_tf: dict[str, AlignedMarketData] = {}
    bindings_by_tf: dict[str, list[Any]] = {}
    for tf in tfs:
        tf_hours = int(tf.replace("h", ""))
        sign = sign_by_tf[tf] if sign_by_tf else 1.0
        aligned = make_aligned_for_tf(tf_hours=tf_hours, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200, sign=sign)
        recipe = make_recipe_for_tf(tf=tf)
        panels_by_tf[tf] = [panel]
        recipes_by_tf[tf] = {recipe.recipe_id: recipe}
        aligned_by_tf[tf] = aligned
        bindings_by_tf[tf] = list(
            bind_panels_to_alpha_recipes(
                panels=[panel],
                recipes=recipes_by_tf[tf],
                timeframe=tf,
                max_recipes_per_family=10,
                include_families=(),
                exclude_families=(),
                enable_synthetic_recipes=True,
            )
        )
        assert bindings_by_tf[tf], f"panel for tf={tf} failed to bind to its own recipe"
    return panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf


class TestBuildCheapGateEvidenceFrame:
    """S1-3: build_cheap_gate_evidence_frame schema."""

    def test_returns_required_schema(self) -> None:
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf="4h", bars=200)
        recipe = make_recipe_for_tf(tf="4h")
        df = build_cheap_gate_evidence_frame(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        _cols = {"family", "variant", "timeframe", "recipe_id", "reject_reasons", "mean_net_bps", "block_lcb_bps"}
        assert _cols.issubset(set(df.columns)), f"missing cols: {_cols - set(df.columns)}"
        assert len(df) == 1
        assert df["family"].iloc[0] == "fam"
        assert df["variant"].iloc[0] == "var_4h"
        assert df["timeframe"].iloc[0] == "4h"

    def test_empty_panels_returns_empty_df(self) -> None:
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        df = build_cheap_gate_evidence_frame(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        assert df.empty
        _cols = {"family", "variant", "timeframe", "recipe_id", "reject_reasons", "mean_net_bps", "block_lcb_bps"}
        assert _cols.issubset(set(df.columns))

    def test_unknown_recipe_id_excluded(self) -> None:
        """E2-4: recipe_id without matching recipe is excluded."""
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf="4h", bars=200, family="unknown", variant="unk")
        recipe = make_recipe_for_tf(tf="4h")
        df = build_cheap_gate_evidence_frame(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        assert len(df) == 0


class TestRunAlphaFoundryL0GateMultiTf:
    """S1-4: multi-TF gate basic behavior."""

    def test_returns_all_requested_tf_keys(self) -> None:
        tfs = ("4h", "6h", "8h", "12h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_s1_4",
        )
        assert set(results) == set(tfs)
        for tf in tfs:
            assert isinstance(results[tf], AlphaFoundryL0Result)
            assert len(results[tf].evidence_rows) == 1, f"tf={tf} should bind and gate exactly 1 recipe"

    def test_raises_on_key_mismatch(self) -> None:
        """X3-1: panels_by_tf and aligned_by_tf key mismatch."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)
        aligned_by_tf.pop("6h")

        with pytest.raises(ValueError, match="aligned_by_tf missing timeframe"):
            run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf,
                bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf,
                aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(),
                runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_x3_1",
            )

    def test_empty_tf_panels_returns_empty_result(self) -> None:
        """E2-1: specific TF with empty panels produces empty result, others unaffected."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)
        panels_by_tf["6h"] = []
        bindings_by_tf["6h"] = []

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_e2_1",
        )
        assert "4h" in results
        assert "6h" in results
        assert len(results["6h"].panels_for_l1) == 0
        assert len(results["4h"].evidence_rows) == 1, "TF unaffected by sibling TF's empty panels"

    def test_cross_tf_corroborated_reaches_real_tf_corroboration(self) -> None:
        """S1-5: 4 TF all same sign positive -> corroboration -> tf_corroboration/corroboration_tier
        reflect real cross-TF agreement (previously always 0.0/"insufficient_coverage" due to the
        2-tuple vs 3-tuple tf_fusion_index key bug)."""
        tfs = ("4h", "6h", "8h", "12h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(
            tfs, sign_by_tf=dict.fromkeys(tfs, 1.0)
        )

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_s1_5",
        )
        for tf in tfs:
            rows = results[tf].evidence_rows
            assert len(rows) == 1, f"tf={tf} expected exactly 1 evidence row"
            row = rows[0]
            assert row.tf_corroboration == pytest.approx(1.0), (
                f"tf={tf}: same-sign panels across all 4 TFs must yield full corroboration, got {row.tf_corroboration}"
            )
            assert row.corroboration_tier == "corroborated", (
                f"tf={tf}: expected 'corroborated', got {row.corroboration_tier!r}"
            )
            assert row.handoff_tier != "blocked", f"tf={tf}: corroborated candidate must not be blocked"

    def test_mixed_sign_gives_contradicted_tier(self) -> None:
        """E2-3: 2 positive, 2 negative -> each TF's 3 siblings agree only 1/3 of the
        time (sign_agreement_ratio=1/3 <= 0.50 contradiction threshold) -> "contradicted",
        which forces tf_corroboration=0.0 (compute_tf_corroboration hard-zeros contradicted)."""
        tfs = ("4h", "6h", "8h", "12h")
        sign_by_tf = {tf: (1.0 if i < 2 else -1.0) for i, tf in enumerate(tfs)}
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(
            tfs, sign_by_tf=sign_by_tf
        )

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_e2_3",
        )
        for tf in tfs:
            rows = results[tf].evidence_rows
            assert len(rows) == 1, f"tf={tf} expected exactly 1 evidence row"
            row = rows[0]
            assert row.sign_agreement_ratio == pytest.approx(1.0 / 3.0), (
                f"tf={tf}: 1-of-3 sibling sign agreement expected, got {row.sign_agreement_ratio}"
            )
            assert row.corroboration_tier == "contradicted", (
                f"tf={tf}: expected 'contradicted', got {row.corroboration_tier!r}"
            )
            assert row.tf_corroboration == pytest.approx(0.0), (
                f"tf={tf}: contradicted tier must force tf_corroboration=0.0, got {row.tf_corroboration}"
            )


class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Test-friendly replacement for ProcessPoolExecutor that runs in-thread
    (no fork). Accepts and ignores `mp_context` to match ProcessPoolExecutor API."""

    def __init__(self, max_workers: int | None = None, mp_context: object = None, **kwargs: object) -> None:
        super().__init__(max_workers=max_workers, **kwargs)  # type: ignore[call-overload]


class TestParallelMaxWorkersValidation:
    """[LIMIT-03] parallel_max_workers must be in [1, 4]."""

    def test_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="parallel_max_workers"):
            run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf={}, bindings_by_tf={}, recipes_by_tf={}, aligned_by_tf={},
                cost_model=None, runtime_config=None, run_id_prefix="test",
                parallel_max_workers=5,
            )

    def test_below_min_raises(self) -> None:
        with pytest.raises(ValueError, match="parallel_max_workers"):
            run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf={}, bindings_by_tf={}, recipes_by_tf={}, aligned_by_tf={},
                cost_model=None, runtime_config=None, run_id_prefix="test",
                parallel_max_workers=0,
            )


class TestSequentialAndParallelIdentical:
    """[LIMIT-06] Sequential and parallel Phase-3 must produce byte-identical results."""

    def test_default_parallel_max_workers_1_matches_original(self) -> None:
        """parallel_max_workers=1 (default) == original signature (regression)."""
        tfs = ("4h", "6h", "8h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        base = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_seq_default",
        )
        explicit = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_seq_explicit", parallel_max_workers=1,
        )
        for tf in tfs:
            base_report = base[tf].report
            explicit_report = explicit[tf].report
            assert base_report is not None
            assert explicit_report is not None
            assert base_report.n_passed == explicit_report.n_passed
            assert base_report.reject_reason_counts == explicit_report.reject_reason_counts
            assert len(base[tf].panels_for_l1) == len(explicit[tf].panels_for_l1)

    def test_parallel_matches_sequential_exact(self) -> None:
        """parallel_max_workers=3 produces value-identical results to sequential."""
        tfs = ("4h", "6h", "8h", "12h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        seq = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_seq",
        )
        with patch(
            "src.domain.futures.alpha_foundry.bridge_helpers.ProcessPoolExecutor",
            new=SafeThreadPoolExecutor,
        ):
            par = run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_par", parallel_max_workers=3,
            )

        assert set(seq.keys()) == set(par.keys())
        for tf in tfs:
            seq_report = seq[tf].report
            par_report = par[tf].report
            assert seq_report is not None
            assert par_report is not None
            assert seq_report.n_passed == par_report.n_passed, f"tf={tf} n_passed mismatch"
            assert seq_report.reject_reason_counts == par_report.reject_reason_counts, (
                f"tf={tf} reject_reason_counts mismatch"
            )
            assert len(seq[tf].panels_for_l1) == len(par[tf].panels_for_l1), f"tf={tf} panels_for_l1 len mismatch"


class TestPrimeCacheAndWorker:
    """[LIMIT-02] _prime_l0_tf_input_cache + _run_l0_gate_worker."""

    def test_worker_result_matches_direct_call(self) -> None:
        """Worker result for a TF must match calling run_alpha_foundry_l0_gate directly."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        # Compute real cheap evidences for the worker test
        tf_key = "4h"
        bound_panels = list(
            _bind_panels_to_recipe_ids(panels_by_tf[tf_key], bindings_by_tf.get(tf_key, []))
        )
        real_evidences = evaluate_alpha_cheap_gate_batch(
            panels=bound_panels, recipes=recipes_by_tf.get(tf_key, {}),
            aligned=aligned_by_tf[tf_key],
            cost_model=ExecutionCostModel(), config=RUNTIME_CONFIG.cheap_gate,
        )
        cheap_evidences_by_tf = {tf_key: real_evidences}

        # Prime cache with actual cheap evidences
        _prime_l0_tf_input_cache(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            evidence_by_tf={}, cheap_evidences_by_tf=cheap_evidences_by_tf,
        )

        # Worker result
        _tf, worker_result = _run_l0_gate_worker(tf_key, "test_worker")
        assert _tf == tf_key
        assert isinstance(worker_result, AlphaFoundryL0Result)

        # Direct call
        direct_result = run_alpha_foundry_l0_gate(
            panels=panels_by_tf[tf_key],
            bindings=bindings_by_tf.get(tf_key, []),
            recipes=recipes_by_tf.get(tf_key, {}),
            aligned=aligned_by_tf[tf_key],
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id="test_direct",
            timeframe=tf_key,
        )
        assert worker_result.report is not None
        assert direct_result.report is not None
        assert worker_result.report.n_passed == direct_result.report.n_passed
        assert worker_result.report.reject_reason_counts == direct_result.report.reject_reason_counts

    def test_cache_lifecycle(self) -> None:
        """Cache is cleared before each prime and does not leak across calls."""
        _L0_TF_INPUT_CACHE.clear()
        assert len(_L0_TF_INPUT_CACHE) == 0

        tfs_first = ("4h", "6h")
        panels_1, recipes_1, aligned_1, bindings_1 = _build_panels_recipes_aligned(tfs_first)
        _prime_l0_tf_input_cache(
            panels_by_tf=panels_1, bindings_by_tf=bindings_1,
            recipes_by_tf=recipes_1, aligned_by_tf=aligned_1,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            evidence_by_tf={}, cheap_evidences_by_tf={},
        )
        assert set(_L0_TF_INPUT_CACHE.keys()) == set(tfs_first)

        # Second prime with different TFs must clear first
        tfs_second = ("8h", "12h")
        panels_2, recipes_2, aligned_2, bindings_2 = _build_panels_recipes_aligned(tfs_second)
        _prime_l0_tf_input_cache(
            panels_by_tf=panels_2, bindings_by_tf=bindings_2,
            recipes_by_tf=recipes_2, aligned_by_tf=aligned_2,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            evidence_by_tf={}, cheap_evidences_by_tf={},
        )
        assert set(_L0_TF_INPUT_CACHE.keys()) == set(tfs_second)
        assert "4h" not in _L0_TF_INPUT_CACHE


class TestParallelDeterminism:
    """[LIMIT-06] Parallel execution is deterministic (fixed seed)."""

    def test_repeated_parallel_run_produces_identical_bootstrap_bps(self) -> None:
        """Two parallel runs with same fixture produce identical bootstrap_lcb_bps."""
        tfs = ("4h", "6h", "8h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        with patch(
            "src.domain.futures.alpha_foundry.bridge_helpers.ProcessPoolExecutor",
            new=SafeThreadPoolExecutor,
        ):
            run_1 = run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_det_1", parallel_max_workers=3,
            )
            run_2 = run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_det_2", parallel_max_workers=3,
            )

        for tf in tfs:
            r1_rows = run_1[tf].evidence_rows
            r2_rows = run_2[tf].evidence_rows
            assert len(r1_rows) == len(r2_rows), f"tf={tf} evidence row count mismatch"
            for i, (row1, row2) in enumerate(zip(r1_rows, r2_rows, strict=True)):
                assert row1.bootstrap_lcb_bps == pytest.approx(row2.bootstrap_lcb_bps), (
                    f"tf={tf} row={i} bootstrap_lcb_bps mismatch"
                )


class TestParallelWorkerExceptionPropagates:
    """[LIMIT-integration] Worker exception must propagate, not mask."""

    def test_worker_exception_raises_through_executor(self) -> None:
        """Exception inside _run_l0_gate_worker propagates as Future exception."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        with (
            patch(
                "src.domain.futures.alpha_foundry.bridge_helpers.ProcessPoolExecutor",
                new=SafeThreadPoolExecutor,
            ),
            patch(
                "src.domain.futures.alpha_foundry.bridge_helpers.run_alpha_foundry_l0_gate",
                side_effect=ValueError("simulated worker failure"),
            ),pytest.raises((ValueError, concurrent.futures.process.BrokenProcessPool))
        ):
            run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_err", parallel_max_workers=2,
            )


class TestCheapGateEvidenceFrameFromEvidences:
    """[LIMIT-05] Extracted DataFrame builder matches fresh computation."""

    def test_frame_from_evidences_matches_fresh_computation(self) -> None:
        """Scenario 1 row 1: DataFrame from precomputed evidences == fresh call."""
        tf = "4h"
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200)
        recipe = make_recipe_for_tf(tf=tf)
        recipes = {recipe.recipe_id: recipe}
        cheap_gate_config = make_gate_config()

        fresh_df = build_cheap_gate_evidence_frame(
            panels=[panel], recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), cheap_gate_config=cheap_gate_config, timeframe=tf,
        )

        evidences = evaluate_alpha_cheap_gate_batch(
            panels=[panel], recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), config=cheap_gate_config,
        )
        reused_df = build_cheap_gate_evidence_frame_from_evidences(
            cheap_evidences=evidences, recipes=recipes,
        )

        pd.testing.assert_frame_equal(fresh_df, reused_df)

    def test_precomputed_evidences_order_independent(self) -> None:
        """Scenario 2 LIMIT-02: Shuffled evidence order does not change output."""
        tf = "4h"
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel_a = make_panel_for_tf(tf=tf, family="fam_a", variant="var_a", bars=200)
        panel_b = make_panel_for_tf(tf=tf, family="fam_b", variant="var_b", bars=200, sign=-1.0)
        recipe_a = make_recipe_for_tf(tf=tf, family="fam_a", variant="var_a")
        recipe_b = make_recipe_for_tf(tf=tf, family="fam_b", variant="var_b")
        recipes = {recipe_a.recipe_id: recipe_a, recipe_b.recipe_id: recipe_b}
        cheap_gate_config = make_gate_config()
        panels = [panel_a, panel_b]

        fresh_df = build_cheap_gate_evidence_frame(
            panels=panels, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), cheap_gate_config=cheap_gate_config, timeframe=tf,
        )
        evidences = evaluate_alpha_cheap_gate_batch(
            panels=panels, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), config=cheap_gate_config,
        )
        shuffled = tuple(reversed(evidences))
        reused_df = build_cheap_gate_evidence_frame_from_evidences(
            cheap_evidences=shuffled, recipes=recipes,
        )

        # Output must match on content regardless of evidence order [LIMIT-02]
        fresh_sorted = fresh_df.sort_values("recipe_id").reset_index(drop=True)
        reused_sorted = reused_df.sort_values("recipe_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(fresh_sorted, reused_sorted)


class TestRunAlphaFoundryL0GateDedup:
    """[LIMIT-04] precomputed_cheap_evidences threading through l0_gate."""

    def test_with_precomputed_evidences_matches_fresh(self) -> None:
        """Scenario 1 row 2: l0_gate with precomputed_cheap_evidences == without."""
        tf = "4h"
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200)
        recipe = make_recipe_for_tf(tf=tf)
        recipes = {recipe.recipe_id: recipe}
        bindings = list(
            bind_panels_to_alpha_recipes(
                panels=[panel], recipes=recipes, timeframe=tf,
                max_recipes_per_family=10, include_families=(),
                exclude_families=(), enable_synthetic_recipes=True,
            )
        )
        assert bindings, "panel must bind to its own recipe"

        fresh_result = run_alpha_foundry_l0_gate(
            panels=[panel], bindings=bindings, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id="test_fresh", timeframe=tf,
        )

        cheap_evidences = evaluate_alpha_cheap_gate_batch(
            panels=[panel], recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), config=RUNTIME_CONFIG.cheap_gate,
        )
        dedup_result = run_alpha_foundry_l0_gate(
            panels=[panel], bindings=bindings, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id="test_dedup", timeframe=tf,
            precomputed_cheap_evidences=cheap_evidences,
        )

        assert fresh_result.report is not None
        assert dedup_result.report is not None
        assert fresh_result.report.n_passed == dedup_result.report.n_passed
        assert fresh_result.report.reject_reason_counts == dedup_result.report.reject_reason_counts
        assert len(fresh_result.panels_for_l1) == len(dedup_result.panels_for_l1)

    def test_precomputed_none_falls_back_to_recompute(self) -> None:
        """Scenario 2 LIMIT-04: precomputed_cheap_evidences=None === unchanged."""
        tf = "6h"
        aligned = make_aligned_for_tf(tf_hours=6, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200)
        recipe = make_recipe_for_tf(tf=tf)
        recipes = {recipe.recipe_id: recipe}
        bindings = list(
            bind_panels_to_alpha_recipes(
                panels=[panel], recipes=recipes, timeframe=tf,
                max_recipes_per_family=10, include_families=(),
                exclude_families=(), enable_synthetic_recipes=True,
            )
        )

        explicit_none = run_alpha_foundry_l0_gate(
            panels=[panel], bindings=bindings, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id="test_none", timeframe=tf, precomputed_cheap_evidences=None,
        )
        default = run_alpha_foundry_l0_gate(
            panels=[panel], bindings=bindings, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id="test_default", timeframe=tf,
        )
        assert explicit_none.report is not None
        assert default.report is not None
        assert explicit_none.report.n_passed == default.report.n_passed
        assert explicit_none.report.reject_reason_counts == default.report.reject_reason_counts

    def test_empty_precomputed_evidences_does_not_recompute(self) -> None:
        """Scenario 3: empty tuple is a valid precomputed result, not a sentinel."""
        tf = "4h"
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200)
        recipe = make_recipe_for_tf(tf=tf)
        recipes = {recipe.recipe_id: recipe}
        bindings = list(
            bind_panels_to_alpha_recipes(
                panels=[panel], recipes=recipes, timeframe=tf,
                max_recipes_per_family=10, include_families=(),
                exclude_families=(), enable_synthetic_recipes=True,
            )
        )

        result = run_alpha_foundry_l0_gate(
            panels=[panel], bindings=bindings, recipes=recipes, aligned=aligned,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id="test_empty", timeframe=tf, precomputed_cheap_evidences=(),
        )
        assert isinstance(result, AlphaFoundryL0Result)


class TestRunAlphaFoundryL0GateMultiTfDedup:
    """End-to-end dedup verification."""

    def test_dedup_wired_gate_multi_tf_identity(self) -> None:
        """Scenario 1 row 3: multi_tf with dedup wired produces identical results."""
        tfs = ("4h", "6h", "8h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_dedup_wired",
        )

        assert set(results) == set(tfs)
        for tf in tfs:
            assert len(results[tf].evidence_rows) == 1, f"tf={tf} expected 1 evidence row"

    def test_l0_gate_multi_tf_calls_cheap_gate_batch_once_per_tf_not_twice(
        self, mocker: Any,
    ) -> None:
        """Scenario 4: Direct proof the redundant computation is eliminated.
        evaluate_alpha_cheap_gate_batch must be called N times (Phase 1 only),
        not 2N times (Phase 1 + Phase 3 recomputation).

        Must patch BOTH call sites: Phase 1 (bridge_helpers.py) resolves the
        name dynamically via a local `from ... import` each iteration (so
        patching cheap_gate's module attribute is sufficient there), but
        Phase 3 (pipeline.py) binds the name once at its own module-import
        time (`from ... import evaluate_alpha_cheap_gate_batch` at the top
        of pipeline.py) -- patching only cheap_gate's attribute leaves
        pipeline.py's own reference untouched (verified empirically: `import
        src.domain.futures.alpha_foundry.pipeline as pl; pl.evaluate_alpha_cheap_gate_batch
        is cheap_gate.evaluate_alpha_cheap_gate_batch` diverges after patching
        only the latter). Patching only cheap_gate's attribute would make
        this test give a false pass if Phase 3 regressed to recomputing.
        """
        from src.domain.futures.alpha_foundry.cheap_gate import (
            evaluate_alpha_cheap_gate_batch as _real_evaluate_alpha_cheap_gate_batch,
        )

        call_count = {"n": 0}

        def _counting_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return _real_evaluate_alpha_cheap_gate_batch(*args, **kwargs)

        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            side_effect=_counting_wrapper,
        )
        mocker.patch(
            "src.domain.futures.alpha_foundry.pipeline.evaluate_alpha_cheap_gate_batch",
            side_effect=_counting_wrapper,
        )

        tfs = ("4h", "6h", "8h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf, bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf, aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(), runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_dedup_spy",
        )

        assert call_count["n"] == len(tfs), (
            f"expected {len(tfs)} calls (Phase 1 only, dedup active), got {call_count['n']} "
            f"(2x{len(tfs)} would mean Phase 3 is still recomputing)"
        )
