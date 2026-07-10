from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    _bind_panels_to_recipe_ids,
    run_alpha_foundry_l0_gate_multi_tf,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaRecipe,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_panel(family: str, variant: str) -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=np.asarray([np.datetime64("2026-01-01T00:00:00")], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        signed_score_2d=np.zeros((1, 1), dtype=np.float64),
        side_hint_2d=np.ones((1, 1), dtype=np.int8),
        expected_holding_bars=1,
        min_holding_bars=1,
        stop_atr_mult=50.0,
        take_profit_atr_mult=50.0,
        turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
        valid_mask_2d=np.ones((1, 1), dtype=bool),
        metadata={},
        archetype="trend",
    )


class _FakeBinding:
    def __init__(self, panel_index: int, recipe_id: str) -> None:
        self.panel_index = panel_index
        self.recipe_id = recipe_id


def test_bind_panels_to_recipe_ids_attaches_only_bound_indices() -> None:
    """S1-01: Happy path — only bound panels returned with recipe_id attached."""
    panel_a = _make_panel("trend_ma", "ema_12_72")
    panel_b = _make_panel("trend_ma", "ema_18_108")
    bindings = [_FakeBinding(panel_index=0, recipe_id="trend_ma:ema_12_72:4h:abc")]

    bound = _bind_panels_to_recipe_ids([panel_a, panel_b], bindings)

    assert len(bound) == 1
    assert bound[0].metadata["recipe_id"] == "trend_ma:ema_12_72:4h:abc"


def test_bind_panels_to_recipe_ids_empty_bindings() -> None:
    """S2-01: Empty bindings returns empty tuple."""
    panel_a = _make_panel("trend_ma", "ema_12_72")
    bound = _bind_panels_to_recipe_ids([panel_a], [])
    assert bound == ()


def test_evidence_by_tf_nonempty_when_bindings_present() -> None:
    """S1-02 + S2-02: With bound panels, evidence_by_tf must have rows."""
    from src.domain.futures.alpha_foundry.bridge_helpers import (
        bind_panels_to_alpha_recipes,
    )

    tf = "4h"
    aligned = _make_aligned(tf_hours=4)
    panel = _make_panel_for_tf(tf="4h", family="fam", variant="var")
    recipe = _make_recipe_for_tf(tf="4h", family="fam", variant="var")
    panels_by_tf = {tf: [panel]}
    recipes_by_tf: dict[str, MutableMapping[str, AlphaRecipe]] = {tf: {recipe.recipe_id: recipe}}
    aligned_by_tf = {tf: aligned}
    bindings_by_tf: dict[str, list[Any]] = {
        tf: list(
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
    }
    assert bindings_by_tf[tf], "panel must bind"

    results = run_alpha_foundry_l0_gate_multi_tf(
        panels_by_tf=panels_by_tf,
        bindings_by_tf=bindings_by_tf,
        recipes_by_tf=recipes_by_tf,
        aligned_by_tf=aligned_by_tf,
        cost_model=ExecutionCostModel(),
        runtime_config=_make_runtime_config(),
        run_id_prefix="test_s1_2",
    )

    evidence_rows = results[tf].evidence_rows
    assert len(evidence_rows) >= 1, (
        f"evidence_by_tf must have >=1 row when bindings exist, got {len(evidence_rows)}"
    )


def test_tf_coverage_count_nonzero_across_two_timeframes() -> None:
    """S1-02: same conceptual recipe bound at 2 TFs -> corroboration produces tf_coverage_count>=1.

    End-to-end regression for the Fix A defect: prior to binding panels before
    Phase 1's evidence_by_tf construction, this always yielded
    tf_coverage_count=0 for every recipe regardless of variant-name matching.
    """
    from src.domain.futures.alpha_foundry.bridge_helpers import bind_panels_to_alpha_recipes

    panels_by_tf: dict[str, list[Any]] = {}
    recipes_by_tf: dict[str, MutableMapping[str, AlphaRecipe]] = {}
    aligned_by_tf: dict[str, AlignedMarketData] = {}
    bindings_by_tf: dict[str, list[Any]] = {}

    for tf, hours in (("4h", 4), ("6h", 6)):
        aligned = _make_aligned(tf_hours=hours)
        panel = _make_panel_for_tf(tf=tf, family="fam", variant="var")
        recipe = _make_recipe_for_tf(tf=tf, family="fam", variant="var")
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
        assert bindings_by_tf[tf], f"panel must bind at tf={tf}"

    results = run_alpha_foundry_l0_gate_multi_tf(
        panels_by_tf=panels_by_tf,
        bindings_by_tf=bindings_by_tf,
        recipes_by_tf=recipes_by_tf,
        aligned_by_tf=aligned_by_tf,
        cost_model=ExecutionCostModel(),
        runtime_config=_make_runtime_config(),
        run_id_prefix="test_s1_02",
    )

    for tf in ("4h", "6h"):
        rows = results[tf].evidence_rows
        assert len(rows) >= 1, f"tf={tf}: expected evidence rows"
        assert any(row.tf_coverage_count >= 1 for row in rows), (
            f"tf={tf}: expected >=1 corroborated row, got tf_coverage_count values "
            f"{[row.tf_coverage_count for row in rows]}"
        )


def test_warning_logged_when_bindings_exist_but_no_evidence(caplog: pytest.LogCaptureFixture) -> None:
    """S2-06: WARNING guard logs but does not raise.

    Binding points to recipe_id not in recipes dict -> 0 evidence rows
    while bindings are non-empty -> WARNING level.
    """
    caplog.set_level(0)
    tf = "4h"
    panel = _make_panel_for_tf(tf="4h", family="fam", variant="var")
    recipe = _make_recipe_for_tf(tf="4h", family="fam", variant="var")
    aligned = _make_aligned(tf_hours=4)
    panels_by_tf = {tf: [panel]}
    recipes_by_tf: dict[str, MutableMapping[str, AlphaRecipe]] = {tf: {recipe.recipe_id: recipe}}
    aligned_by_tf = {tf: aligned}
    # Bind to a recipe_id that does NOT exist in recipes -> skip in cheap gate
    bindings_by_tf: dict[str, list[Any]] = {
        tf: [_FakeBinding(panel_index=0, recipe_id="nonexistent:recipe:id")]
    }

    results = run_alpha_foundry_l0_gate_multi_tf(
        panels_by_tf=panels_by_tf,
        bindings_by_tf=bindings_by_tf,
        recipes_by_tf=recipes_by_tf,
        aligned_by_tf=aligned_by_tf,
        cost_model=ExecutionCostModel(),
        runtime_config=_make_runtime_config(),
        run_id_prefix="test_s2_6",
    )
    assert len(results[tf].evidence_rows) == 0
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) >= 1, (
        f"Expected WARNING log when bindings present but 0 evidence rows, got {caplog.records}"
    )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_aligned(*, tf_hours: int, bars: int = 200) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(tf_hours * bars, "h"),
        np.timedelta64(tf_hours, "h"),
        dtype="datetime64[ns]",
    )
    t = dt.shape[0]
    close = 100.0 * np.exp(0.001 * np.arange(t, dtype=np.float64))[:, None]
    mask = np.ones((t, 1), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt, symbols=("BTCUSDT",),
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
        volume_2d=np.full((t, 1), 1000.0), funding_2d=np.full((t, 1), 0.00005),
        active_mask=mask, warm_mask=mask,
        entry_block_mask=np.zeros((t, 1), dtype=np.bool_), kill_mask=np.zeros((t, 1), dtype=np.bool_),
    )


def _make_panel_for_tf(*, tf: str, family: str, variant: str, bars: int = 200) -> CandidateSignalPanel:
    score = np.full((bars, 1), 0.0)
    side = np.zeros((bars, 1), dtype=np.int8)
    score[10, 0] = 0.8
    side[10:13, 0] = 1
    return CandidateSignalPanel(
        family=family, variant=f"{variant}_{tf}",
        params={"lookback": 20}, datetimes=np.arange(bars, dtype=np.int64),
        symbols=("BTCUSDT",),
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=np.ones((bars, 1), dtype=np.bool_),
        metadata={},
    )


def _make_panel_no_events(family: str, variant: str, bars: int = 200) -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family=family, variant=variant,
        params={}, datetimes=np.arange(bars, dtype=np.int64),
        symbols=("BTCUSDT",),
        signed_score_2d=np.zeros((bars, 1), dtype=np.float64),
        side_hint_2d=np.zeros((bars, 1), dtype=np.int8),
        expected_holding_bars=1, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((bars, 1), dtype=np.float64),
        valid_mask_2d=np.zeros((bars, 1), dtype=bool),
        metadata={},
    )


def _make_recipe_for_tf(*, tf: str, family: str, variant: str) -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id=f"{family}:{variant}:{tf}", family=family, variant=f"{variant}_{tf}",
        timeframe=tf, archetype="trend", indicator_params={"lookback": 20},
        side_rule_id="trend_follow", exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1, max_turnover_per_year=365.0,
    )


def _make_gate_config() -> AlphaGateConfig:
    return AlphaGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0, min_nw_tstat=0.0,
        max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0, min_candidate_rank_ic_tstat=0.0,
        archetype_event_floors={},
    )


def _make_runtime_config() -> AlphaFoundryRuntimeConfig:
    return AlphaFoundryRuntimeConfig(
        mode="gate",
        cheap_gate=_make_gate_config(),
        max_recipes_per_family=10,
    )
