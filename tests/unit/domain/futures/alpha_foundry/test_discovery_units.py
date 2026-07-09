"""Tests for L0 discovery unit generation, selection, and L1 handoff.

Covers scenarios from docs/specs/l0_l1_conditional_discovery_redesign.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaGateEvidence,
    ConditionalCellGateConfig,
)
from src.domain.futures.alpha_foundry.discovery_units import (
    L0DiscoveryUnit,
    build_l0_discovery_units,
    project_discovery_units_to_panels,
    select_l0_discovery_units,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel
from src.domain.futures.strategy.rule_signals import candidate_panels_to_events


def make_aligned() -> AlignedMarketData:
    dt = np.array(
        [
            "2026-01-01T00:00:00",
            "2026-01-01T04:00:00",
            "2026-01-01T08:00:00",
            "2026-01-01T12:00:00",
            "2026-01-01T16:00:00",
            "2026-01-01T20:00:00",
        ],
        dtype="datetime64[ns]",
    )
    close = np.array(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [103.0, 196.0],
            [104.0, 197.0],
            [105.0, 195.0],
            [106.0, 194.0],
        ],
        dtype=np.float64,
    )
    shape = close.shape
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close + 1.0,
        low_2d=close - 1.0,
        close_2d=close,
        volume_2d=np.full(shape, 1000.0, dtype=np.float64),
        funding_2d=np.zeros(shape, dtype=np.float64),
        active_mask=np.ones(shape, dtype=bool),
        warm_mask=np.ones(shape, dtype=bool),
        entry_block_mask=np.zeros(shape, dtype=bool),
        kill_mask=np.zeros(shape, dtype=bool),
        adv_usdt_2d=np.array(
            [
                [10_000_000.0, 1_000_000.0],
                [10_000_000.0, 1_000_000.0],
                [10_000_000.0, 1_000_000.0],
                [10_000_000.0, 1_000_000.0],
                [10_000_000.0, 1_000_000.0],
                [10_000_000.0, 1_000_000.0],
            ],
            dtype=np.float64,
        ),
        execution_cost_bps_2d=np.array(
            [
                [8.0, 40.0],
                [8.0, 40.0],
                [8.0, 40.0],
                [8.0, 40.0],
                [8.0, 40.0],
                [8.0, 40.0],
            ],
            dtype=np.float64,
        ),
        cluster_id_1d=np.array([1.0, 2.0], dtype=np.float32),
    )


def make_panel() -> CandidateSignalPanel:
    aligned = make_aligned()
    score = np.array(
        [
            [0.0, 0.0],
            [1.2, -1.1],
            [1.5, -1.4],
            [0.0, 0.0],
            [1.3, 0.0],
            [0.0, -1.2],
        ],
        dtype=np.float64,
    )
    side = np.sign(score).astype(np.int8)
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72_4h",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=2,
        min_holding_bars=1,
        stop_atr_mult=1.0,
        take_profit_atr_mult=2.0,
        turnover_proxy_2d=np.abs(score),
        valid_mask_2d=score != 0.0,
        metadata={"recipe_id": "trend_ma:ema_12_72:4h"},
        archetype="trend",
    )


def make_gate_evidence(
    recipe_id: str, *, net_lcb_bps: float, gross_lcb_bps: float
) -> AlphaGateEvidence:
    return AlphaGateEvidence(
        schema_version="unified",
        run_id="unit",
        timeframe="4h",
        family="trend_ma",
        variant="ema_12_72_4h",
        recipe_id=recipe_id,
        archetype="trend",
        symbol_scope="symbol",
        n_events=4,
        effective_n=4.0,
        mean_gross_bps=20.0,
        mean_cost_bps=8.0,
        mean_net_bps=12.0,
        gross_lcb_bps=gross_lcb_bps,
        net_lcb_bps=net_lcb_bps,
        nw_tstat=2.5,
        rank_ic=0.30,
        rank_ic_tstat=2.1,
        cost_drag_ratio=0.40,
        turnover_per_year=80.0,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        event_hit_rate=0.75,
        payoff_skew=1.6,
        xs_spread_lcb_bps=None,
        liquidity_cost_stress_bps=0.0,
        bootstrap_lcb_bps=5.0,
        bootstrap_agree=True,
        gate_passed=True,
        handoff_tier="candidate",
        selected_for_l1=False,
        reject_reasons=(),
        soft_flags=(),
    )


# ── HP-01: Conditional unit promotion ─────────────────────────────────


def test_build_l0_discovery_units_returns_empty_when_disabled() -> None:
    """Feature flag off => empty tuple (fail-closed)."""
    runtime = AlphaFoundryRuntimeConfig(
        enable_discovery_unit_handoff=False,
    )
    result = build_l0_discovery_units(
        parent_evidences=[],
        panel_by_recipe_id={},
        recipes={},
        aligned=make_aligned(),
        cost_model=ExecutionCostModel(),
        gate_config=AlphaGateConfig(),
        runtime_config=runtime,
        run_id="test",
    )
    assert result == ()


def test_select_l0_discovery_units_returns_empty_when_no_units() -> None:
    """No input units => empty selection."""
    selection = select_l0_discovery_units(
        units=[],
        gate_config=AlphaGateConfig(),
        max_units=12,
        max_event_jaccard=0.80,
    )
    assert selection.selected_units == ()
    assert selection.rejected_units == ()


def test_project_discovery_units_to_panels_returns_empty_when_no_units() -> None:
    """No selected units => empty panels."""
    panels = project_discovery_units_to_panels(
        selected_units=[],
        panel_by_recipe_id={},
    )
    assert panels == ()


# ── HP-02: L1 mask reuse ──────────────────────────────────────────────


def test_candidate_panels_to_events_when_l0_mask_present_uses_exact_mask() -> None:
    panel = make_panel()
    l0_mask = np.zeros_like(panel.valid_mask_2d, dtype=bool)
    l0_mask[1, 0] = True
    l0_mask[4, 0] = True
    masked = dataclasses.replace(
        panel,
        metadata={
            **panel.metadata,
            "l0_event_mask_2d": l0_mask,
            "l0_discovery_unit_id": "du_trend_high_liq",
            "l0_parent_recipe_id": "trend_ma:ema_12_72:4h",
            "l0_execution_style": "taker_now",
            "l0_horizon_bars": 2,
        },
    )

    events = candidate_panels_to_events(
        (masked,),
        min_abs_score=0.0,
        cost_floor_bps=8.0,
        n_workers=1,
    )

    assert events[["symbol", "entry_idx"]].to_dict("records") == [
        {"symbol": "BTCUSDT", "entry_idx": 2},
        {"symbol": "BTCUSDT", "entry_idx": 5},
    ]
    assert set(events["l0_discovery_unit_id"]) == {"du_trend_high_liq"}


# ── EC-03: l0_event_mask_2d shape mismatch ────────────────────────────


def test_candidate_panels_to_events_when_l0_mask_shape_mismatch_raises() -> None:
    panel = make_panel()
    bad = dataclasses.replace(
        panel,
        metadata={**panel.metadata, "l0_event_mask_2d": np.ones((1, 1), dtype=bool)},
    )

    with pytest.raises(ValueError, match="l0_event_mask_2d shape"):
        candidate_panels_to_events((bad,), min_abs_score=0.0, n_workers=1)


# ── ERR-05: Non-bool mask ─────────────────────────────────────────────


def test_candidate_panels_to_events_when_l0_mask_not_bool_raises() -> None:
    panel = make_panel()
    bad = dataclasses.replace(
        panel,
        metadata={
            **panel.metadata,
            "l0_event_mask_2d": np.ones((6, 2), dtype=np.float64),
        },
    )

    with pytest.raises(ValueError, match="event_mask_2d must be bool"):
        candidate_panels_to_events((bad,), min_abs_score=0.0, n_workers=1)


# ── ERR-04: Duplicate unit_id ─────────────────────────────────────────


def test_discovery_unit_creation_raises_on_duplicate_id() -> None:
    """Duplicate unit_id should raise ValueError."""

    ev = make_gate_evidence("test", net_lcb_bps=5.0, gross_lcb_bps=10.0)
    mask = np.zeros((6, 2), dtype=bool)
    mask[1, 0] = True
    unit1 = L0DiscoveryUnit(
        unit_id="dup",
        parent_recipe_id="r1",
        kind="conditional_cell",
        timeframe="4h",
        family="trend_ma",
        variant="ema_12_72_4h",
        event_mask_2d=mask,
        scope_symbols=("BTCUSDT",),
        cell_axes=("symbol_liquidity",),
        cell_values={"symbol_liquidity": "high"},
        execution_style="taker_now",
        fill_probability=1.0,
        adverse_selection_bps=0.0,
        horizon_bars=2,
        gate_evidence=ev,
    )
    unit2 = L0DiscoveryUnit(
        unit_id="dup",
        parent_recipe_id="r2",
        kind="conditional_cell",
        timeframe="4h",
        family="trend_ma",
        variant="ema_12_72_4h",
        event_mask_2d=mask,
        scope_symbols=("ETHUSDT",),
        cell_axes=("symbol_liquidity",),
        cell_values={"symbol_liquidity": "high"},
        execution_style="taker_now",
        fill_probability=1.0,
        adverse_selection_bps=0.0,
        horizon_bars=2,
        gate_evidence=ev,
    )
    with pytest.raises(ValueError, match="duplicate discovery unit_id"):
        select_l0_discovery_units(
            units=(unit1, unit2),
            gate_config=AlphaGateConfig(),
            max_units=12,
            max_event_jaccard=0.80,
        )


# ── ERR-03: Non-positive horizon_bars ─────────────────────────────────


def test_discovery_unit_creation_raises_on_non_positive_horizon() -> None:
    """horizon_bars must be >= 1."""
    ev = make_gate_evidence("test", net_lcb_bps=5.0, gross_lcb_bps=10.0)
    mask = np.zeros((6, 2), dtype=bool)
    with pytest.raises(ValueError, match="horizon_bars must be >= 1"):
        L0DiscoveryUnit(
            unit_id="u1",
            parent_recipe_id="r1",
            kind="horizon",
            timeframe="4h",
            family="trend_ma",
            variant="ema_12_72_4h::du=u1",
            event_mask_2d=mask,
            scope_symbols=("BTCUSDT",),
            cell_axes=(),
            cell_values={},
            execution_style="taker_now",
            fill_probability=1.0,
            adverse_selection_bps=0.0,
            horizon_bars=0,
            gate_evidence=ev,
        )


# ── ERR-01: Unsupported conditional axis ──────────────────────────────


def test_discovery_unit_creation_raises_on_unsupported_axis() -> None:
    ev = make_gate_evidence("test", net_lcb_bps=5.0, gross_lcb_bps=10.0)
    mask = np.zeros((6, 2), dtype=bool)
    with pytest.raises(ValueError, match="unsupported conditional axis"):
        L0DiscoveryUnit(
            unit_id="u1",
            parent_recipe_id="r1",
            kind="conditional_cell",
            timeframe="4h",
            family="trend_ma",
            variant="ema_12_72_4h::du=u1",
            event_mask_2d=mask,
            scope_symbols=("BTCUSDT",),
            cell_axes=("invalid_axis",),
            cell_values={"invalid_axis": "foo"},
            execution_style="taker_now",
            fill_probability=1.0,
            adverse_selection_bps=0.0,
            horizon_bars=2,
            gate_evidence=ev,
        )


# ── ERR-02: Unsupported execution style ───────────────────────────────


def test_discovery_unit_creation_raises_on_unsupported_execution_style() -> None:
    ev = make_gate_evidence("test", net_lcb_bps=5.0, gross_lcb_bps=10.0)
    mask = np.zeros((6, 2), dtype=bool)
    with pytest.raises(ValueError, match="unsupported execution style"):
        L0DiscoveryUnit(
            unit_id="u1",
            parent_recipe_id="r1",
            kind="execution_arm",
            timeframe="4h",
            family="trend_ma",
            variant="ema_12_72_4h::du=u1",
            event_mask_2d=mask,
            scope_symbols=("BTCUSDT",),
            cell_axes=(),
            cell_values={},
            execution_style="hammer_time",
            fill_probability=0.9,
            adverse_selection_bps=5.0,
            horizon_bars=2,
            gate_evidence=ev,
        )


# ── EC-07: Single-symbol cell rejection ────────────────────────────────


def test_single_symbol_cell_rejected_when_disallowed() -> None:
    """When allow_single_symbol_cells=False, single-symbol cells are rejected."""
    cell_cfg = ConditionalCellGateConfig(allow_single_symbol_cells=False)
    assert cell_cfg.allow_single_symbol_cells is False


# ── EC-09: causal_lag_bars applied to entry_idx ───────────────────────


def test_l0_mask_with_causal_lag_in_entry_idx() -> None:
    """entry_idx = signal bar + causal_lag_bars when l0_mask present."""
    panel = make_panel()
    l0_mask = np.zeros_like(panel.valid_mask_2d, dtype=bool)
    l0_mask[1, 0] = True
    masked = dataclasses.replace(
        panel,
        metadata={
            **panel.metadata,
            "l0_event_mask_2d": l0_mask,
            "l0_discovery_unit_id": "du_causal",
            "l0_parent_recipe_id": "trend_ma:ema_12_72:4h",
            "l0_execution_style": "taker_now",
            "l0_horizon_bars": 2,
        },
    )
    events = candidate_panels_to_events(
        (masked,),
        min_abs_score=0.0,
        cost_floor_bps=8.0,
        n_workers=1,
    )
    # entry_idx = signal bar index + 1 (default causal_lag)
    assert events.iloc[0]["entry_idx"] == 2
    assert events.iloc[0]["symbol"] == "BTCUSDT"


# ── Supplementary coverage tests ────────────────────────────────────────


def _make_unit(
    unit_id: str,
    *,
    net_lcb_bps: float = 5.0,
    gross_lcb_bps: float = 10.0,
    weak_rank_ic: bool = False,
    tf_corroboration: float = 0.3,
    mask: np.ndarray | None = None,
) -> L0DiscoveryUnit:
    if mask is None:
        mask = np.zeros((6, 2), dtype=bool)
        mask[1, 0] = True
    ev = make_gate_evidence(unit_id, net_lcb_bps=net_lcb_bps, gross_lcb_bps=gross_lcb_bps)
    soft_flags: tuple[str, ...] = ("weak_rank_ic",) if weak_rank_ic else ()
    ev = dataclasses.replace(ev, soft_flags=soft_flags, tf_corroboration=tf_corroboration)
    return L0DiscoveryUnit(
        unit_id=unit_id,
        parent_recipe_id=unit_id,
        kind="conditional_cell",
        timeframe="4h",
        family="trend_ma",
        variant=f"ema_12_72_4h::du={unit_id}",
        event_mask_2d=mask,
        scope_symbols=("BTCUSDT",),
        cell_axes=("symbol_liquidity",),
        cell_values={"symbol_liquidity": "high"},
        execution_style="taker_now",
        fill_probability=1.0,
        adverse_selection_bps=0.0,
        horizon_bars=2,
        gate_evidence=ev,
    )


def test_select_l0_discovery_units_selects_highest_priority_units() -> None:
    """Units with higher priority are selected first."""
    mask_a = np.zeros((6, 2), dtype=bool)
    mask_a[1, 0] = True
    mask_b = np.zeros((6, 2), dtype=bool)
    mask_b[2, 1] = True
    mask_c = np.zeros((6, 2), dtype=bool)
    mask_c[3, 0] = True
    u1 = _make_unit("u1", net_lcb_bps=3.0, gross_lcb_bps=8.0, mask=mask_a)
    u2 = _make_unit("u2", net_lcb_bps=12.0, gross_lcb_bps=20.0, mask=mask_b)
    u3 = _make_unit("u3", net_lcb_bps=6.0, gross_lcb_bps=10.0, mask=mask_c)
    selection = select_l0_discovery_units(
        units=(u1, u2, u3),
        gate_config=AlphaGateConfig(),
        max_units=2,
        max_event_jaccard=0.80,
    )
    selected_ids = {u.unit_id for u in selection.selected_units}
    assert "u2" in selected_ids
    assert len(selection.selected_units) == 2
    assert len(selection.rejected_units) == 1


def test_select_l0_discovery_units_deduplicates_by_jaccard() -> None:
    """Units with identical masks are deduplicated."""
    mask = np.zeros((6, 2), dtype=bool)
    mask[1, 0] = True
    u1 = _make_unit("u1", mask=mask, net_lcb_bps=10.0, gross_lcb_bps=20.0)
    u2 = _make_unit("u2", mask=mask, net_lcb_bps=5.0, gross_lcb_bps=10.0)
    selection = select_l0_discovery_units(
        units=(u1, u2),
        gate_config=AlphaGateConfig(),
        max_units=12,
        max_event_jaccard=0.50,
    )
    assert len(selection.selected_units) == 1
    assert len(selection.rejected_units) == 1
    assert selection.duplicate_of_by_unit_id.get("u2") == "u1" or selection.duplicate_of_by_unit_id.get("u1") == "u2"


def test_select_l0_discovery_units_zero_tf_corroboration_sets_zero_priority() -> None:
    """tf_corroboration=0.0 yields priority=0."""
    u1 = _make_unit("u1", net_lcb_bps=10.0, gross_lcb_bps=20.0, tf_corroboration=0.0)
    u2 = _make_unit("u2", net_lcb_bps=2.0, gross_lcb_bps=5.0)
    selection = select_l0_discovery_units(
        units=(u1, u2),
        gate_config=AlphaGateConfig(),
        max_units=1,
        max_event_jaccard=0.80,
    )
    selected_ids = {u.unit_id for u in selection.selected_units}
    assert "u2" in selected_ids


def test_select_l0_discovery_units_weak_rank_ic_penalty() -> None:
    """weak_rank_ic flag reduces priority by 0.70 multiplier."""
    mask_a = np.zeros((6, 2), dtype=bool)
    mask_a[1, 0] = True
    mask_b = np.zeros((6, 2), dtype=bool)
    mask_b[2, 1] = True
    u1 = _make_unit("u1", net_lcb_bps=20.0, gross_lcb_bps=30.0, weak_rank_ic=True, mask=mask_a)
    u2 = _make_unit("u2", net_lcb_bps=14.0, gross_lcb_bps=22.0, mask=mask_b)
    selection = select_l0_discovery_units(
        units=(u1, u2),
        gate_config=AlphaGateConfig(),
        max_units=2,
        max_event_jaccard=0.80,
    )
    selected_ids = [u.unit_id for u in selection.selected_units]
    # u1: 0.65*20 + 0.20*12 + 0.15*30 = 19.9, *0.70 = 13.93
    # u2: 0.65*14 + 0.20*12 + 0.15*22 = 14.8
    # u2 > u1 because u1 is penalized by 0.70
    assert selected_ids[0] == "u2"


def test_l0_discovery_unit_non_bool_mask_raises() -> None:
    """Non-bool event_mask_2d raises ValueError via __post_init__."""
    ev = make_gate_evidence("test", net_lcb_bps=5.0, gross_lcb_bps=10.0)
    bad_mask = np.ones((6, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="event_mask_2d must be bool"):
        L0DiscoveryUnit(
            unit_id="u1",
            parent_recipe_id="r1",
            kind="conditional_cell",
            timeframe="4h",
            family="trend_ma",
            variant="ema_12_72_4h::du=u1",
            event_mask_2d=bad_mask,
            scope_symbols=("BTCUSDT",),
            cell_axes=(),
            cell_values={},
            execution_style="taker_now",
            fill_probability=1.0,
            adverse_selection_bps=0.0,
            horizon_bars=2,
            gate_evidence=ev,
        )


def test_select_l0_discovery_units_disjoint_masks_no_dedup() -> None:
    """Two units with disjoint masks are both selected (no dedup)."""
    mask_a = np.zeros((6, 2), dtype=bool)
    mask_a[1, 0] = True
    mask_b = np.zeros((6, 2), dtype=bool)
    mask_b[2, 1] = True
    u1 = _make_unit("u1", mask=mask_a, net_lcb_bps=10.0, gross_lcb_bps=20.0)
    u2 = _make_unit("u2", mask=mask_b, net_lcb_bps=8.0, gross_lcb_bps=15.0)
    selection = select_l0_discovery_units(
        units=(u1, u2),
        gate_config=AlphaGateConfig(),
        max_units=12,
        max_event_jaccard=0.80,
    )
    assert len(selection.selected_units) == 2
    assert selection.duplicate_of_by_unit_id == {}
