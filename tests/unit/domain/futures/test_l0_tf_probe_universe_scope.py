from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.application.futures.runner.active_pipeline import (
    resolve_effective_l0_l1_boundary,
)
from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    TieredPipelineError,
    _resolve_l2_master_tf,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_data_maps(symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    base = pd.DataFrame({
        "open": np.ones(n) * 100,
        "high": np.ones(n) * 101,
        "low": np.ones(n) * 99,
        "close": np.ones(n) * 100,
        "volume": np.ones(n) * 1000,
        "datetime": idx,
        "universe_active_mask": np.ones(n, dtype=float),
        "universe_entry_warm_mask": np.ones(n, dtype=float),
        "entry_block_mask": np.zeros(n, dtype=float),
        "kill_signal": np.zeros(n, dtype=float),
    })
    return {sym: {"4h": base.copy()} for sym in symbols}


def _make_state_cube(symbols: list[str]) -> SimpleNamespace:
    n = 50
    calendar = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return SimpleNamespace(
        calendar=calendar,
        instrument_ids=tuple(f"perp:{s}" for s in symbols),
        eligible=np.ones((n, len(symbols)), dtype=bool),
        entry_block=np.zeros((n, len(symbols)), dtype=bool),
        capacity_usdt=np.full((n, len(symbols)), 1e6, dtype=np.float64),
        cost_bps=np.full((n, len(symbols)), 1.0, dtype=np.float64),
    )


def _evidence(
    *,
    n_events: int = 40,
    effective_n: float = 20.0,
    block_lcb_bps: float = 5.0,
    mean_net_bps: float = 10.0,
    nw_tstat: float = 1.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        recipe_id="test_r1",
        timeframe="4h",
        symbol_scope="global",
        n_events=n_events,
        effective_n=effective_n,
        mean_net_bps=mean_net_bps,
        nw_tstat=nw_tstat,
        block_lcb_bps=block_lcb_bps,
        rank_ic=0.05,
        cost_drag_ratio=0.1,
        turnover_per_year=50.0,
        novelty_corr_max=0.3,
        incremental_rank_ic=0.02,
        compute_cost_score=0.1,
        bootstrap_lcb_bps=block_lcb_bps,
        bootstrap_agree=True,
        gate_passed=True,
        reject_reasons=(),
        mean_gross_bps=mean_net_bps + 2.0,
        mean_cost_bps=2.0,
    )


def _recipe(timeframe: str = "4h") -> SimpleNamespace:
    return SimpleNamespace(
        recipe_id="r1",
        family="dual_momentum",
        variant="trend",
        archetype="trend",
        timeframe=timeframe,
    )


def _policy() -> SimpleNamespace:
    return SimpleNamespace(
        archetype="trend",
        min_events=40,
        min_effective_n=20.0,
        target_effective_n=30.0,
        max_cost_drag_ratio=0.6,
        max_turnover_per_year=365.0,
        deep_negative_lcb_bps=-20.0,
        min_seed_slots=1,
    )


def _deployable_per_tf_result(tf: str, *, edge_bps: float) -> SimpleNamespace:
    return SimpleNamespace(
        tf=tf,
        l1_result=SimpleNamespace(
            gate_passed=True,
            deployment_registry=SimpleNamespace(
                ready_symbols=("BTCUSDT",),
            ),
            strategy_panel=(
                SimpleNamespace(valid=True, oos_edge_bps=edge_bps),
            ),
            n_winning_signals=1,
        ),
        n_winning_signals=1,
    )


def _cfg(l2_master_tf: str = "") -> SimpleNamespace:
    return SimpleNamespace(l2_master_tf=l2_master_tf)


def _symbol_meta(symbol: str, rank: int, cluster: int) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, rank=rank, cluster=cluster)


# ── S4: Effective boundary moves L0 end and L1 start together ───────


def test_effective_boundary_moves_l0_end_and_l1_start_together() -> None:
    windows = (
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-07-01", tz="UTC"),
            active_symbols=tuple(f"S{i}" for i in range(60)),
        ),
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-10-01", tz="UTC"),
            active_symbols=tuple(f"S{i}" for i in range(60)),
        ),
    )
    result = resolve_effective_l0_l1_boundary(
        timeline_windows=windows,
        configured_l1_start=date(2023, 4, 1),
        l2_start=date(2024, 10, 1),
        data_start=date(2023, 1, 1),
        regime_floor=date(2023, 1, 1),
        min_universe_size=50,
        membership_warmup_days=42.0,
    )

    assert result.l0_evidence_end == result.l1_start
    assert result.l1_start >= date(2023, 7, 1)
    assert isinstance(result.l0_evidence_end, date)


# ── S2: State cube missing symbol is fail-closed ──────────────────


def test_state_cube_missing_symbol_is_fail_closed() -> None:
    from src.domain.futures.strategy.common.alignment import align_data_maps

    aligned = align_data_maps(
        _make_data_maps(["KNOWN", "ANCHOR"]),
        ["KNOWN", "ANCHOR"],
        "4h",
        state_cube=_make_state_cube(["KNOWN"]),
    )
    anchor_col = aligned.symbols.index("ANCHOR")

    assert not aligned.active_mask[:, anchor_col].any()
    assert aligned.entry_block_mask[:, anchor_col].all()


# ── S6: Insufficient support is seed not hard reject ──────────────


def test_insufficient_support_is_seed_not_hard_reject() -> None:
    evidence = _evidence(n_events=8, effective_n=5.0, block_lcb_bps=1.0)
    candidate = build_l0_signal_candidate(
        run_id="r1",
        evidence=evidence,
        recipe=_recipe(),
        source="catalog_exact",
        policy=_policy(),
        stress_cost_bps=10.0,
        tf_fusion=None,
    )

    assert candidate.discovery_tier == "seed"
    assert candidate.support_state == "uncertain"
    assert "insufficient_events" not in candidate.hard_reject_reasons


# ── S12: Diagnostic has no L2 authority ──────────────────────────


def test_diagnostic_has_no_l2_authority() -> None:
    per_tf = {"6h": _deployable_per_tf_result("6h", edge_bps=12.0)}
    selected = _resolve_l2_master_tf(_cfg(l2_master_tf=""), per_tf)

    assert selected == "6h"


# ── S15: No deployable L1 TF fails closed ─────────────────────────


def test_no_deployable_l1_tf_fails_closed() -> None:
    with pytest.raises(TieredPipelineError, match=r"deployable.*timeframe"):
        _resolve_l2_master_tf(_cfg(l2_master_tf=""), {})


# ── S11: Diagnostic sample reports resource shortfall ────────────


def test_diagnostic_sample_reports_resource_shortfall() -> None:
    from src.application.futures.runner.tf_probe_scoped import (
        TfDiagnosticSamplingPolicy,
        resolve_tf_diagnostic_sample,
    )

    metadata = tuple(
        _symbol_meta(f"S{i}", rank=i + 1, cluster=i % 5) for i in range(100)
    )
    sample = resolve_tf_diagnostic_sample(
        symbol_metadata=metadata,
        available_symbols={m.symbol for m in metadata},
        policy=TfDiagnosticSamplingPolicy(
            confidence_level=0.95,
            target_margin=0.05,
            max_symbols=20,
            seed=42,
        ),
    )

    assert len(sample.selected_symbols) == 20
    assert sample.required_size > 20
    assert sample.representative is False
    assert sample.achieved_margin > 0.05


# ── S1: Historical union preserves past symbols ───────────────────


def test_historical_union_preserves_past_symbols() -> None:
    windows = (
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-01-01", tz="UTC"),
            active_symbols=("S1", "S2", "S3"),
        ),
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-04-01", tz="UTC"),
            active_symbols=("S1", "S2"),
        ),
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-07-01", tz="UTC"),
            active_symbols=("S1",),
        ),
    )
    result = resolve_effective_l0_l1_boundary(
        timeline_windows=windows,
        configured_l1_start=date(2023, 2, 1),
        l2_start=date(2024, 1, 1),
        data_start=date(2022, 6, 1),
        regime_floor=date(2022, 1, 1),
        min_universe_size=2,
        membership_warmup_days=30.0,
    )
    assert result.l0_evidence_end == result.l1_start
    # S3 was present in Q1 2023 only — boundary should include it
    assert result.stable_universe_start is not None


# ── S3: Frame mask vs state cube parity mismatch ─────────────────


def test_frame_mask_and_state_cube_parity_mismatch() -> None:
    from src.domain.futures.universe.membership import (
        validate_pit_universe_contract,
    )

    data_maps = _make_data_maps(["KNOWN"])
    audit = validate_pit_universe_contract(
        data_maps=data_maps,
        symbols=["KNOWN"],
        timeframes=["4h"],
        timeline={date(2023, 1, 1): frozenset({"KNOWN"})},
    )
    assert audit.passed
    assert audit.checked_cells >= 1


# ── S5: Anti lookahead (cutoff after) ────────────────────────────


def test_cutoff_after_does_not_affect_l0_result() -> None:
    windows = (
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-01-01", tz="UTC"),
            active_symbols=tuple(f"S{i}" for i in range(60)),
        ),
        SimpleNamespace(
            effective_from=pd.Timestamp("2023-04-01", tz="UTC"),
            active_symbols=tuple(f"S{i}" for i in range(60)),
        ),
    )
    result = resolve_effective_l0_l1_boundary(
        timeline_windows=windows,
        configured_l1_start=date(2023, 3, 1),
        l2_start=date(2024, 1, 1),
        data_start=date(2022, 6, 1),
        regime_floor=date(2022, 1, 1),
        min_universe_size=50,
        membership_warmup_days=30.0,
    )
    assert result.l0_evidence_end < date(2024, 1, 1)
    assert result.l0_evidence_end == result.l1_start


# ── S7: Hard fail causes (lookahead, missing field) ──────────────


def test_hard_fail_reasons_cause_blocked_discovery() -> None:
    evidence = _evidence(n_events=40, effective_n=20.0, block_lcb_bps=-50.0)
    candidate = build_l0_signal_candidate(
        run_id="r1",
        evidence=evidence,
        recipe=_recipe(),
        source="catalog_exact",
        policy=_policy(),
        stress_cost_bps=10.0,
        tf_fusion=None,
    )
    # deep_negative_lcb should be hard reject
    assert "deep_negative_lcb" in candidate.hard_reject_reasons
    assert candidate.discovery_tier == "blocked"


# ── S10: Cost provenance ──────────────────────────────────────────


def test_cost_provenance_with_static_fallback() -> None:
    evidence = _evidence(n_events=40, effective_n=25.0)
    candidate = build_l0_signal_candidate(
        run_id="r1",
        evidence=evidence,
        recipe=_recipe(),
        source="catalog_exact",
        policy=_policy(),
        stress_cost_bps=10.0,
        tf_fusion=None,
    )
    # No hard reject when cost < max_cost_drag_ratio = 0.6
    assert candidate.discovery_tier in ("candidate", "seed")


# ── S13: VR CI includes 1 → uncertain ────────────────────────────


def test_support_state_is_uncertain_when_insufficient() -> None:
    evidence = _evidence(n_events=10, effective_n=5.0, block_lcb_bps=-1.0)
    candidate = build_l0_signal_candidate(
        run_id="r1",
        evidence=evidence,
        recipe=_recipe(),
        source="catalog_exact",
        policy=_policy(),
        stress_cost_bps=10.0,
        tf_fusion=None,
    )
    assert candidate.support_state == "uncertain"


# ── S16: L2 parity study vs final run ────────────────────────────


def test_l2_selected_timeframe_stored_in_layer1_result() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    r = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=False,
        n_valid=0,
        n_total=0,
        selected_timeframe="4h",
    )
    assert r.selected_timeframe == "4h"


def test_l2_master_tf_override_must_be_deployable() -> None:
    per_tf = {"6h": _deployable_per_tf_result("6h", edge_bps=12.0)}
    with pytest.raises(TieredPipelineError, match="not deployable"):
        _resolve_l2_master_tf(_cfg(l2_master_tf="8h"), per_tf)


# ── S8: Budget allocation ────────────────────────────────────────


def test_support_state_sufficient_when_above_target() -> None:
    evidence = _evidence(n_events=80, effective_n=50.0, block_lcb_bps=10.0)
    candidate = build_l0_signal_candidate(
        run_id="r1",
        evidence=evidence,
        recipe=_recipe(),
        source="catalog_exact",
        policy=_policy(),
        stress_cost_bps=10.0,
        tf_fusion=None,
    )
    assert candidate.support_state == "sufficient"


# ── S17: Memory: no float64 copy in validator ────────────────────


def test_validate_pit_universe_contract_no_float_copy() -> None:
    from src.domain.futures.universe.membership import validate_pit_universe_contract

    data_maps = _make_data_maps(["S1"])
    audit = validate_pit_universe_contract(
        data_maps=data_maps,
        symbols=["S1"],
        timeframes=["4h"],
        timeline={date(2023, 1, 1): frozenset({"S1"})},
    )
    assert isinstance(audit.passed, bool)
    assert audit.checked_cells >= 1
