from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from src.application.futures.optimization import universe_service
from src.application.futures.optimization.universe_service import (
    UniverseMembershipTimeline,
)
from src.domain.futures.universe.models import SymbolMeta, UniverseSnapshot


def _empty_snapshot() -> UniverseSnapshot:
    return UniverseSnapshot(
        as_of="2025-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg",
        data_manifest_hash="manifest",
        basket_ref=(),
        basket_weights=(),
        selected=(),
        rejected={},
        generated_at_utc="2025-01-01T00:00:00Z",
        ledger_confidence="high",
        n_stage0=0,
        n_stage1_pass=0,
        n_stage2_pass=0,
        n_stage3_pass=0,
        n_stage4_pass=0,
        n_stage5_pass=0,
        n_stage6_selected=0,
    )


def _snapshot_with_selected() -> UniverseSnapshot:
    return UniverseSnapshot(
        as_of="2025-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg",
        data_manifest_hash="manifest",
        basket_ref=(),
        basket_weights=(),
        selected=(
            SymbolMeta(
                symbol="BTCUSDT",
                role="core",
                adv_usdt=1.0,
                execution_cost_bps=1.0,
                funding_carry_8h=0.0,
                beta_vs_market=1.0,
                cluster_id=0,
                tradeable_rank=1,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(1.0,),
            ),
        ),
        rejected={},
        generated_at_utc="2025-01-01T00:00:00Z",
        ledger_confidence="high",
        n_stage0=1,
        n_stage1_pass=1,
        n_stage2_pass=1,
        n_stage3_pass=1,
        n_stage4_pass=1,
        n_stage5_pass=1,
        n_stage6_selected=1,
        training_panel=("BTCUSDT",),
        live_inference_panel=("BTCUSDT",),
        historical_trading_panel=("BTCUSDT", "ETHUSDT"),
        inference_panel_quarter_membership={
            date(2025, 1, 1): ("BTCUSDT",),
        },
        stage5_research_panel=("BTCUSDT", "ETHUSDT"),
    )


def test_discover_universe_timeline_uses_previous_selection(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...] | None] = []
    symbols_by_call = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("BTCUSDT",),
    ]

    def _fake_discover(**kwargs: object) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
        calls.append(cast(tuple[str, ...] | None, kwargs.get("previous_selection")))
        idx = len(calls) - 1
        return symbols_by_call[idx], _empty_snapshot(), pd.DataFrame()

    monkeypatch.setattr(universe_service, "_discover_symbols_via_universe", _fake_discover)
    monkeypatch.setattr(universe_service, "_UNIVERSE_AUDIT_DIR", tmp_path)
    result = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 4, 1),
        end_date=date(2025, 7, 1),
        force_rebuild=False,
    )

    assert calls == [
        None,
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
    ]
    assert result.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert isinstance(result.timeline, UniverseMembershipTimeline)
    assert len(result.timeline.windows) == 3
    assert result.timeline.windows[0].active_symbols == ("BTCUSDT", "ETHUSDT")
    assert result.timeline.windows[0].entry_symbols == ("BTCUSDT", "ETHUSDT")
    assert result.timeline.windows[0].exit_symbols == ()
    assert result.timeline.windows[1].active_symbols == ("BTCUSDT", "SOLUSDT")
    assert result.timeline.windows[1].entry_symbols == ("SOLUSDT",)
    assert result.timeline.windows[1].exit_symbols == ("ETHUSDT",)
    assert result.timeline.windows[2].active_symbols == ("BTCUSDT",)
    assert result.timeline.windows[2].entry_symbols == ()
    assert result.timeline.windows[2].exit_symbols == ("SOLUSDT",)
    assert result.timeline.windows[0].effective_to == result.timeline.windows[1].effective_from
    assert result.timeline.windows[2].effective_to is None
    assert len(result.snapshots) == 3
    assert result.state_cube.instrument_ids == result.symbols
    assert result.state_cube.eligible.shape[1] == len(result.symbols)
    assert not result.audit.empty
    assert (tmp_path / "universe_timeline.parquet").exists()
    assert (tmp_path / "membership_state.parquet").exists()


def test_discover_universe_timeline_writes_membership_state_columns(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_discover(**_kwargs: object) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
        return ("BTCUSDT",), _snapshot_with_selected(), pd.DataFrame()

    monkeypatch.setattr(universe_service, "_discover_symbols_via_universe", _fake_discover)
    monkeypatch.setattr(universe_service, "_UNIVERSE_AUDIT_DIR", tmp_path)
    result = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        force_rebuild=False,
    )
    membership = pd.read_parquet(tmp_path / "membership_state.parquet")
    expected_cols = {
        "quarter_start",
        "symbol",
        "is_selected",
        "selection_reason",
        "rank",
        "dwell_days",
        "was_prev_member",
    }
    assert expected_cols.issubset(set(membership.columns))
    assert result.snapshot.historical_trading_panel == ("BTCUSDT", "ETHUSDT")


def test_discover_universe_timeline_uses_separate_trading_and_inference_membership(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_discover(**_kwargs: object) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
        return ("BTCUSDT",), _snapshot_with_selected(), pd.DataFrame()

    monkeypatch.setattr(universe_service, "_discover_symbols_via_universe", _fake_discover)
    monkeypatch.setattr(universe_service, "_UNIVERSE_AUDIT_DIR", tmp_path)
    result = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        force_rebuild=False,
    )

    assert result.symbols == ("BTCUSDT", "ETHUSDT")
    assert result.inference_symbols == ("BTCUSDT",)
    assert result.inference_panel_quarter_membership[date(2025, 1, 1)] == frozenset({"BTCUSDT"})
    assert result.snapshot.inference_panel == ("BTCUSDT",)
    assert result.snapshot.historical_trading_panel == ("BTCUSDT", "ETHUSDT")
    assert result.timeline.windows[0].active_symbols == ("BTCUSDT", "ETHUSDT")
    assert result.inference_timeline is not None


def test_discover_universe_timeline_preserves_min_history_bars_signature(
    monkeypatch: MonkeyPatch,
) -> None:
    def _fake_discover(**_kwargs: object) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
        return ("BTCUSDT",), _empty_snapshot(), pd.DataFrame()

    monkeypatch.setattr(universe_service, "_discover_symbols_via_universe", _fake_discover)
    result = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        force_rebuild=False,
        min_history_bars=100,
    )

    assert result.symbols == ("BTCUSDT",)


def test_discover_universe_timeline_does_not_promote_rejected_symbols_into_state_cube(
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = replace(_snapshot_with_selected(), historical_trading_panel=())
    fake_report = pd.DataFrame({"symbol": ["ETHUSDT"], "stage": ["stage6"], "passed": [False]})

    monkeypatch.setattr(
        universe_service,
        "load_or_build_universe_snapshot",
        lambda **_kwargs: (snapshot, pd.DataFrame(columns=["symbol"]), fake_report),
    )
    discovered_symbols, _, _ = universe_service._discover_symbols_via_universe(
        tf="4h",
        reference_date="2025-04-01",
        force_rebuild=False,
        previous_selection=None,
    )

    assert discovered_symbols == ("BTCUSDT",)


def test_discover_universe_timeline_state_cube_uses_symbol_meta_for_cost_and_capacity(
    monkeypatch: MonkeyPatch,
) -> None:
    def _fake_discover(**_kwargs: object) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
        return ("BTCUSDT",), _snapshot_with_selected(), pd.DataFrame()

    monkeypatch.setattr(universe_service, "_discover_symbols_via_universe", _fake_discover)
    result = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        force_rebuild=False,
    )

    assert float(result.state_cube.cost_bps[0, 0]) == 1.0
    assert float(result.state_cube.capacity_usdt[0, 0]) == 1.0


def test_validate_universe_quality_prefers_historical_trading_panel_symbol_scope(
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = UniverseSnapshot(
        as_of="2025-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg",
        data_manifest_hash="manifest",
        basket_ref=(),
        basket_weights=(),
        rejected={},
        generated_at_utc="2025-01-01T00:00:00Z",
        ledger_confidence="high",
        n_stage0=1,
        n_stage1_pass=1,
        n_stage2_pass=1,
        n_stage3_pass=1,
        n_stage4_pass=1,
        n_stage5_pass=1,
        n_stage6_selected=2,
        selected=(
            SymbolMeta(
                symbol="BTCUSDT",
                role="core",
                adv_usdt=30_000_000.0,
                execution_cost_bps=10.0,
                funding_carry_8h=0.0,
                beta_vs_market=1.0,
                cluster_id=0,
                tradeable_rank=1,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(10_000.0,),
            ),
            SymbolMeta(
                symbol="ETHUSDT",
                role="core",
                adv_usdt=1_000_000.0,
                execution_cost_bps=200.0,
                funding_carry_8h=0.0,
                beta_vs_market=1.0,
                cluster_id=1,
                tradeable_rank=2,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(1_000.0,),
            ),
        ),
        historical_trading_panel=("BTCUSDT",),
    )

    monkeypatch.setattr(
        universe_service,
        "load_universe_snapshot",
        lambda **_kwargs: None,
    )
    assert universe_service.validate_universe_quality(
        snapshot=snapshot,
        report=pd.DataFrame(),
        reference_date="2025-01-01",
        tf="4h",
    )
