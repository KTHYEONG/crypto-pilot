from __future__ import annotations

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
    universe_service.discover_universe_timeline(
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


def test_discover_universe_timeline_uses_stage6_for_inference_membership(
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

    assert result.symbols == ("BTCUSDT",)
    assert result.inference_symbols == ("BTCUSDT",)
    assert result.inference_panel_quarter_membership[date(2025, 1, 1)] == frozenset({"BTCUSDT"})
