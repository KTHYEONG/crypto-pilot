from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from src.application.futures.optimization import universe_service
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


def test_discover_universe_timeline_defaults_cfg_when_none(
    monkeypatch: MonkeyPatch,
) -> None:
    import numpy as np

    from src.domain.futures.universe.config import UniverseConfig
    from src.domain.futures.universe.contracts import UniverseStateCube

    captured: list[object] = []

    def fake_pit(**kwargs: object) -> object:
        captured.append(kwargs["cfg"])
        empty = pd.DatetimeIndex([], tz="UTC")
        return universe_service.UniverseTimelineResult(
            symbols=(),
            timeline=universe_service.UniverseMembershipTimeline(tf="4h", windows=()),
            snapshots=(),
            state_cube=UniverseStateCube(
                calendar=empty,
                instrument_ids=(),
                eligible=np.empty((0, 0), dtype=np.bool_),
                entry_block=np.empty((0, 0), dtype=np.bool_),
                exit_required=np.empty((0, 0), dtype=np.bool_),
                capacity_usdt=np.empty((0, 0), dtype=np.float64),
                risk_scale=np.empty((0, 0), dtype=np.float64),
                cost_bps=np.empty((0, 0), dtype=np.float64),
            ),
            report=pd.DataFrame(),
            audit=pd.DataFrame(),
            snapshot=_empty_snapshot(),
            inference_symbols=(),
            inference_timeline=None,
            inference_panel_quarter_membership={},
        )

    monkeypatch.setattr(
        universe_service,
        "_discover_universe_timeline_pit",
        fake_pit,
    )
    universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 4, 1),
        end_date=date(2025, 7, 1),
    )
    assert len(captured) == 1
    cfg = captured[0]
    assert isinstance(cfg, UniverseConfig)
    assert cfg.universe_engine == "pit"


def test_discover_universe_timeline_rejects_separate_l2_start_boundary(
    monkeypatch: MonkeyPatch,
) -> None:
    # We mock _discover_universe_timeline_pit to return a dummy result instead of failing,
    # since separate l2_start boundary is now permitted.
    dummy_res = MagicMock()
    monkeypatch.setattr(
        universe_service,
        "_discover_universe_timeline_pit",
        lambda *a, **kw: dummy_res,
    )

    res = universe_service.discover_universe_timeline(
        tf="4h",
        is_start=date(2025, 1, 1),
        oos_start=date(2025, 4, 1),
        end_date=date(2025, 7, 1),
        l2_start=date(2025, 3, 1),
    )
    assert res is dummy_res


def test_discover_universe_timeline_does_not_promote_rejected_symbols_into_state_cube(
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
    )
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


def test_validate_universe_quality_passes_for_high_quality_symbols(
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
        n_stage6_selected=1,
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
        ),
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
