from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.runner import active_pipeline as opt_main_futures
from src.domain.futures.strategy.timeframe_probe import TfCellEvidence, TfProbeManifest


def _window() -> opt_main_futures.QuarterlyWindow:
    return opt_main_futures.QuarterlyWindow(
        fetch_start="2023-01-01",
        is_start="2024-01-01",
        oos_start="2025-10-01",
        end_date="2026-03-31",
        fetch_start_date=datetime.strptime("2023-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2024-01-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2025-10-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-03-31", "%Y-%m-%d").date(),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=8, freq="4h", tz="UTC"),
            "open": [1.0] * 8,
            "high": [2.0] * 8,
            "low": [0.5] * 8,
            "close": [1.5] * 8,
            "volume": [100.0] * 8,
        }
    )


def test_run_strategy_stage_injects_probe_cells_before_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": _frame()}},
        oos_data_maps={"BTCUSDT": {"4h": _frame()}},
        valid_symbols=["BTCUSDT"],
    )
    captured: dict[str, Any] = {}

    monkeypatch.setitem(
        cast(dict[str, Any], opt_main_futures.__dict__["OPT_FUTURES_CONFIG"]),
        "USE_CS_RANK_ENGINE",
        True,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_resolve_layered_window",
        lambda *_args, **_kwargs: SimpleNamespace(
            fetch_start=datetime.strptime("2023-01-01", "%Y-%m-%d").date(),
            holdout_start=datetime.strptime("2025-10-01", "%Y-%m-%d").date(),
            holdout_end=datetime.strptime("2026-03-31", "%Y-%m-%d").date(),
        ),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_resolve_base_symbol_scope",
        lambda **_kwargs: ("BTCUSDT",),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_resolve_tradeable_scope",
        lambda **_kwargs: opt_main_futures.TradeableScopeResult(
            admitted=("BTCUSDT",),
            dropped_by_reason={
                "missing_map": (),
                "empty_frame": (),
                "late_start": (),
                "min_bars": (),
                "no_holdout": (),
                "holdout_coverage": (),
            },
        ),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "pick_strategy_data_maps",
        lambda **_kwargs: {"BTCUSDT": {"4h": _frame()}},
    )
    _cell = TfCellEvidence(
        symbol="BTCUSDT",
        family="carry_rev",
        variant="funding_carry",
        archetype="carry_rev",
        tf="1h",
        n_obs=100,
        n_events=20,
        ic_mean=0.05,
        ic_tstat_hac=2.5,
        ic_fold_sign_consistency=0.8,
        alpha_half_life_h=24.0,
        net_edge_bps=5.0,
        turnover_per_year=50.0,
        vr_label="mean_rev",
        hurst=0.4,
        passed_fdr=True,
    )
    monkeypatch.setattr(
        "src.application.futures.runner.active_pipeline._run_tf_probe_stage_scoped",
        lambda *_args, **_kwargs: opt_main_futures.TfProbeStageResult(
            manifest=TfProbeManifest(
                cells=(_cell,),
                tf_grid=("1h",),
                coverage_by_tf={"1h": 8},
                diversity_corr={},
            ),
            winning_cells=(_cell,),
            selected_tfs=frozenset({"1h"}),
        ),
    )

    def _fake_bridge(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            aligned=SimpleNamespace(
                symbols=("BTCUSDT",),
                datetimes=pd.date_range("2023-01-01", "2026-03-31", freq="4d").to_numpy(),
            ),
            labeled_unfiltered=pd.DataFrame({"l0_recipe_id": [""]}),
            labeled=None,
            l0_delivery_manifest=None,
        )

    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        _fake_bridge,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.common.alignment.align_data_maps",
        lambda *_args, **_kwargs: SimpleNamespace(
            datetimes=pd.date_range("2023-01-01", "2026-03-31", freq="4d").to_numpy()
        ),
    )
    def _fake_run_tiered_pipeline(**kwargs: Any) -> tuple[SimpleNamespace, None, None]:
        captured["extra_probe_cells"] = kwargs.get("probe_manifest")
        return SimpleNamespace(gate_passed=True), None, None

    monkeypatch.setattr(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_tiered_pipeline",
        _fake_run_tiered_pipeline,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_tiered_labeled_events",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    result = opt_main_futures._run_strategy_stage(
        cast(FuturesRunConfig, SimpleNamespace(timeframe="4h", phase="l1", date="2026-06-22")),
        _window(),
        data_stage,
        universe_snapshot=None,
        layered_window=None,
        universe_result=None,
    )

    assert captured["extra_probe_cells"] is not None
    assert len(captured["extra_probe_cells"]) == 1
    assert result is not None


def test_run_tf_probe_stage_logs_selection_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": _frame()}},
        oos_data_maps={"BTCUSDT": {"4h": _frame()}},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setitem(
        cast(dict[str, Any], opt_main_futures.__dict__["OPT_FUTURES_CONFIG"]),
        "ENABLE_TF_PROBE",
        True,
    )
    monkeypatch.setitem(
        cast(dict[str, Any], opt_main_futures.__dict__["OPT_FUTURES_CONFIG"]),
        "TF_PROBE_GRID",
        ["1h", "2h"],
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.timeframe_probe.probe_timeframe_alpha",
        lambda **_kwargs: TfProbeManifest(
            cells=(
                TfCellEvidence(
                    symbol="BTCUSDT",
                    family="carry_rev",
                    variant="funding_carry",
                    archetype="carry_rev",
                    tf="1h",
                    n_obs=100,
                    n_events=20,
                    ic_mean=0.05,
                    ic_tstat_hac=2.5,
                    ic_fold_sign_consistency=0.8,
                    alpha_half_life_h=24.0,
                    net_edge_bps=5.0,
                    turnover_per_year=50.0,
                    vr_label="mean_rev",
                    hurst=0.4,
                    passed_fdr=True,
                ),
                TfCellEvidence(
                    symbol="BTCUSDT",
                    family="trend",
                    variant="breakout",
                    archetype="trend",
                    tf="2h",
                    n_obs=120,
                    n_events=25,
                    ic_mean=0.06,
                    ic_tstat_hac=2.1,
                    ic_fold_sign_consistency=0.75,
                    alpha_half_life_h=30.0,
                    net_edge_bps=4.0,
                    turnover_per_year=40.0,
                    vr_label="trend",
                    hurst=0.5,
                    passed_fdr=True,
                ),
            ),
            tf_grid=("1h", "2h"),
            coverage_by_tf={"1h": 1, "2h": 1},
            diversity_corr={},
        ),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.timeframe_probe.select_tf_family_cells",
        lambda manifest, **_kwargs: tuple(manifest.cells),
    )

    messages: list[str] = []

    def _fake_info(msg: str, *args: Any, **kwargs: Any) -> None:
        del kwargs
        messages.append(msg % args if args else msg)

    monkeypatch.setattr(opt_main_futures._logger, "info", _fake_info)

    result = opt_main_futures._run_tf_probe_stage(
        cast(FuturesRunConfig, SimpleNamespace(timeframe="4h")),
        data_stage,
        object(),
    )
    out = "\n".join(messages)

    assert result is not None
    assert "[L0-PROBE]" in out
    assert "1h" in out
    assert "2h" in out
