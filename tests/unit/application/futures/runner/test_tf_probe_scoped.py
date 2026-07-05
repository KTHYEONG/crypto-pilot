from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.runner.active_pipeline import (
    DataStageResult,
    TfProbeStageResult,
)
from src.application.futures.runner.tf_probe_scoped import _run_tf_probe_stage_scoped


def _make_ohlcv_df(n: int = 2000) -> pd.DataFrame:
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": rng.random(n) * 100,
            "high": rng.random(n) * 100,
            "low": rng.random(n) * 100,
            "close": rng.random(n) * 100,
            "volume": rng.random(n) * 1000,
            "datetime": idx,
        },
        index=idx,
    )


def make_data_stage_with_maps(symbols: tuple[str, ...]) -> DataStageResult:
    data_maps = {
        sym: {"1h": _make_ohlcv_df(n=2000), "4h": _make_ohlcv_df(n=500)}
        for sym in symbols
    }
    return DataStageResult(
        data_maps=data_maps,
        oos_data_maps={s: {} for s in symbols},
        valid_symbols=list(symbols),
    )


def make_run_config(phase: str = "l1") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=3,
        phase=phase,  # type: ignore[arg-type]
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
    )


def _disable_tiered(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    monkeypatch.setitem(OPT_FUTURES_CONFIG, "USE_CS_RANK_ENGINE", False)


# ---------------------------------------------------------------------------
# Scenario 1 (Happy Path)
# ---------------------------------------------------------------------------


class TestTfProbeScopedHappyPath:
    def test_tf_probe_scoped_runs_before_data_clear(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify _run_tf_probe_stage_scoped is called before data_stage.data_maps.clear()."""
        from src.application.futures.runner.active_pipeline import _run_strategy_stage

        _disable_tiered(monkeypatch)
        order: list[str] = []
        cfg = make_run_config("l1")

        class _TrackClear(dict[str, Any]):
            def clear(self) -> None:
                order.append("clear")
                super().clear()

        ds = DataStageResult(
            data_maps=_TrackClear(),
            oos_data_maps=_TrackClear(),
            valid_symbols=["BTCUSDT", "ETHUSDT"],
        )

        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_tf_probe_stage_scoped",
            side_effect=lambda *a, **kw: order.append("probe"),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline.run_active_strategy_output_bridge",
            return_value=MagicMock(),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._inject_universe_metadata_into_maps",
        )

        _run_strategy_stage(
            run_config=cfg,
            window=MagicMock(),
            data_stage=ds,
            trading_symbols=("BTCUSDT", "ETHUSDT"),
        )

        assert "probe" in order, "probe should be called"
        assert "clear" in order, "clear should be called"
        probe_idx = order.index("probe")
        clear_idx = order.index("clear")
        assert probe_idx < clear_idx, (
            f"probe at {probe_idx} should precede clear at {clear_idx}; "
            f"full order={order}"
        )

    def test_tf_probe_scoped_uses_independent_cfg(
        self, mocker: MockerFixture
    ) -> None:
        """Verify probe_cfg is independent from tiered_cfg (different objects)."""
        from src.application.futures.runner.active_pipeline import (
            _run_strategy_stage,
        )
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        cfg_calls: list[object] = []

        def _track_build(*a: Any, **kw: Any) -> MagicMock:
            obj = MagicMock()
            obj.candidate = CandidateStrategyConfig()
            cfg_calls.append(obj.candidate)
            return obj

        mocker.patch(
            "src.application.futures.runner.active_pipeline.build_candidate_strategy_config",
            side_effect=_track_build,
        )
        captured_probe_cfg: dict[str, object] = {}

        def _capture_probe_call(*a: Any, **kw: Any) -> None:
            captured_probe_cfg["cfg"] = kw.get("probe_cfg")

        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_tf_probe_stage_scoped",
            side_effect=_capture_probe_call,
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_tradeable_scope",
            return_value=MagicMock(admitted=["BTCUSDT"], dropped_by_reason={}),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_base_symbol_scope",
            return_value=["BTCUSDT"],
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_layered_window",
            return_value=MagicMock(),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline.run_active_strategy_output_bridge",
            return_value=MagicMock(),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._inject_universe_metadata_into_maps",
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_universe_state_cube",
            return_value=None,
        )
        mocker.patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=MagicMock(datetimes=[pd.Timestamp("2024-01-01")]),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._tiered_labeled_events",
            return_value=pd.DataFrame(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.run_tiered_pipeline",
            return_value=(MagicMock(gate_passed=True), None, None),
        )

        cfg = make_run_config("l1")
        ds = make_data_stage_with_maps(("BTCUSDT",))

        _run_strategy_stage(
            run_config=cfg,
            window=MagicMock(),
            data_stage=ds,
            trading_symbols=("BTCUSDT",),
        )

        probe_cfg = captured_probe_cfg.get("cfg")
        assert probe_cfg is not None, "probe_cfg was not captured"
        assert len(cfg_calls) >= 2, (
            f"expected >=2 build_candidate_strategy_config calls, "
            f"got {len(cfg_calls)}"
        )
        cfg_0 = cfg_calls[0]
        cfg_1 = cfg_calls[1]
        assert id(cfg_0) != id(cfg_1), (
            "probe_cfg and tiered_cfg must be different objects"
        )
        assert cfg_0 is probe_cfg, (
            "first build_candidate_strategy_config call should be probe_cfg"
        )


# ---------------------------------------------------------------------------
# Scenario 2 (Edge Cases - Validation/Bounds)
# ---------------------------------------------------------------------------


class TestTfProbeScopedEdgeCases:
    def test_tf_probe_scoped_defaults_to_major_symbols_only(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scope_symbols=None -> probe_timeframe_alpha gets _TF_PROBE_FALLBACK_SYMBOLS."""
        from src.application.futures.runner.tf_probe_scoped import (
            _TF_PROBE_FALLBACK_SYMBOLS,
        )
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
        from src.domain.futures.strategy.timeframe_probe import (
            TfProbeManifest,
        )

        monkeypatch.setitem(OPT_FUTURES_CONFIG, "ENABLE_TF_PROBE", True)

        mock_probe = mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.probe_timeframe_alpha",
            return_value=TfProbeManifest(
                cells=(),
                tf_grid=("4h",),
                coverage_by_tf={"4h": 0},
                diversity_corr={},
            ),
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.select_tf_family_cells",
            return_value=(),
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.summarize_tf_probe_gate_audit",
            return_value=(),
        )

        _run_tf_probe_stage_scoped(
            run_config=make_run_config("l1"),
            full_strategy_maps={},
            probe_cfg=MagicMock(),
            scope_symbols=None,
        )

        mock_probe.assert_called_once()
        call_kwargs = mock_probe.call_args.kwargs
        assert "symbols" in call_kwargs, "probe_timeframe_alpha called without symbols"
        assert call_kwargs["symbols"] == list(_TF_PROBE_FALLBACK_SYMBOLS), (
            f"expected symbols={list(_TF_PROBE_FALLBACK_SYMBOLS)}, "
            f"got {call_kwargs['symbols']}"
        )

    def test_tf_probe_scoped_l3_phase_receives_populated_data(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L3 phase (no data clear) — probe is still called."""
        from src.application.futures.runner.active_pipeline import _run_strategy_stage

        _disable_tiered(monkeypatch)
        order: list[str] = []
        cfg = make_run_config("l3")

        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_tf_probe_stage_scoped",
            side_effect=lambda *a, **kw: order.append("probe"),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline.run_active_strategy_output_bridge",
            return_value=MagicMock(),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._inject_universe_metadata_into_maps",
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_candidate_evaluation_report",
            return_value=MagicMock(),
        )
        ds = make_data_stage_with_maps(("BTCUSDT", "ETHUSDT"))

        _run_strategy_stage(
            run_config=cfg,
            window=MagicMock(),
            data_stage=ds,
            trading_symbols=("BTCUSDT", "ETHUSDT"),
        )

        assert "probe" in order, "probe should be called in L3 phase"

    def test_tf_probe_scoped_l1_phase_receives_populated_data(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L1 phase: probe returns non-empty result from full_strategy_maps (bug D regress)."""
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
        from src.domain.futures.strategy.timeframe_probe import (
            TfCellEvidence,
            TfProbeManifest,
        )

        monkeypatch.setitem(OPT_FUTURES_CONFIG, "ENABLE_TF_PROBE", True)

        cell = TfCellEvidence(
            symbol="BTCUSDT", family="dual_momentum", variant="trend",
            archetype="trend", tf="4h", n_obs=100, n_events=50,
            ic_mean=0.05, ic_tstat_hac=2.5, ic_fold_sign_consistency=0.8,
            alpha_half_life_h=48.0, net_edge_bps=15.0, turnover_per_year=12.0,
            vr_label="trend", hurst=0.6, passed_fdr=True,
        )
        manifest = TfProbeManifest(
            cells=(cell,), tf_grid=("4h",),
            coverage_by_tf={"4h": 1}, diversity_corr={},
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.probe_timeframe_alpha",
            return_value=manifest,
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.select_tf_family_cells",
            return_value=(cell,),
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.summarize_tf_probe_gate_audit",
            return_value=(),
        )

        result = _run_tf_probe_stage_scoped(
            run_config=make_run_config("l1"),
            full_strategy_maps={"BTCUSDT": {"4h": MagicMock()}},
            probe_cfg=MagicMock(),
            scope_symbols=("BTCUSDT",),
        )

        assert result is not None, "probe should return TfProbeStageResult"
        assert len(result.manifest.cells) > 0, (
            "manifest should not be empty — probe uses pre-captured "
            "full_strategy_maps that survives data_maps.clear() (bug D)"
        )
        assert result.manifest.cells[0].symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# Scenario 3 (Error Handling - Exceptions)
# ---------------------------------------------------------------------------


class TestTfProbeScopedErrorHandling:
    def test_tf_probe_scoped_oom_guard_rejects_large_scope(
        self, mocker: MockerFixture
    ) -> None:
        """scope_symbols > 20 symbols raises ValueError (OOM guard)."""
        large_scope = tuple(f"SYM{i}" for i in range(25))

        with pytest.raises(ValueError, match=r"scope_symbols.*20|20.*symbols"):
            _run_tf_probe_stage_scoped(
                run_config=make_run_config("l1"),
                full_strategy_maps={},
                probe_cfg=MagicMock(),
                scope_symbols=large_scope,
            )

    def test_tf_probe_scoped_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENABLE_TF_PROBE=False -> _run_tf_probe_stage_scoped returns None."""
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

        monkeypatch.setitem(OPT_FUTURES_CONFIG, "ENABLE_TF_PROBE", False)

        result = _run_tf_probe_stage_scoped(
            run_config=make_run_config("l1"),
            full_strategy_maps={},
            probe_cfg=MagicMock(),
            scope_symbols=("BTCUSDT",),
        )
        assert result is None

    def test_tf_probe_scoped_probe_exception_fallback(
        self, mocker: MockerFixture
    ) -> None:
        """probe_timeframe_alpha exception -> _run_tf_probe_stage_scoped returns None."""

        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.probe_timeframe_alpha",
            side_effect=RuntimeError("probe crashed"),
        )

        result = _run_tf_probe_stage_scoped(
            run_config=make_run_config("l1"),
            full_strategy_maps={"BTCUSDT": {"1h": MagicMock()}},
            probe_cfg=MagicMock(),
            scope_symbols=("BTCUSDT",),
        )
        assert result is None

    def test_tf_probe_scoped_worker_exception_isolated(
        self, mocker: MockerFixture
    ) -> None:
        """Worker exception isolation: partial cells survive as TfProbeStageResult."""
        from src.domain.futures.strategy.timeframe_probe import (
            TfCellEvidence,
            TfProbeManifest,
        )

        surviving_cell = TfCellEvidence(
            symbol="BTCUSDT", family="dual_momentum", variant="trend",
            archetype="trend", tf="4h", n_obs=100, n_events=50,
            ic_mean=0.05, ic_tstat_hac=2.5, ic_fold_sign_consistency=0.8,
            alpha_half_life_h=48.0, net_edge_bps=15.0, turnover_per_year=12.0,
            vr_label="trend", hurst=0.6, passed_fdr=True,
        )
        manifest = TfProbeManifest(
            cells=(surviving_cell,),
            tf_grid=("4h", "8h"),
            coverage_by_tf={"4h": 1, "8h": 0},
            diversity_corr={},
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.probe_timeframe_alpha",
            return_value=manifest,
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.select_tf_family_cells",
            return_value=(surviving_cell,),
        )
        mocker.patch(
            "src.domain.futures.strategy.timeframe_probe.summarize_tf_probe_gate_audit",
            return_value=(),
        )

        result = _run_tf_probe_stage_scoped(
            run_config=make_run_config("l1"),
            full_strategy_maps={"BTCUSDT": {"1h": MagicMock()}},
            probe_cfg=MagicMock(),
            scope_symbols=("BTCUSDT",),
        )

        assert result is not None, "partial result should survive worker failures"
        assert isinstance(result, TfProbeStageResult)
        assert len(result.winning_cells) > 0, "at least one cell should survive"
        assert result.winning_cells[0] is surviving_cell
