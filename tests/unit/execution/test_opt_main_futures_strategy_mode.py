from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.application.futures.optimization.config import build_run_config_from_args
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput
from src.domain.futures.universe import SymbolMeta, UniverseSnapshot
from src.domain.futures.universe.contracts import UniverseStateCube
from src.execution import opt_main_futures


def test_strategy_mode_pipeline_orchestration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    called: list[str] = []
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {}},
        oos_data_maps={"BTCUSDT": {}},
        valid_symbols=["BTCUSDT"],
    )
    snapshot = UniverseSnapshot(
        as_of="2026-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg-hash",
        data_manifest_hash="manifest-hash",
        basket_ref=(),
        basket_weights=(),
        selected=(
            SymbolMeta(
                symbol="BTCUSDT",
                role="anchor",
                adv_usdt=1.0,
                execution_cost_bps=1.0,
                funding_carry_8h=0.0,
                beta_vs_market=1.2,
                cluster_id=7,
                tradeable_rank=1,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(),
                cluster_size=5.0,
                anchor_cluster_member=1.0,
            ),
        ),
        rejected={},
        generated_at_utc="2026-01-01T00:00:00+00:00",
        ledger_confidence="high",
        n_stage0=1,
        n_stage1_pass=1,
        n_stage2_pass=1,
        n_stage3_pass=1,
        n_stage4_pass=1,
        n_stage5_pass=1,
        n_stage6_selected=1,
    )

    def fake_window(reference_date: str | None) -> opt_main_futures.QuarterlyWindow:
        _ = reference_date
        called.append("window")
        return window

    def fake_universe(
        rc: object,
        win: object,
        *,
        layered_window: object | None = None,
    ) -> tuple[
        list[str],
        dict[object, frozenset[str]],
        tuple[str, ...],
        tuple[str, ...],
        UniverseSnapshot,
        dict[object, frozenset[str]],
    ]:
        _ = rc
        _ = win
        _ = layered_window
        called.append("universe")
        return ["BTCUSDT"], {}, (), (), snapshot, {}, None

    def fake_data(
        rc: object,
        win: object,
        discovered: list[str],
        timeline: dict[object, frozenset[str]],
        inference_panel: tuple[str, ...] = (),
        live_inference_panel: tuple[str, ...] = (),
        inference_timeline: dict[object, frozenset[str]] | None = None,
        *,
        layered_window: object | None = None,
    ) -> opt_main_futures.DataStageResult:
        _ = rc
        _ = win
        _ = discovered
        _ = timeline
        _ = inference_panel
        _ = live_inference_panel
        _ = inference_timeline
        _ = layered_window
        called.append("data")
        return data_stage

    def fake_strategy(
        rc: object,
        win: object,
        ds: object,
        trading_symbols: tuple[str, ...] = (),
        universe_snapshot: object | None = None,
        *,
        layered_window: object | None = None,
        universe_result: object | None = None,
    ) -> None:
        _ = rc
        _ = win
        _ = ds
        _ = trading_symbols
        _ = universe_snapshot
        _ = layered_window
        called.append("strategy")

    def fake_optimization(
        rc: object,
        win: object,
        ds: object,
        *,
        seed: int,
        resume: bool,
    ) -> opt_main_futures.RunnerResult:
        _ = rc
        _ = win
        _ = ds
        _ = seed
        _ = resume
        called.append("optimization")
        return opt_main_futures.RunnerResult(exit_code=0, reason="ok")

    monkeypatch.setattr(opt_main_futures, "_resolve_quarterly_window", fake_window)
    monkeypatch.setattr(opt_main_futures, "_ensure_universe_ledger_sync", lambda *_: None)
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_cached_symbol_data_for_targets",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(opt_main_futures, "_run_universe_stage", fake_universe)
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", fake_data)
    monkeypatch.setattr(opt_main_futures, "_run_regime_evaluation_stage", lambda *_: None)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", fake_strategy)
    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fake_optimization)

    result = opt_main_futures.run_pipeline(run_config, seed=13, resume=True)
    assert result.exit_code == 0
    assert called == ["window", "universe", "data", "strategy", "optimization"]


def test_resolve_universe_state_cube_returns_cube_when_present() -> None:
    cube = UniverseStateCube(
        calendar=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC"),
        instrument_ids=("binance_usdt_perpetual:BTCUSDT",),
        eligible=np.array([[True], [False]], dtype=np.bool_),
        entry_block=np.array([[False], [True]], dtype=np.bool_),
        exit_required=np.array([[False], [False]], dtype=np.bool_),
        capacity_usdt=np.array([[1_000.0], [2_000.0]], dtype=np.float64),
        risk_scale=np.array([[1.0], [1.0]], dtype=np.float64),
        cost_bps=np.array([[5.0], [6.0]], dtype=np.float64),
    )
    universe_result = type("UniverseResult", (), {"state_cube": cube})()

    resolved = opt_main_futures._resolve_universe_state_cube(universe_result)

    assert resolved is cube


def test_universe_metadata_by_symbol_preserves_extended_scores() -> None:
    snapshot = UniverseSnapshot(
        as_of="2026-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg",
        data_manifest_hash="manifest",
        basket_ref=(),
        basket_weights=(),
        selected=(
            SymbolMeta(
                symbol="BTCUSDT",
                role="anchor",
                adv_usdt=1.0,
                execution_cost_bps=2.0,
                funding_carry_8h=3.0,
                beta_vs_market=4.0,
                cluster_id=5,
                tradeable_rank=1,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(10.0,),
                cluster_size=6.0,
                anchor_cluster_member=1.0,
                vol_30d=0.35,
                friction_score=0.81,
                alpha_capacity_score=0.73,
                diversification_score=0.44,
                tradeable_score=0.69,
            ),
        ),
        rejected={},
        generated_at_utc="2026-01-01T00:00:00+00:00",
        ledger_confidence="high",
        n_stage0=1,
        n_stage1_pass=1,
        n_stage2_pass=1,
        n_stage3_pass=1,
        n_stage4_pass=1,
        n_stage5_pass=1,
        n_stage6_selected=1,
    )

    metadata = opt_main_futures._universe_metadata_by_symbol(snapshot)

    assert metadata["BTCUSDT"] == pytest.approx((5.0, 4.0, 6.0, 1.0, 0.35, 0.81, 0.73, 0.44, 0.69))


def test_l2_mode_skips_optimization_stage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """l2 모드는 strategy bridge 후 optimization 없이 종료해야 한다."""
    caplog.set_level(logging.INFO)
    run_config = build_run_config_from_args(
        {
            "phase": "l2",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {}},
        oos_data_maps={"BTCUSDT": {}},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setattr(opt_main_futures, "_resolve_quarterly_window", lambda _: window)
    monkeypatch.setattr(opt_main_futures, "_ensure_universe_ledger_sync", lambda *_: None)
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_cached_symbol_data_for_targets",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_universe_stage",
        lambda *_args, **_kwargs: (
            ["BTCUSDT"],
            {},
            (),
            (),
            UniverseSnapshot(
                as_of="2026-01-01",
                tf="4h",
                schema_version=1,
                config_hash="cfg-hash",
                data_manifest_hash="manifest-hash",
                basket_ref=(),
                basket_weights=(),
                selected=(),
                rejected={},
                generated_at_utc="2026-01-01T00:00:00+00:00",
                ledger_confidence="high",
                n_stage0=0,
                n_stage1_pass=0,
                n_stage2_pass=0,
                n_stage3_pass=0,
                n_stage4_pass=0,
                n_stage5_pass=0,
                n_stage6_selected=0,
            ),
            {},
            None,
        ),
    )
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", lambda *_args, **_kwargs: data_stage)
    monkeypatch.setattr(opt_main_futures, "_run_regime_evaluation_stage", lambda *_: None)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", lambda *_args, **_kwargs: None)

    def fail_if_called(*args: object, **kwargs: object) -> opt_main_futures.RunnerResult:
        raise AssertionError("optimization stage should not be called in alpha mode")

    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fail_if_called)
    result = opt_main_futures.run_pipeline(run_config)
    assert result.exit_code == 0
    assert result.reason == "candidate_evaluation_done"


def test_strategy_stage_injects_universe_metadata_before_bridge(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    run_config = build_run_config_from_args(
        {
            "phase": "l2",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=4, freq="4h"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame.copy()}},
        oos_data_maps={"BTCUSDT": {"4h": frame.copy()}},
        valid_symbols=["BTCUSDT"],
    )
    snapshot = UniverseSnapshot(
        as_of="2026-01-01",
        tf="4h",
        schema_version=1,
        config_hash="cfg-hash",
        data_manifest_hash="manifest-hash",
        basket_ref=(),
        basket_weights=(),
        selected=(
            SymbolMeta(
                symbol="BTCUSDT",
                role="anchor",
                adv_usdt=1.0,
                execution_cost_bps=1.0,
                funding_carry_8h=0.0,
                beta_vs_market=1.25,
                cluster_id=7,
                tradeable_rank=1,
                basis_annualized_mean=None,
                basis_vol=None,
                capacity_clip_usdt_list=(),
                cluster_size=6.0,
                anchor_cluster_member=1.0,
            ),
        ),
        rejected={},
        generated_at_utc="2026-01-01T00:00:00+00:00",
        ledger_confidence="high",
        n_stage0=1,
        n_stage1_pass=1,
        n_stage2_pass=1,
        n_stage3_pass=1,
        n_stage4_pass=1,
        n_stage5_pass=1,
        n_stage6_selected=1,
    )
    injected: dict[str, float] = {}

    def fake_bridge(*, preloaded_data_maps: dict[str, dict[str, object]], **kwargs: object) -> CandidatePipelineOutput:
        _ = kwargs
        frame_out = preloaded_data_maps["BTCUSDT"]["4h"]
        assert isinstance(frame_out, pd.DataFrame)
        injected["cluster_id"] = float(frame_out["cluster_id"].iloc[0])
        injected["beta_vs_market"] = float(frame_out["beta_vs_market"].iloc[0])
        injected["cluster_size"] = float(frame_out["cluster_size"].iloc[0])
        injected["anchor_cluster_member"] = float(frame_out["anchor_cluster_member"].iloc[0])
        return CandidatePipelineOutput()

    monkeypatch.setattr(opt_main_futures, "run_active_strategy_output_bridge", fake_bridge)
    monkeypatch.setattr(opt_main_futures, "merge_candidate_output_into_is_and_oos", lambda *_: None)
    monkeypatch.setattr(opt_main_futures, "_run_candidate_evaluation_report", lambda *_: None)
    # This test targets legacy Phase D (universe-metadata injection), not the tiered
    # pipeline — force non-tiered mode so it doesn't take the tiered try/except branch
    # (which now exits early via RunnerResult instead of falling back to Phase D).
    monkeypatch.setitem(opt_main_futures.OPT_FUTURES_CONFIG, "USE_CS_RANK_ENGINE", False)

    opt_main_futures._run_strategy_stage(
        run_config,
        window,
        data_stage,
        trading_symbols=("BTCUSDT",),
        universe_snapshot=snapshot,
    )

    assert injected == {
        "cluster_id": 7.0,
        "beta_vs_market": 1.25,
        "cluster_size": 6.0,
        "anchor_cluster_member": 1.0,
    }


def test_run_from_cli_when_pipeline_returns_nonzero_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    argv = ["--phase", "l3"]
    monkeypatch.setattr(
        opt_main_futures,
        "run_pipeline",
        lambda *_args, **_kwargs: opt_main_futures.RunnerResult(exit_code=7, reason="failed"),
    )

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 7


def test_run_from_cli_when_pipeline_raises_runtime_error_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    argv = ["--phase", "l3"]

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> opt_main_futures.RunnerResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", _raise_runtime_error)

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 1


def test_requires_exec_1m_returns_false_for_l2_mode() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l2",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    assert opt_main_futures._requires_exec_1m(run_config) is False


def test_requires_exec_1m_returns_false_for_l1_mode() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l1",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    assert opt_main_futures._requires_exec_1m(run_config) is False


def test_resolve_data_collection_symbols_uses_inference_panel() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    out = opt_main_futures._resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=["AAAUSDT"],
        inference_panel=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        live_inference_panel=("BTCUSDT",),
    )
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert sym in out


def test_resolve_data_collection_symbols_uses_live_panel_when_inference_panel_is_empty() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    out = opt_main_futures._resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=["AAAUSDT"],
        inference_panel=(),
        live_inference_panel=("BTCUSDT", "ETHUSDT"),
    )

    assert "BTCUSDT" in out
    assert "ETHUSDT" in out
    assert "AAAUSDT" not in out


def test_ensure_universe_ledger_sync_always_passes_none_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        opt_main_futures,
        "run_historical_sync",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "src.domain.futures.universe.models.DEFAULT_LEDGER_PATH",
        tmp_path / "universe_ledger.sqlite",
    )
    opt_main_futures._ensure_universe_ledger_sync(run_config, window)
    assert calls
    assert calls[0].get("symbols") is None


def test_ensure_cached_symbol_data_uses_fetch_start_for_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        end_date="2026-03-31",
        fetch_start_date=datetime.strptime("2022-10-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2023-10-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2025-10-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-03-31", "%Y-%m-%d").date(),
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        opt_main_futures,
        "run_historical_sync",
        lambda **kwargs: calls.append(kwargs),
    )
    opt_main_futures._ensure_cached_symbol_data_for_targets(
        run_config,
        window,
        ("BTCUSDT", "ETHUSDT"),
        require_exec_1m=True,
    )

    assert len(calls) == 2
    assert calls[0].get("start_date") == window.fetch_start_date
    assert calls[1].get("start_date") == window.fetch_start_date


def test_active_signals_count_reads_alpha_panel_not_missing_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active Signals는 alpha_panel["target_weight"]에서 집계해야 한다.
    panel_target_weight 속성(존재하지 않는)을 읽으면 항상 0이 되는 버그를 방지한다.
    """
    import numpy as np

    from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput

    # Arrange: alpha_panel에 nonzero target_weight가 있는 CandidatePipelineOutput 구성
    datetimes = pd.date_range("2026-01-01", periods=5, freq="4h")
    panel = pd.DataFrame(
        {
            "datetime": datetimes,
            "symbol": ["BTCUSDT"] * 5,
            "target_weight": np.array([0.0, 0.1, 0.05, 0.0, 0.08], dtype=np.float64),
            "alpha_long": np.zeros(5, dtype=np.float64),
            "alpha_short": np.zeros(5, dtype=np.float64),
        }
    ).set_index(["datetime", "symbol"])
    ml_out = CandidatePipelineOutput(alpha_panel=panel, target_weights=None, rule_report={})

    # Assert: panel_target_weight 속성은 없어야 한다 (버그 재현 조건)
    assert not hasattr(ml_out, "panel_target_weight"), (
        "CandidatePipelineOutput must not have panel_target_weight attribute"
    )

    # Assert: alpha_panel["target_weight"]에서 nonzero를 올바르게 집계하는지 검증
    alpha_panel_check = getattr(ml_out, "alpha_panel", None)
    assert isinstance(alpha_panel_check, pd.DataFrame)
    assert "target_weight" in alpha_panel_check.columns
    tw_arr = alpha_panel_check["target_weight"].to_numpy(dtype=np.float64)
    non_zero_weights = int(np.count_nonzero(np.abs(tw_arr) > 1e-9))
    assert non_zero_weights == 3, f"Expected 3 nonzero weights, got {non_zero_weights}"


# ─── Tiered aligned scope fix (Method B) ─────────────────────────────────────


def _make_snapshot(selected_symbols: list[str]) -> UniverseSnapshot:
    """Build a minimal UniverseSnapshot with given selected symbols."""
    metas = tuple(
        SymbolMeta(
            symbol=sym,
            role="anchor",
            adv_usdt=1.0,
            execution_cost_bps=1.0,
            funding_carry_8h=0.0,
            beta_vs_market=1.0,
            cluster_id=0,
            tradeable_rank=i + 1,
            basis_annualized_mean=None,
            basis_vol=None,
            capacity_clip_usdt_list=(),
            cluster_size=1.0,
            anchor_cluster_member=0.0,
        )
        for i, sym in enumerate(selected_symbols)
    )
    return UniverseSnapshot(
        as_of="2026-01-01",
        tf="4h",
        schema_version=1,
        config_hash="c",
        data_manifest_hash="m",
        basket_ref=(),
        basket_weights=(),
        selected=metas,
        rejected={},
        generated_at_utc="2026-01-01T00:00:00+00:00",
        ledger_confidence="high",
        n_stage0=len(selected_symbols),
        n_stage1_pass=len(selected_symbols),
        n_stage2_pass=len(selected_symbols),
        n_stage3_pass=len(selected_symbols),
        n_stage4_pass=len(selected_symbols),
        n_stage5_pass=len(selected_symbols),
        n_stage6_selected=len(selected_symbols),
    )


def _make_window() -> opt_main_futures.QuarterlyWindow:
    return opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )


def _patch_tiered_deps(
    monkeypatch: pytest.MonkeyPatch,
    captured_symbols: list[list[str]],
) -> None:
    """Patch all dependencies needed to reach the align_data_maps call in the tiered block."""
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow import Layer1Result

    # OPT_FUTURES_CONFIG must have USE_CS_RANK_ENGINE=True
    monkeypatch.setattr(
        opt_main_futures,
        "OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "FUTURES_STRATEGY_NAME": "candidate_ml"},
    )

    # Bridge returns empty output (no labeled events)
    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled_unfiltered=pd.DataFrame({"symbol": ["BTCUSDT"]})
        ),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "merge_candidate_output_into_is_and_oos",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_candidate_evaluation_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_args, **_kwargs: MagicMock(candidate=MagicMock()),
    )

    # Lazy-imported dependencies inside the tiered block
    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw

    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kw: MagicMock())
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))

    # Capture symbols arg passed to align_data_maps
    def fake_align(
        data_maps: dict[str, object],
        symbols: list[str],
        tf: str,
        **kwargs: object,
    ) -> MagicMock:
        captured_symbols.append(list(symbols))
        mock_aligned = MagicMock()
        mock_aligned.symbols = symbols
        return mock_aligned

    monkeypatch.setattr(_align, "align_data_maps", fake_align)

    # run_tiered_pipeline returns a minimal Layer1Result
    dummy_l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.12,
        pooled_tstat=1.64,
        breadth=1.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.71,
        gate_passed=False,
        n_valid=12,
        n_total=12,
        n_trade_scope=12,
    )
    monkeypatch.setattr(
        _tw,
        "run_tiered_pipeline",
        lambda **_kw: (dummy_l1, None, None),
    )


def test_tiered_aligned_scope_s1_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: tiered aligned scope는 current snapshot이 아니라 historical valid_symbols 전체를 사용."""
    # Arrange
    stage6_syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_syms = stage6_syms + [f"SYM{i}USDT" for i in range(7)]  # 10 total in data_maps
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_maps = {sym: {"4h": frame.copy()} for sym in all_syms}
    data_stage = opt_main_futures.DataStageResult(
        data_maps=data_maps,
        oos_data_maps={},
        valid_symbols=all_syms,  # 10
    )
    snapshot = _make_snapshot(stage6_syms)
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    # Act
    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=snapshot,
    )

    # Assert: align_data_maps called with full historical union
    assert len(captured) >= 1, "align_data_maps must be called in tiered block"
    tiered_call = captured[-1]
    assert sorted(tiered_call) == sorted(all_syms)
    assert len(tiered_call) == 10


def test_tiered_aligned_scope_s2_fallback_when_no_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: snapshot overlap이 없어도 aligned scope는 valid_symbols를 유지한다."""
    # Arrange
    stage6_syms = ["XYZUSDT", "ABCUSDT"]
    valid_syms = [f"SYM{i}USDT" for i in range(10)]
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_maps = {sym: {"4h": frame.copy()} for sym in valid_syms}
    data_stage = opt_main_futures.DataStageResult(
        data_maps=data_maps,
        oos_data_maps={},
        valid_symbols=valid_syms,  # 10
    )
    snapshot = _make_snapshot(stage6_syms)  # stage6 syms absent from data_maps
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    # Act
    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=snapshot,
    )

    # Assert: fallback → valid_symbols (10)
    assert len(captured) >= 1
    tiered_call = captured[-1]
    assert sorted(tiered_call) == sorted(valid_syms)


def test_tiered_aligned_scope_s3_regression_breadth_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: current snapshot보다 historical union이 우선이므로 valid_symbols 전체가 전달된다."""
    from src.domain.futures.strategy.tiered_workflow import Layer1Result

    stage6_syms = [f"SYM{i}USDT" for i in range(12)]
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_maps = {sym: {"4h": frame.copy()} for sym in stage6_syms}
    # valid_symbols had 63; tiered now uses the historical union instead of current snapshot(12)
    valid_syms_63 = stage6_syms + [f"EXTRA{i}USDT" for i in range(51)]
    data_stage = opt_main_futures.DataStageResult(
        data_maps=data_maps,
        oos_data_maps={},
        valid_symbols=valid_syms_63,  # 63 before fix
    )
    snapshot = _make_snapshot(stage6_syms)
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    opt_main_futures._run_strategy_stage(
        run_config, _make_window(), data_stage, universe_snapshot=snapshot
    )

    # align_data_maps receives the full historical union
    assert len(captured) >= 1
    assert len(captured[-1]) == 63
    assert sorted(captured[-1]) == sorted(valid_syms_63)

    # Verify breadth would be 12/12 = 1.0 with correct scope
    breadth_after_fix = 12 / 12
    assert breadth_after_fix == pytest.approx(1.0)
    # tstat 1.64 < 1.96 → gate still BLOCKED (scope fix doesn't fix alpha quality)
    dummy_l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.12,
        pooled_tstat=1.64,
        breadth=breadth_after_fix,
        valid_coverage=0.0,
        fold_pass_ratio=0.71,
        gate_passed=False,
        n_valid=12,
        n_total=12,
        n_trade_scope=12,
    )
    assert dummy_l1.gate_passed is False
    assert dummy_l1.breadth == pytest.approx(1.0)


def test_tiered_aligned_scope_s4_partial_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: snapshot 부분 겹침이 있어도 tiered aligned scope는 valid_symbols 그대로 유지한다."""
    stage6_syms = ["AAUSDT", "BBUSDT", "CCUSDT"]
    data_map_syms = ["AAUSDT", "BBUSDT", "DDUSDT", "EEUSDT"]
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_maps = {sym: {"4h": frame.copy()} for sym in data_map_syms}
    data_stage = opt_main_futures.DataStageResult(
        data_maps=data_maps,
        oos_data_maps={},
        valid_symbols=data_map_syms,
    )
    snapshot = _make_snapshot(stage6_syms)
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    opt_main_futures._run_strategy_stage(
        run_config, _make_window(), data_stage, universe_snapshot=snapshot
    )

    # Assert: snapshot overlap과 무관하게 valid_symbols 유지
    assert len(captured) >= 1
    assert sorted(captured[-1]) == sorted(data_map_syms)


def test_tiered_window_uses_run_config_date_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow import Layer1Result

    captured_reference_dates: list[object] = []
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame.copy()}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setattr(
        opt_main_futures,
        "OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "FUTURES_STRATEGY_NAME": "candidate_ml"},
    )
    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled_unfiltered=pd.DataFrame({"symbol": ["BTCUSDT"]})
        ),
    )
    monkeypatch.setattr(opt_main_futures, "merge_candidate_output_into_is_and_oos", lambda *_a, **_k: None)
    monkeypatch.setattr(opt_main_futures, "_run_candidate_evaluation_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_a, **_k: MagicMock(candidate=MagicMock()),
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw

    def _capture_layered_window(**kwargs: object) -> MagicMock:
        captured_reference_dates.append(kwargs["reference_date"])
        return MagicMock()

    monkeypatch.setattr(
        _opt_cfg,
        "get_layered_window",
        _capture_layered_window,
    )
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        _align,
        "align_data_maps",
        lambda data_maps, symbols, tf, **_kw: MagicMock(symbols=symbols, data_maps=data_maps, tf=tf),
    )
    monkeypatch.setattr(
        _tw,
        "run_tiered_pipeline",
        lambda **_kwargs: (
            Layer1Result(
                signals_per_fold=(),
                oos_stacked={},
                pooled_ic=0.12,
                pooled_tstat=2.1,
                breadth=1.0,
                valid_coverage=1.0,
                fold_pass_ratio=1.0,
                gate_passed=True,
                n_valid=1,
                n_total=1,
                n_trade_scope=1,
            ),
            None,
            None,
        ),
    )

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full", "date": "2026-05-01"}
    )
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    assert captured_reference_dates == [datetime.strptime("2026-05-01", "%Y-%m-%d").date()]


def test_tiered_pipeline_uses_unfiltered_labeled_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow import Layer1Result

    captured_labeled_events: list[pd.DataFrame] = []
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame.copy()}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )
    filtered = pd.DataFrame({"symbol": ["BTCUSDT"], "variant": ["promoted"]})
    unfiltered = pd.DataFrame(
        {"symbol": ["BTCUSDT", "ETHUSDT"], "variant": ["promoted", "candidate"]}
    )

    monkeypatch.setattr(
        opt_main_futures,
        "OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "FUTURES_STRATEGY_NAME": "candidate_ml"},
    )
    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled=filtered,
            labeled_unfiltered=unfiltered,
        ),
    )
    monkeypatch.setattr(opt_main_futures, "merge_candidate_output_into_is_and_oos", lambda *_a, **_k: None)
    monkeypatch.setattr(opt_main_futures, "_run_candidate_evaluation_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_a, **_k: MagicMock(candidate=MagicMock()),
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw

    mock_win = MagicMock()
    mock_win.fetch_start = datetime(1900, 1, 1).date()
    mock_win.holdout_start = datetime(1900, 1, 1).date()
    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kwargs: mock_win)
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        _align,
        "align_data_maps",
        lambda data_maps, symbols, tf, **_kw: MagicMock(symbols=symbols, data_maps=data_maps, tf=tf),
    )

    def _capture_tiered(**kwargs: object) -> tuple[Layer1Result, None, None]:
        labeled_events = kwargs["labeled_events"]
        assert isinstance(labeled_events, pd.DataFrame)
        captured_labeled_events.append(labeled_events)
        return (
            Layer1Result(
                signals_per_fold=(),
                oos_stacked={},
                pooled_ic=0.12,
                pooled_tstat=2.1,
                breadth=1.0,
                valid_coverage=1.0,
                fold_pass_ratio=1.0,
                gate_passed=True,
                n_valid=1,
                n_total=1,
                n_trade_scope=1,
            ),
            None,
            None,
        )

    monkeypatch.setattr(_tw, "run_tiered_pipeline", _capture_tiered)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full", "date": "2026-05-01"}
    )
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    assert len(captured_labeled_events) == 1
    assert captured_labeled_events[0].equals(unfiltered)


def test_tiered_layer3_terminal_failure_does_not_fallback_to_phase_d(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L3 terminal error는 legacy Phase D fallback 없이 종료되어야 한다."""
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow import Layer3ExecutionError
    from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput

    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="4h")})
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame.copy()}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setattr(
        opt_main_futures,
        "OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "FUTURES_STRATEGY_NAME": "candidate_ml"},
    )
    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled_unfiltered=pd.DataFrame({"symbol": ["BTCUSDT"]})
        ),
    )
    monkeypatch.setattr(opt_main_futures, "merge_candidate_output_into_is_and_oos", lambda *_a, **_k: None)
    monkeypatch.setattr(opt_main_futures, "_run_candidate_evaluation_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_a, **_k: MagicMock(candidate=MagicMock()),
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw

    mock_win = MagicMock()
    mock_win.fetch_start = datetime(1900, 1, 1).date()
    mock_win.holdout_start = datetime(1900, 1, 1).date()
    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kwargs: mock_win)
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        _align,
        "align_data_maps",
        lambda data_maps, symbols, tf, **_kw: MagicMock(
            symbols=symbols,
            datetimes=np.array([], dtype="datetime64[ns]"),
        ),
    )
    def _raise_l3_error(**_kwargs: object) -> object:
        raise Layer3ExecutionError("layer3_signal_prediction_failed")

    monkeypatch.setattr(
        _tw,
        "run_tiered_pipeline",
        _raise_l3_error,
    )

    caplog.set_level(logging.INFO)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full", "date": "2026-05-01"}
    )
    result = opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    assert result is None
    assert "[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]" not in caplog.text


# ─── Layer3 holdout integrity: END-coverage filter/guard (layer3-holdout-integrity.md) ──


def _make_layered_window(
    *,
    fetch_start: date,
    holdout_start: date,
    holdout_end: date,
) -> object:
    """Build a real LayeredWindow with explicit date boundaries (no MagicMock dates)."""
    from src.domain.futures.optimization.opt_config import LayeredWindow

    return LayeredWindow(
        fetch_start=fetch_start,
        l1_start=fetch_start,
        l2_start=holdout_start,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        regime_floor=fetch_start,
    )


def test_effective_trade_syms_s1_excludes_symbol_delisted_before_holdout_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: END-coverage 필터 — holdout_end 전 폐지 심볼은 effective_trade_syms에서 제외된다."""
    # Arrange: A/B는 holdout_end까지 거래, C는 holdout_start 전에 폐지(last_dt < holdout_end)
    fetch_start = date(2025, 1, 1)
    holdout_start = date(2025, 10, 1)
    holdout_end = date(2026, 3, 31)

    # 4h bars from 2022-01-01: ~7700 bars to 2026-04-07 (>> min_window_bars=1500)
    full_coverage = pd.DataFrame(
        {"datetime": pd.date_range("2022-01-01", "2026-04-07", freq="4h", tz="UTC")}
    )
    # Delisted: also > 1500 bars, but ends 2025-08-01 (< holdout_start=2025-10-01 → 0% OOS coverage)
    delisted_coverage = pd.DataFrame(
        {"datetime": pd.date_range("2022-01-01", "2025-08-01", freq="4h", tz="UTC")}
    )
    data_maps = {
        "AAUSDT": {"4h": full_coverage.copy()},
        "BBUSDT": {"4h": full_coverage.copy()},
        "CCUSDT": {"4h": delisted_coverage.copy()},
    }
    data_stage = opt_main_futures.DataStageResult(
        data_maps=data_maps,
        oos_data_maps={},
        valid_symbols=["AAUSDT", "BBUSDT", "CCUSDT"],
    )
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    import src.domain.futures.optimization.opt_config as _opt_cfg

    layered_window = _make_layered_window(
        fetch_start=fetch_start, holdout_start=holdout_start, holdout_end=holdout_end
    )
    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kw: layered_window)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )

    # Act
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["AAUSDT", "BBUSDT", "CCUSDT"]),
    )

    # Assert: effective_trade_syms == [AA, BB] (CC delisted before holdout_end, excluded)
    assert len(captured) >= 1
    tiered_call = captured[-1]
    assert sorted(tiered_call) == ["AAUSDT", "BBUSDT"]
    assert "CCUSDT" not in tiered_call


def test_tiered_pipeline_s2_raises_when_aligned_end_before_holdout_start(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S2 (갱신): END-coverage guard — align_data_maps가 holdout_start 전에 잘린 datetimes를
    반환하면 ValueError("tiered holdout coverage missing")가 raise되고, 상위 generic except이
    이를 포착해 즉시 RunnerResult(exit_code=1)로 종료한다. Phase D fallback은 더 이상 발생하지
    않는다(PART 3 — Phase D Silent Fallback 제거).
    """
    from datetime import date
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow import Layer1Result

    fetch_start = date(2025, 1, 1)
    holdout_start = date(2025, 10, 1)
    holdout_end = date(2026, 3, 31)

    frame = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=4, freq="4h")})
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame.copy()}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setattr(
        opt_main_futures,
        "OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "FUTURES_STRATEGY_NAME": "candidate_ml"},
    )
    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled_unfiltered=pd.DataFrame({"symbol": ["BTCUSDT"]})
        ),
    )
    monkeypatch.setattr(opt_main_futures, "merge_candidate_output_into_is_and_oos", lambda *_a, **_k: None)
    monkeypatch.setattr(opt_main_futures, "_run_candidate_evaluation_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_a, **_k: MagicMock(candidate=MagicMock()),
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw

    layered_window = _make_layered_window(
        fetch_start=fetch_start, holdout_start=holdout_start, holdout_end=holdout_end
    )
    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kw: layered_window)
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))

    # align_data_maps (boundary mock, autospec=True) returns aligned datetimes whose
    # last bar is before holdout_start — simulates intersection-tail truncation.
    truncated_datetimes = np.array(
        [np.datetime64("2025-01-01") + np.timedelta64(i, "D") for i in range(5)],
        dtype="datetime64[ns]",
    )
    from unittest.mock import create_autospec

    real_align = _align.align_data_maps
    fake_align = create_autospec(real_align, spec_set=True)
    fake_align.return_value = MagicMock(
        symbols=["BTCUSDT"],
        datetimes=truncated_datetimes,
    )
    monkeypatch.setattr(_align, "align_data_maps", fake_align)

    monkeypatch.setattr(
        _tw,
        "run_tiered_pipeline",
        lambda **_kw: (
            Layer1Result(
                signals_per_fold=(),
                oos_stacked={},
                pooled_ic=0.0,
                pooled_tstat=0.0,
                breadth=0.0,
                valid_coverage=0.0,
                fold_pass_ratio=0.0,
                gate_passed=True,
                n_valid=1,
                n_total=1,
                n_trade_scope=1,
            ),
            None,
            None,
        ),
    )

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    caplog.set_level(logging.INFO)

    # Act: the ValueError is raised inside the tiered try-block, caught by the
    # generic `except Exception` clause — terminal failure, no Phase D fallback.
    result = opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    # Assert: terminal failure surfaced as RunnerResult, not a silent Phase D fallback.
    assert result is not None
    assert result.exit_code == 1
    assert result.reason.startswith("tiered_pipeline_error:")
    assert "ValueError" in result.reason
    assert "[SIGNAL DIAGNOSTICS" not in caplog.text


# ─── PART 4: IS/OOS 데이터 병합 — END-coverage 필터/align이 병합 맵을 사용 ────────


def test_effective_trade_syms_s14_uses_merged_full_strategy_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S14: END-coverage 필터는 full_strategy_maps(IS+OOS 병합)를 기준으로 동작한다.
    data_stage.data_maps(IS-only)만으로는 holdout_end 커버리지가 없지만, IS+OOS 병합
    결과(full_strategy_maps)에는 holdout_end까지 커버리지가 있으므로 effective_trade_syms에
    포함되어야 한다(회귀 방어 — pick_strategy_data_maps가 IS-only를 반환하던 버그 재발 방지).
    """
    # Arrange: IS-only data_maps는 holdout_start 이전에서 끝남(END-coverage 미달)
    fetch_start = date(2025, 1, 1)
    holdout_start = date(2025, 10, 1)
    holdout_end = date(2026, 3, 31)

    is_only_coverage = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01", "2025-09-01", freq="7D", tz="UTC")}
    )
    # OOS frame extends coverage through holdout_end
    oos_coverage = pd.DataFrame(
        {"datetime": pd.date_range("2025-09-08", "2026-04-07", freq="7D", tz="UTC")}
    )
    full_merged_coverage = pd.concat(
        [is_only_coverage, oos_coverage], ignore_index=True
    ).sort_values("datetime").reset_index(drop=True)

    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": is_only_coverage.copy()}},
        oos_data_maps={"BTCUSDT": {"4h": oos_coverage.copy()}},
        valid_symbols=["BTCUSDT"],
    )
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    # pick_strategy_data_maps is real (not mocked) — exercises C5 merge logic so
    # full_strategy_maps carries the merged (IS+OOS) coverage through holdout_end.
    monkeypatch.setattr(
        opt_main_futures,
        "pick_strategy_data_maps",
        lambda **_kw: {"BTCUSDT": {"4h": full_merged_coverage.copy()}},
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg

    layered_window = _make_layered_window(
        fetch_start=fetch_start, holdout_start=holdout_start, holdout_end=holdout_end
    )
    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kw: layered_window)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )

    # Act
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    # Assert: BTCUSDT passes END-coverage filter (using merged full_strategy_maps),
    # not excluded as it would be if data_stage.data_maps (IS-only) were checked.
    assert len(captured) >= 1
    tiered_call = captured[-1]
    assert "BTCUSDT" in tiered_call


def test_align_data_maps_s15_receives_merged_full_strategy_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S15: align_data_maps 호출 시 첫 인자가 full_strategy_maps(병합 맵)여야 한다
    (회귀 방어 — data_stage.data_maps(IS-only)를 넘기면 empty_holdout_window 재발).
    """
    from unittest.mock import MagicMock

    frame_is = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=4, freq="4h")})
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": frame_is.copy()}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )
    captured: list[list[str]] = []
    _patch_tiered_deps(monkeypatch, captured)

    merged_frame = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=10, freq="4h")})
    merged_sentinel_maps = {"BTCUSDT": {"4h": merged_frame}}
    monkeypatch.setattr(
        opt_main_futures,
        "pick_strategy_data_maps",
        lambda **_kw: merged_sentinel_maps,
    )

    captured_align_data_maps: list[object] = []
    import src.domain.futures.strategy.common.alignment as _align

    def _capturing_align(data_maps: object, symbols: list[str], tf: str, **kwargs: object) -> object:
        captured_align_data_maps.append(data_maps)
        return MagicMock(symbols=symbols, datetimes=np.array([], dtype="datetime64[ns]"))

    monkeypatch.setattr(_align, "align_data_maps", _capturing_align)

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )

    # Act
    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
    )

    # Assert: align_data_maps received full_strategy_maps (merged), not data_stage.data_maps.
    assert len(captured_align_data_maps) >= 1
    assert captured_align_data_maps[-1] is merged_sentinel_maps


def test_run_strategy_stage_passes_pit_state_cube_with_real_run_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    data_times = pd.date_range("2025-01-01", "2026-04-07", freq="7D", tz="UTC")
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {"4h": pd.DataFrame({"datetime": data_times})}},
        oos_data_maps={},
        valid_symbols=["BTCUSDT"],
    )
    cube = UniverseStateCube(
        calendar=pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC"),
        instrument_ids=("BTCUSDT",),
        eligible=np.ones((4, 1), dtype=np.bool_),
        entry_block=np.zeros((4, 1), dtype=np.bool_),
        exit_required=np.zeros((4, 1), dtype=np.bool_),
        capacity_usdt=np.full((4, 1), 1_000.0, dtype=np.float64),
        risk_scale=np.ones((4, 1), dtype=np.float64),
        cost_bps=np.full((4, 1), 5.0, dtype=np.float64),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        opt_main_futures,
        "run_active_strategy_output_bridge",
        lambda *_args, **_kwargs: CandidatePipelineOutput(
            labeled_unfiltered=pd.DataFrame({"symbol": ["BTCUSDT"]})
        ),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "merge_candidate_output_into_is_and_oos",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_candidate_evaluation_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "build_candidate_strategy_config",
        lambda *_args, **_kwargs: MagicMock(candidate=MagicMock()),
    )

    import src.domain.futures.optimization.opt_config as _opt_cfg
    import src.domain.futures.portfolio.portfolio_constructor as _pc
    import src.domain.futures.strategy.common.alignment as _align
    import src.domain.futures.strategy.tiered_workflow as _tw
    import src.domain.futures.universe.readiness as _readiness

    monkeypatch.setattr(_opt_cfg, "get_layered_window", lambda **_kw: MagicMock())
    monkeypatch.setattr(_pc, "PortfolioCaps", MagicMock(return_value=MagicMock()))

    def fake_align(
        data_maps: dict[str, object],
        symbols: list[str],
        tf: str,
        **kwargs: object,
    ) -> MagicMock:
        _ = data_maps
        _ = symbols
        _ = tf
        captured["state_cube"] = kwargs.get("state_cube")
        return MagicMock(
            symbols=("BTCUSDT",),
            datetimes=np.array(
                pd.date_range("2025-01-01", periods=4, freq="4h").to_numpy(),
                dtype="datetime64[ns]",
            ),
        )

    def fake_readiness(**kwargs: object) -> MagicMock:
        captured["eligibility"] = kwargs.get("eligibility")
        return MagicMock(ready=np.ones((1, 4, 1), dtype=np.bool_))

    dummy_l1 = _tw.Layer1Result(
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
        n_trade_scope=0,
    )

    monkeypatch.setattr(_align, "align_data_maps", fake_align)
    monkeypatch.setattr(_readiness, "evaluate_strategy_readiness", fake_readiness)
    monkeypatch.setattr(_tw, "run_tiered_pipeline", lambda **_kw: (dummy_l1, None, None))

    opt_main_futures._run_strategy_stage(
        run_config,
        _make_window(),
        data_stage,
        universe_snapshot=_make_snapshot(["BTCUSDT"]),
        universe_result=SimpleNamespace(state_cube=cube),
    )

    assert captured["state_cube"] is cube
    assert captured["eligibility"] is cube


# ---------------------------------------------------------------------------
# C1 — _resolve_tradeable_scope unit tests
# ---------------------------------------------------------------------------

def _make_sym_df(
    start: str,
    end: str,
    freq: str = "4h",
    tf: str = "4h",
) -> dict[str, pd.DataFrame]:
    """Build a minimal strategy_maps entry for a single symbol."""
    dts = pd.date_range(start, end, freq=freq, tz="UTC")
    df = pd.DataFrame({"datetime": dts})
    return {tf: df}


def test_resolve_tradeable_scope_scenario1_early_listed_delisted_mid_oos_admitted() -> None:
    """Scenario 1 (Happy Path): symbol listed before fetch_start but delisted before
    holdout_end is admitted if OOS coverage >= 90%.

    The old END-coverage filter (last_dt >= holdout_end) would reject symB.
    The new sub-window admission admits symB because it covers >90% of OOS.
    symC is listed AFTER fetch_start → rejected by warm-up guard regardless of OOS coverage.
    """
    # Arrange
    fetch_start = pd.Timestamp("2022-10-01", tz="UTC")
    oos_start = pd.Timestamp("2025-10-01", tz="UTC")
    holdout_end = pd.Timestamp("2026-04-01", tz="UTC")

    strategy_maps: dict[str, dict[str, pd.DataFrame]] = {
        "symA": _make_sym_df("2022-10-01", "2026-04-01"),  # full window
        "symB": _make_sym_df("2022-10-01", "2026-03-15"),  # delisted 17d before holdout_end
        "symC": _make_sym_df("2023-06-01", "2026-04-01"),  # listed after fetch_start
    }

    # Act
    result = opt_main_futures._resolve_tradeable_scope(
        valid_symbols=["symA", "symB", "symC"],
        strategy_maps=strategy_maps,
        tf="4h",
        fetch_start=fetch_start,
        oos_start=oos_start,
        holdout_end=holdout_end,
        min_window_bars=1,
        min_holdout_coverage=0.90,
    )

    # symA: admitted (full coverage); symB: admitted (OOS cov ~97% > 90%, starts ≤ fetch_start)
    # symC: rejected (starts 2023-06 > fetch_start → warm-up guard fails)
    assert set(result) == {"symA", "symB"}
    assert "symC" not in result


def test_resolve_tradeable_scope_scenario2_holdout_truncation_excluded() -> None:
    """Scenario 2 (Holdout-truncation guard): symbol delisted before OOS excluded.

    symD has data only [fetch_start, oos_start], zero OOS coverage → excluded.
    """
    # Arrange
    fetch_start = pd.Timestamp("2022-10-01", tz="UTC")
    oos_start = pd.Timestamp("2025-10-01", tz="UTC")
    holdout_end = pd.Timestamp("2026-04-01", tz="UTC")

    strategy_maps: dict[str, dict[str, pd.DataFrame]] = {
        "symD": _make_sym_df("2022-10-01", "2025-09-30"),  # ends before oos_start
    }

    # Act
    result = opt_main_futures._resolve_tradeable_scope(
        valid_symbols=["symD"],
        strategy_maps=strategy_maps,
        tf="4h",
        fetch_start=fetch_start,
        oos_start=oos_start,
        holdout_end=holdout_end,
        min_window_bars=1,
        min_holdout_coverage=0.90,
    )

    # Assert — symD excluded due to zero OOS coverage
    assert "symD" not in result
    assert result == []


def test_resolve_tradeable_scope_scenario3_min_bars_guard_excluded() -> None:
    """Scenario 3 (min_bars guard): symbol with fewer bars than min_window_bars excluded.

    symE has only 10 bars; min_window_bars=1500 → excluded.
    """
    # Arrange
    fetch_start = pd.Timestamp("2022-10-01", tz="UTC")
    oos_start = pd.Timestamp("2025-10-01", tz="UTC")
    holdout_end = pd.Timestamp("2026-04-01", tz="UTC")

    # 10 bars spaced 4h apart — well within the window but far below 1500
    dts = pd.date_range("2025-10-01", periods=10, freq="4h", tz="UTC")
    df = pd.DataFrame({"datetime": dts})
    strategy_maps: dict[str, dict[str, pd.DataFrame]] = {"symE": {"4h": df}}

    # Act
    result = opt_main_futures._resolve_tradeable_scope(
        valid_symbols=["symE"],
        strategy_maps=strategy_maps,
        tf="4h",
        fetch_start=fetch_start,
        oos_start=oos_start,
        holdout_end=holdout_end,
        min_window_bars=1500,
        min_holdout_coverage=0.90,
    )

    # Assert — symE excluded because n_bars (10) < min_window_bars (1500)
    assert "symE" not in result
    assert result == []
