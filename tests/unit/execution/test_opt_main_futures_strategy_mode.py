from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.application.futures.optimization.config import build_run_config_from_args
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput
from src.domain.futures.universe import SymbolMeta, UniverseSnapshot
from src.execution import opt_main_futures


def test_strategy_mode_pipeline_orchestration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "strategy",
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
        training_panel=("BTCUSDT",),
        live_inference_panel=("BTCUSDT",),
    )

    def fake_window(reference_date: str | None) -> opt_main_futures.QuarterlyWindow:
        _ = reference_date
        called.append("window")
        return window

    def fake_universe(
        rc: object,
        win: object,
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
        called.append("universe")
        return ["BTCUSDT"], {}, (), (), snapshot, {}

    def fake_data(
        rc: object,
        win: object,
        discovered: list[str],
        timeline: dict[object, frozenset[str]],
        inference_panel: tuple[str, ...] = (),
        live_inference_panel: tuple[str, ...] = (),
        inference_timeline: dict[object, frozenset[str]] | None = None,
    ) -> opt_main_futures.DataStageResult:
        _ = rc
        _ = win
        _ = discovered
        _ = timeline
        _ = inference_panel
        _ = live_inference_panel
        _ = inference_timeline
        called.append("data")
        return data_stage

    def fake_strategy(
        rc: object,
        win: object,
        ds: object,
        inference_panel: tuple[str, ...] = (),
        live_inference_panel: tuple[str, ...] = (),
        trading_symbols: tuple[str, ...] = (),
        universe_snapshot: object | None = None,
    ) -> None:
        _ = rc
        _ = win
        _ = ds
        _ = inference_panel
        _ = live_inference_panel
        _ = trading_symbols
        _ = universe_snapshot
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
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", fake_strategy)
    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fake_optimization)

    result = opt_main_futures.run_pipeline(run_config, seed=13, resume=True)
    assert result.exit_code == 0
    assert called == ["window", "universe", "data", "strategy", "optimization"]


def test_alpha_mode_skips_optimization_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alpha 모드는 strategy bridge 후 optimization 없이 종료해야 한다."""
    run_config = build_run_config_from_args(
        {
            "phase": "alpha",
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
        lambda *_: (
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
        ),
    )
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", lambda *_: data_stage)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", lambda *_args, **_kwargs: None)

    def fail_if_called(*args: object, **kwargs: object) -> opt_main_futures.RunnerResult:
        raise AssertionError("optimization stage should not be called in alpha mode")

    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fail_if_called)
    result = opt_main_futures.run_pipeline(run_config)
    assert result.exit_code == 0
    assert result.reason == "candidate_evaluation_done"


def test_strategy_stage_injects_universe_metadata_before_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "alpha",
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
        training_panel=("BTCUSDT",),
        live_inference_panel=("BTCUSDT",),
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

    opt_main_futures._run_strategy_stage(
        run_config,
        window,
        data_stage,
        inference_panel=("BTCUSDT",),
        live_inference_panel=("BTCUSDT",),
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
    argv = ["--phase", "full"]
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
    argv = ["--phase", "full"]

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> opt_main_futures.RunnerResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", _raise_runtime_error)

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 1


def test_requires_exec_1m_returns_false_for_ml_mode() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "ml",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    assert opt_main_futures._requires_exec_1m(run_config) is False


def test_requires_exec_1m_returns_false_for_strategy_phase() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "strategy",
            "timeframe": "4h",
            "trials": 1,
            "sync": "full",
        }
    )
    assert opt_main_futures._requires_exec_1m(run_config) is False


def test_resolve_data_collection_symbols_uses_inference_panel() -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "strategy",
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
            "phase": "strategy",
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
            "phase": "strategy",
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
    monkeypatch.setattr(opt_main_futures, "FUTURES_DATA_DIR", tmp_path)
    opt_main_futures._ensure_universe_ledger_sync(run_config, window)
    assert calls
    assert calls[0].get("symbols") is None


def test_ensure_cached_symbol_data_uses_fetch_start_for_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "phase": "strategy",
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


# ─── --mode signal ────────────────────────────────────────────────────────────

def test_build_run_config_accepts_mode_signal() -> None:
    # "--mode signal" is mapped to phase="signal" for backward compatibility
    cfg = build_run_config_from_args(
        {"phase": "strategy", "timeframe": "4h", "trials": 1, "sync": "full", "mode": "signal"}
    )
    assert cfg.phase == "signal"


def test_cli_mode_signal_is_accepted() -> None:
    """--mode signal parsed from CLI args must be accepted."""
    parser = opt_main_futures.build_arg_parser()
    args = parser.parse_args(["--phase", "full", "--mode", "signal"])
    assert args.mode == "signal"
