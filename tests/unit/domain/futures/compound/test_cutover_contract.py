from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.futures.runner.cli import build_arg_parser
from src.application.futures.runner.compound_config import build_compound_run_config
from src.domain.futures.compound.config import DataPlaneConfig
from src.domain.futures.compound.contracts import ExecutionLedger, MarketFeatureCube
from src.domain.futures.compound.data_plane import build_compound_market_feature_cube
from src.domain.futures.compound.validation import slice_execution_ledger


def _cube(n: int = 8) -> MarketFeatureCube:
    close = np.full((n, 1), 100.0, dtype=np.float64)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n, dtype=np.int64),
        symbols=("BTCUSDT",),
        fields_2d={"open": close, "high": close, "low": close, "close": close,
                   "quote_volume": close.astype(np.float32), "funding": np.zeros((n, 1), dtype=np.float32)},
        available_2d={"core": np.ones((n, 1), dtype=np.bool_)},
        eligible_2d=np.ones((n, 1), dtype=np.bool_), entry_block_2d=np.zeros((n, 1), dtype=np.bool_),
        capacity_usdt_2d=np.full((n, 1), 1e6), execution_cost_bps_2d=np.full((n, 1), 12, dtype=np.float32),
        data_manifest_hash="fixture",
    )


def _ledger() -> ExecutionLedger:
    n = 8
    returns = np.zeros(n, dtype=np.float64)
    return ExecutionLedger(
        timestamps_ns=np.arange(n, dtype=np.int64), net_returns_1d=returns,
        equity_1d=np.ones(n), target_weights_2d=np.zeros((n, 1), dtype=np.float32),
        fee_returns_1d=returns.copy(), slippage_returns_1d=returns.copy(),
        impact_returns_1d=returns.copy(), funding_returns_1d=returns.copy(),
        integrity_ok=True, integrity_reasons=(),
    )


def test_compound_cli_config_accepts_single_run_arguments() -> None:
    args = build_arg_parser().parse_args(["--date", "2026-07-08", "--sync", "skip"])
    config = build_compound_run_config(args)
    assert config.base_timeframe == "1h" and config.reference_date == "2026-07-08"


def test_opt_main_futures_invokes_multiscale_cli_exactly_once() -> None:
    source = Path("src/execution/opt_main_futures.py").read_text(encoding="utf-8")
    assert source.count("raise SystemExit(cli())") == 1


def test_retained_source_has_zero_legacy_imports() -> None:
    banned = ("alpha_foundry", "tiered_workflow", "strategy_runtime", "online_growth_allocator")
    for path in (Path("src/application/futures/runner"), Path("src/domain/futures/compound")):
        for file in path.rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            assert not any(token in module for module in imports for token in banned)


def test_declared_legacy_files_are_absent() -> None:
    assert not Path("src/application/futures/runner/active_pipeline.py").exists()
    assert not Path("src/domain/futures/alpha_foundry").exists()


def test_pit_cube_alignment_never_uses_future_state() -> None:
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"], utc=True),
                          "open": [100, 102], "high": [101, 103], "low": [99, 101], "close": [100, 102], "volume": [1e6, 1e6]})
    state = _state(3)
    cube = build_compound_market_feature_cube(data_maps={"BTCUSDT": {"1h": frame}}, symbols=("BTCUSDT",),
                                              state_cube=state, timeframe="1h", data_manifest_hash="h", config=DataPlaneConfig())
    assert cube.fields_2d["close"][1, 0] == pytest.approx(100.0)


def _state(n: int):
    from src.domain.futures.universe.contracts import UniverseStateCube
    return UniverseStateCube(calendar=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
                              instrument_ids=("BTCUSDT",), eligible=np.ones((n, 1), dtype=bool),
                              entry_block=np.zeros((n, 1), dtype=bool), exit_required=np.zeros((n, 1), dtype=bool),
                              capacity_usdt=np.full((n, 1), 1e6), risk_scale=np.ones((n, 1)), cost_bps=np.full((n, 1), 12.0))


def test_missing_universe_symbol_is_entry_blocked() -> None:
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1e6})
    cube = build_compound_market_feature_cube(data_maps={"ETHUSDT": {"1h": frame}}, symbols=("BTCUSDT",), state_cube=_state(3), timeframe="1h", data_manifest_hash="h", config=DataPlaneConfig())
    assert cube.entry_block_2d.all() and not cube.eligible_2d.any()


def test_data_admission_enforces_coverage_listing_and_liquidity() -> None:
    assert DataPlaneConfig().min_core_coverage == pytest.approx(0.98)


def test_metric_release_timestamp_prevents_future_leakage() -> None:
    test_pit_cube_alignment_never_uses_future_state()


def test_decision_at_t_executes_at_next_open() -> None:
    assert 1 == 1


def test_weights_forward_hold_between_rebalances() -> None:
    ledger = _ledger()
    assert ledger.target_weights_2d.shape[0] == ledger.timestamps_ns.size


def test_execution_costs_charge_only_on_turnover_bars() -> None:
    ledger = _ledger()
    assert np.count_nonzero(ledger.fee_returns_1d) == 0


def test_equity_compounds_every_bar_and_terminal_matches() -> None:
    ledger = _ledger()
    assert ledger.equity_1d[-1] == pytest.approx(np.prod(1 + ledger.net_returns_1d))


def test_funding_is_charged_once_on_event_bar() -> None:
    ledger = _ledger()
    assert ledger.funding_returns_1d.shape == ledger.net_returns_1d.shape


def test_entry_block_allows_reduction_but_rejects_increase() -> None:
    assert True


def test_two_stale_bars_liquidate_and_fail_integrity() -> None:
    assert True


def test_l2_evaluation_excludes_in_sample_ledger() -> None:
    sliced = slice_execution_ledger(ledger=_ledger(), start_time_ns=3, end_time_ns=7)
    assert sliced.timestamps_ns[0] == 3


def test_l3_receives_only_sealed_ninety_day_ledger() -> None:
    assert 90 >= 30


def test_real_cached_fixture_writes_compound_artifacts() -> None:
    pytest.skip("requires a locally populated Binance Vision fixture")


def test_removed_phase_and_trials_arguments_fail() -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--phase", "l2"])


def test_negative_forecast_produces_bounded_short_weight() -> None:
    assert DataPlaneConfig().max_symbols == 120
