from __future__ import annotations

import argparse

import pytest

from src.application.futures.runner.cli import build_arg_parser, check_removed_flags, run_from_cli
from src.application.futures.runner.compound_config import (
    CompoundRunConfig,
    build_compound_run_config,
)
from src.application.futures.runner.compound_main import run_compound_main
from src.domain.futures.compound.validation import slice_execution_ledger


def _make_ledger() -> object:
    import numpy as np
    from src.domain.futures.compound.contracts import ExecutionLedger

    n = 100
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.random.default_rng(42).normal(0.0005, 0.02, n - 1)
    equity = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        equity[i] = equity[i - 1] * (1.0 + returns[i])
    ts = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    return ExecutionLedger(
        timestamps_ns=ts,
        net_returns_1d=returns,
        equity_1d=equity,
        target_weights_2d=np.zeros((n, 3), dtype=np.float32),
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True,
        integrity_reasons=(),
    )


# ── CompoundRunConfig ──────────────────────────────────────────────────────────


class TestBuildCompoundRunConfig:
    def test_defaults(self) -> None:
        args = argparse.Namespace(date=None, sync="auto", refresh_universe=False, seed=42, max_rss_mb=12_000)
        config = build_compound_run_config(args)
        assert config.reference_date is None
        assert config.sync == "auto"
        assert config.refresh_universe is False
        assert config.seed == 42
        assert config.base_timeframe == "1h"
        assert config.max_rss_mb == 12_000

    def test_with_date_and_skip_sync(self) -> None:
        args = argparse.Namespace(date="2026-07-08", sync="skip", refresh_universe=True, seed=99, max_rss_mb=6000)
        config = build_compound_run_config(args)
        assert config.reference_date == "2026-07-08"
        assert config.sync == "skip"
        assert config.refresh_universe is True
        assert config.seed == 99
        assert config.max_rss_mb == 6000

    def test_invalid_sync_raises(self) -> None:
        args = argparse.Namespace(sync="invalid")
        with pytest.raises(ValueError, match="invalid sync mode"):
            build_compound_run_config(args)

    def test_negative_max_rss_raises(self) -> None:
        args = argparse.Namespace(date=None, sync="auto", refresh_universe=False, seed=42, max_rss_mb=-1)
        with pytest.raises(ValueError, match="max_rss_mb must be positive"):
            build_compound_run_config(args)

    def test_missing_date_is_optional(self) -> None:
        args = argparse.Namespace(date=None, sync="skip", refresh_universe=False, seed=42, max_rss_mb=12_000)
        config = build_compound_run_config(args)
        assert config.reference_date is None


# ── CLI compound-only flags ───────────────────────────────────────────────────


class TestCliCompoundOnly:
    def test_build_arg_parser_accepts_only_allowed_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--date", "2026-07-08", "--sync", "skip", "--seed", "42"])
        assert args.date == "2026-07-08"
        assert args.sync == "skip"
        assert args.seed == 42

    def test_check_removed_flags_does_not_raise_for_allowed(self) -> None:
        args = argparse.Namespace(date="2026-07-08", sync="skip", phase=None, trials=None, timeframe=None)
        check_removed_flags(args)

    def test_check_removed_flags_raises_for_removed_flag(self) -> None:
        args = argparse.Namespace(phase="l3")
        with pytest.raises(SystemExit):
            check_removed_flags(args)

    def test_check_removed_flags_raises_for_trials(self) -> None:
        args = argparse.Namespace(trials=42)
        with pytest.raises(SystemExit):
            check_removed_flags(args)

    def test_run_from_cli_returns_2_for_removed_flag(self, mocker) -> None:
        mocker.patch("src.application.futures.runner.cli.run_compound_main")
        result = run_from_cli(["--phase", "l3"])
        assert result == 2

    def test_run_from_cli_returns_2_for_unknown_flag(self, mocker) -> None:
        mocker.patch("src.application.futures.runner.cli.run_compound_main")
        result = run_from_cli(["--unknown-flag"])
        assert result == 2


# ── run_compound_main (mocked) ────────────────────────────────────────────────


class TestRunCompoundMain:
    def test_mocked_invocation(self, mocker) -> None:
        import pandas as pd

        sample_df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
                                   "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0})
        mocker.patch("src.application.futures.runner.compound_main.load_hourly_data",
                      return_value={"BTCUSDT": {"1h": sample_df}, "ETHUSDT": {"1h": sample_df}})
        mocker.patch("src.application.futures.runner.compound_main.check_data_readiness",
                      return_value=True)
        import numpy as np
        from src.domain.futures.compound.contracts import (
            AlphaForecastTape,
            DeploymentVerdict,
            ExecutionLedger,
            L2Evaluation,
            L3ValidationResult,
            MarketFeatureCube,
        )

        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1, 2], dtype=np.int64)
        mock_cube.data_manifest_hash = "test-hash"
        mock_build_cube = mocker.patch(
            "src.application.futures.runner.compound_main.build_compound_market_feature_cube",
            return_value=mock_cube,
        )

        mock_run = mocker.patch(
            "src.application.futures.runner.compound_main._run_engine_from_loaded_data",
        )

        mock_ledger = mocker.Mock(spec=ExecutionLedger)
        mock_ledger.equity_1d = np.array([1.0, 1.1])
        mock_ledger.timestamps_ns = np.array([0, 1], dtype=np.int64)
        mock_ledger.target_weights_2d = np.zeros((2, 2), dtype=np.float32)
        mock_ledger.net_returns_1d = np.array([0.0, 0.1], dtype=np.float64)
        mock_ledger.fee_returns_1d = np.zeros(2, dtype=np.float64)
        mock_ledger.slippage_returns_1d = np.zeros(2, dtype=np.float64)
        mock_ledger.impact_returns_1d = np.zeros(2, dtype=np.float64)
        mock_ledger.funding_returns_1d = np.zeros(2, dtype=np.float64)
        mock_ledger.integrity_ok = True
        mock_ledger.integrity_reasons = ()

        mock_l2 = mocker.Mock(spec=L2Evaluation)
        mock_l2.annualized_log_growth = 0.05
        mock_l2.growth_ci90 = (0.01, 0.09)
        mock_l2.equity_multiple = 1.1
        mock_l2.max_drawdown = 0.02
        mock_l2.daily_cvar95 = -0.01
        mock_l2.annual_volatility = 0.15
        mock_l2.turnover = 0.5
        mock_l2.safe = True
        mock_l2.integrity_ok = True

        mock_l3 = mocker.Mock(spec=L3ValidationResult)
        mock_l3.verdict = DeploymentVerdict.SHADOW
        mock_l3.posterior_growth_probability = 0.5
        mock_l3.holdout_days = 90
        mock_l3.max_drawdown = 0.02
        mock_l3.daily_cvar95 = -0.01
        mock_l3.reasons = ()

        mock_tape = mocker.Mock(spec=AlphaForecastTape)
        mock_tape.model_version = "test-v1"
        mock_tape.data_manifest_hash = "test-hash"

        engine_result = mocker.Mock()
        engine_result.ledger = mock_ledger
        engine_result.l2 = mock_l2
        engine_result.l3 = mock_l3
        engine_result.alpha_tape = mock_tape
        mock_run.return_value = engine_result

        config = CompoundRunConfig(reference_date="2026-07-08", sync="skip", refresh_universe=False)
        result = run_compound_main(config)
        mock_run.assert_called_once()
        assert result.exit_code == 0


# ── slice_execution_ledger ─────────────────────────────────────────────────────


class TestSliceExecutionLedger:
    def test_slice_middle_returns_copied_ledger(self) -> None:

        ledger = _make_ledger()
        ts = ledger.timestamps_ns
        mid_start = ts[20]
        mid_end = ts[50]
        sliced = slice_execution_ledger(ledger=ledger, start_time_ns=mid_start, end_time_ns=mid_end)
        assert sliced.timestamps_ns.shape[0] == 31
        assert sliced.equity_1d[0] == ledger.equity_1d[20]
        assert sliced.equity_1d[-1] == ledger.equity_1d[50]
        assert sliced.integrity_ok is True
        assert len(sliced.net_returns_1d) == 31
        assert len(sliced.target_weights_2d) == 31

    def test_slice_raises_for_empty_range(self) -> None:

        ledger = _make_ledger()
        with pytest.raises(ValueError, match="empty slice"):
            slice_execution_ledger(ledger=ledger, start_time_ns=ledger.timestamps_ns[-1] + 1, end_time_ns=ledger.timestamps_ns[-1] + 2)

    def test_slice_raises_for_empty_ledger(self) -> None:
        import numpy as np
        from src.domain.futures.compound.contracts import ExecutionLedger

        empty = ExecutionLedger(
            timestamps_ns=np.array([], dtype=np.int64),
            net_returns_1d=np.array([], dtype=np.float64),
            equity_1d=np.array([], dtype=np.float64),
            target_weights_2d=np.empty((0, 0), dtype=np.float32),
            fee_returns_1d=np.array([], dtype=np.float64),
            slippage_returns_1d=np.array([], dtype=np.float64),
            impact_returns_1d=np.array([], dtype=np.float64),
            funding_returns_1d=np.array([], dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )
        with pytest.raises(ValueError, match="cannot slice empty ledger"):
            slice_execution_ledger(ledger=empty, start_time_ns=0, end_time_ns=1)

    def test_memory_independence(self) -> None:

        ledger = _make_ledger()
        sliced = slice_execution_ledger(ledger=ledger, start_time_ns=ledger.timestamps_ns[10], end_time_ns=ledger.timestamps_ns[20])
        original_val = float(sliced.equity_1d[0])
        sliced.equity_1d[0] = 999.0
        assert float(sliced.equity_1d[0]) != float(ledger.equity_1d[10])
        assert float(ledger.equity_1d[10]) == pytest.approx(original_val, rel=1e-12)


# ── legacy import census (PERF-07) ───────────────────────────────────────────


def test_no_legacy_imports_in_retained_source() -> None:
    retained = (
        "src/execution/opt_main_futures.py",
        "src/application/futures/runner/cli.py",
        "src/application/futures/runner/compound_config.py",
        "src/application/futures/runner/compound_main.py",
        "src/application/futures/runner/compound_universe.py",
        "src/application/futures/runner/compound_data.py",
        "src/domain/futures/compound",
    )
    disallowed = (
        "active_pipeline",
        "alpha_foundry",
        "tiered_workflow",
        "strategy_runtime",
        "online_growth_allocator",
        "policy_shadow_book",
    )
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", """
import ast, sys
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names if isinstance(node, ast.Import) else [node]:
                    name = (alias.name if isinstance(node, ast.Import) else node.module) or ""
                    print(f"{path}:::{name}")
    except Exception:
        pass
"""],
        input="\n".join(retained),
        capture_output=True, text=True,
    )
    imports = result.stdout.strip().split("\n") if result.stdout.strip() else []
    offending = [line for line in imports if any(d in line.split(":::")[1] for d in disallowed)]
    assert not offending, f"legacy imports found in retained source: {offending}"
