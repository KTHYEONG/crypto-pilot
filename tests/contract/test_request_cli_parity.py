"""I-NOARG-PARITY contract: generated MHS CLI is the production path."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from src.cli.commands.research.mhs import add_mhs_commands
from src.cli.dataclass_args import explicit_field_values
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.pipeline.config import resolve_cli_request

BASE = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]

FROZEN_FLAGS = frozenset(
    [
        "--beta-neutralize",
        "--committee-book",
        "--committee-growth-diagnostic",
        "--committee-member-attribution",
        "--committee-member-set",
        "--committee-target-gross",
        "--committee-tranche-count",
        "--committee-tranche-smoothing",
        "--crash-regime-tilt-alpha",
        "--data-policy",
        "--discovery-gate",
        "--discovery-gate-adjusted-net-t",
        "--discovery-gate-regime-scaled-net-t",
        "--end",
        "--ensemble-signal",
        "--execution-coverage-gate",
        "--execution-timeframe",
        "--execution-universe-size",
        "--exposure-drawdown-brake",
        "--fast-book-mode",
        "--final-oos-2026h1",
        "--fold-safe-horizon",
        "--forward-execution-quality-dir",
        "--forward-registration",
        "--forward-strategy-digest",
        "--funding-carry-weight",
        "--growth-envelope",
        "--input-manifest-path",
        "--ladder-diagnostic",
        "--leverage-frontier-multiples",
        "--leverage-frontier-scan",
        "--liquidity-cost-model",
        "--max-rss-bytes",
        "--multi-feature-book",
        "--name-drift-trim",
        "--no-committee-capital",
        "--no-committee-evidence-weighting",
        "--no-committee-kelly-sizing",
        "--no-committee-regime-adaptive-tranche",
        "--no-committee-target-gross",
        "--no-exposure-scale-two-sided",
        "--no-funding-carry-sleeve",
        "--no-log-run",
        "--no-pnl-vol-target",
        "--no-ram-guard",
        "--output-tier",
        "--passive-timeout-minutes",
        "--peg-chase-diagnostic",
        "--phase-diagnostic",
        "--placebo-diagnostic",
        "--signal-48h-diagnostic",
        "--bootstrap-ci-diagnostic",
        "--reference-books-diagnostic",
        "--patient-reference-diagnostic",
        "--pnl-vol-target-mode",
        "--rebalance-filter",
        "--register-procedure",
        "--run-id",
        "--slow-book-mode",
        "--start",
        "--touch-diagnostic",
        "--trend-efficiency-overlay",
        "--trend-sleeve",
        "--trend-sleeve-gross",
    ]
)


def _mhs_parser() -> argparse.ArgumentParser:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    return sub.choices["mhs-horizon-diagnostic"]


def _flag_set(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in ("-h", "--help")
    }


def _parse(extra: list[str]) -> argparse.Namespace:
    from src.cli.main import build_root_parser

    return build_root_parser().parse_args([*BASE, *extra])


def test_exposed_option_set_frozen() -> None:
    assert _flag_set(_mhs_parser()) == set(FROZEN_FLAGS)
    assert len(FROZEN_FLAGS) == 64


def test_no_arg_parity() -> None:
    from src.cli.main import build_root_parser

    args = build_root_parser().parse_args(BASE)
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit == {}
    assert resolve_cli_request(explicit) == MhsDiagnosticRequest()
    req = resolve_cli_request(explicit)
    assert req.placebo_diagnostic is False
    assert req.phase_diagnostic is False
    assert req.signal_48h_diagnostic is False
    assert req.bootstrap_ci_diagnostic is False
    assert req.reference_books_diagnostic is False
    assert req.patient_reference_diagnostic is False


def _value_cases() -> list[tuple[str, list[str], str, object]]:
    return [
        ("start", ["--start", "2025-01-01"], "start", "2025-01-01"),
        ("end", ["--end", "2025-01-01"], "end", "2025-01-01"),
        ("execution_timeframe", ["--execution-timeframe", "3m"], "execution_timeframe", "3m"),
        ("execution_universe_size", ["--execution-universe-size", "60"], "execution_universe_size", 60),
        ("max_rss_bytes", ["--max-rss-bytes", "8000000000"], "max_rss_bytes", 8000000000),
        ("liquidity_cost_model", ["--liquidity-cost-model", "flat"], "liquidity_cost_model", "flat"),
        ("passive_timeout_minutes", ["--passive-timeout-minutes", "33"], "passive_timeout_minutes", 33),
        ("crash_regime_tilt_alpha", ["--crash-regime-tilt-alpha", "0.5"], "crash_regime_tilt_alpha", 0.5),
        ("slow_book_mode", ["--slow-book-mode", "single_horizon"], "slow_book_mode", "single_horizon"),
        ("fast_book_mode", ["--fast-book-mode", "single_horizon"], "fast_book_mode", "single_horizon"),
        ("rebalance_filter", ["--rebalance-filter", "per_symbol_deadband"], "rebalance_filter", "per_symbol_deadband"),
        ("ensemble_signal", ["--ensemble-signal", "raw"], "ensemble_signal", "raw"),
        ("pnl_vol_target_mode", ["--pnl-vol-target-mode", "median_relative"], "pnl_vol_target_mode", "median_relative"),
        ("trend_sleeve_gross", ["--trend-sleeve", "--trend-sleeve-gross", "0.2"], "trend_sleeve_gross", 0.2),
        ("committee_member_set", ["--committee-member-set", "flow_momentum"], "committee_member_set", "flow_momentum"),
        ("committee_tranche_count", ["--committee-tranche-count", "5"], "committee_tranche_count", 5),
        ("committee_target_gross", ["--committee-target-gross", "1.2"], "committee_target_gross", 1.2),
        ("funding_carry_weight", ["--funding-carry-weight", "0.25"], "funding_carry_weight", 0.25),
        ("growth_envelope", ["--growth-envelope", "balanced"], "growth_envelope", "balanced"),
        ("data_policy", ["--data-policy", "legacy"], "data_policy", "legacy"),
        ("input_manifest_path", ["--input-manifest-path", "/x"], "input_manifest_path", "/x"),
        (
            "forward_execution_quality_dir",
            ["--forward-execution-quality-dir", "/x"],
            "forward_execution_quality_dir",
            "/x",
        ),
        ("forward_strategy_digest", ["--forward-strategy-digest", "a" * 32], "forward_strategy_digest", "a" * 32),
        (
            "forward_registration_digest",
            ["--forward-registration", "a" * 32, "--end", "2025-12-31"],
            "forward_registration_digest",
            "a" * 32,
        ),
    ]


@pytest.mark.parametrize(("label", "argv", "field", "value"), _value_cases())
def test_every_value_flag_round_trips(label: str, argv: list[str], field: str, value: object) -> None:
    args = _parse(argv)
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    if field == "trend_sleeve_gross":
        assert explicit == {"trend_sleeve": True, "trend_sleeve_gross": value}
    elif field == "forward_registration_digest":
        assert explicit[field] == value
        assert explicit["end"] == "2025-12-31"
    else:
        assert explicit == {field: value}


def test_switch_polarity_matches_defaults() -> None:
    for f in dataclasses.fields(MhsDiagnosticRequest):
        meta = f.metadata
        if not meta.get("flag") or not isinstance(f.default, bool):
            continue
        assert (meta["flag"].startswith("--no-")) == (f.default is True), f.name
        args = _parse([meta["flag"]])
        explicit = explicit_field_values(MhsDiagnosticRequest, args)
        assert explicit == {f.name: (not f.default)}, f.name


def test_gross_opt_out_pair_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse(["--committee-target-gross", "1.2", "--no-committee-target-gross"])
    args = _parse(["--no-committee-target-gross"])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit == {"committee_target_gross": None}
    req = resolve_cli_request(explicit)
    assert req.committee_target_gross is None
    assert req.funding_carry_sleeve is False
    assert req.funding_carry_weight == 0.0


def test_metadata_key_set_closed() -> None:
    for f in dataclasses.fields(MhsDiagnosticRequest):
        meta = f.metadata
        if not meta.get("flag"):
            continue
        assert set(meta.keys()) == {"flag", "help", "choices", "negate_flag", "arg_type"}, f.name
        assert meta["help"]


def test_choices_come_from_registries() -> None:
    from src.core.panel import DATA_POLICIES
    from src.core.params import (
        CLI_GROWTH_ENVELOPE_DEFAULT,
        COMMITTEE_MEMBER_SETS,
        GROWTH_RISK_ENVELOPES,
    )

    by_name = {f.name: f for f in dataclasses.fields(MhsDiagnosticRequest)}
    assert by_name["growth_envelope"].metadata["choices"] == tuple(sorted(GROWTH_RISK_ENVELOPES))
    assert CLI_GROWTH_ENVELOPE_DEFAULT in by_name["growth_envelope"].metadata["choices"]
    assert by_name["committee_member_set"].metadata["choices"] == tuple(sorted(COMMITTEE_MEMBER_SETS))
    assert set(by_name["data_policy"].metadata["choices"]) == set(DATA_POLICIES)
    assert len(by_name["pnl_vol_target_mode"].metadata["choices"]) == 4
    assert "constant_risk" in by_name["pnl_vol_target_mode"].metadata["choices"]
    for f in dataclasses.fields(MhsDiagnosticRequest):
        choices = f.metadata.get("choices")
        if choices is not None:
            assert f.default in choices, f.name


def test_generator_is_production_path() -> None:
    import pathlib

    import src.cli.dataclass_args as da

    text = pathlib.Path("src/cli/commands/research/mhs.py").read_text(encoding="utf-8")
    assert "add_dataclass_arguments" in text
    assert 'add_argument("--start"' not in text
    assert not hasattr(da, "build_parser_from_dataclass")
    assert not hasattr(da, "request_from_namespace")


def test_request_default_execution_is_3m() -> None:
    import dataclasses

    from src.mhs.contracts import MhsDiagnosticRequest

    request = MhsDiagnosticRequest()
    assert request.execution_timeframe == "3m"
    field = next(f for f in dataclasses.fields(MhsDiagnosticRequest) if f.name == "execution_timeframe")
    assert field.metadata["choices"] == ("3m",)
    assert dataclasses.asdict(request)["execution_timeframe"] == "3m"


def test_request_rejects_legacy_execution_intervals() -> None:
    import pytest

    from src.mhs.contracts import MhsDiagnosticRequest

    for legacy in ("1m", "5m"):
        with pytest.raises(ValueError, match="execution_timeframe"):
            MhsDiagnosticRequest(execution_timeframe=legacy)  # type: ignore[arg-type]


def test_request_timeout_must_align_to_three_minutes() -> None:
    import pytest

    from src.mhs.contracts import MhsDiagnosticRequest

    MhsDiagnosticRequest(passive_timeout_minutes=30)
    with pytest.raises(ValueError, match="multiple of 3"):
        MhsDiagnosticRequest(passive_timeout_minutes=31)


def test_request_default_3m_rejects_one_minute() -> None:
    import pytest

    from src.mhs.contracts import MhsDiagnosticRequest
    from tests.fixtures.mhs_requests import research_baseline

    request = research_baseline(committee_capital=True)
    assert request.execution_timeframe == "3m"
    with pytest.raises(ValueError, match="execution_timeframe"):
        MhsDiagnosticRequest(execution_timeframe="1m")  # type: ignore[arg-type]


def test_execution_grids_use_three_minute_steps() -> None:
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.core.types import ExecutionSpec

    request = MhsDiagnosticRequest()
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01T01:00:00", tz="UTC")
    empty_weights = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    empty_signals = pd.DatetimeIndex([], tz="UTC")
    windows = list(
        _iter_mhs_execution_windows(
            empty_weights, empty_signals, "/nonexistent", "3m",
            start, end, {}, ExecutionSpec(),
        )
    )
    assert len(windows) == 1
    grid = windows[0].minute_grid
    assert len(grid) == 20
    assert grid[-1] == end - pd.Timedelta(minutes=3)
    assert (grid[1] - grid[0]) == pd.Timedelta(minutes=3)


def test_marks_align_on_three_minute_grid() -> None:
    import pandas as pd

    from src.core.marks import _align_minute_frames

    idx = pd.date_range("2025-01-01", periods=4, freq="3min", tz="UTC")
    frame = pd.DataFrame({"high": [1.0, 2.0, 3.0, 4.0], "low": [1.0, 2.0, 3.0, 4.0], "close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    result = _align_minute_frames({"AAA": frame}, "3m", idx[0], idx[-1])
    assert result is not None
    highs, _, _ = result
    assert (highs.index[1] - highs.index[0]) == pd.Timedelta(minutes=3)


def test_missing_execution_cache_rejected_without_fallback() -> None:
    import pytest

    from src.market_data.services import mhs_execution as mec

    assert mec._coverage("AAA", "3m", "2025-01-01", "2025-01-02", root="/nonexistent")["status"] == "MISSING"
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec._coverage("AAA", "1m", "2025-01-01", "2025-01-02", root="/nonexistent")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.build_mhs_execution_plan("2025-01-01", "2025-01-02", timeframe="5m")  # type: ignore[arg-type]


def test_execution_coverage_counts_three_minute_bars(tmp_path) -> None:
    import pandas as pd

    from src.market_data.services import mhs_execution as mec

    root = tmp_path / "ohlcv" / "3m"
    root.mkdir(parents=True)
    stamps = pd.date_range("2025-01-01", periods=4, freq="3min", tz="UTC")
    pd.DataFrame({"timestamp": (stamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")}).to_parquet(
        root / "AAA.parquet",
    )
    result = mec._coverage("AAA", "3m", "2025-01-01T00:00:00Z", "2025-01-01T00:09:00Z", root=str(tmp_path / "ohlcv"))
    assert result["status"] == "PRESENT"
    assert result["rows"] == 4


def test_hourly_and_generic_contracts_preserved() -> None:
    import pandas as pd
    import pytest

    import src.market_data.services.futures_collection as fc
    from src.market_data.services import mhs_execution as mec

    assert fc._TIMEFRAME_MS["1m"] == 60_000
    assert fc._TIMEFRAME_MS["5m"] == 300_000
    empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    mec.apply_dynamic_gap_exclusion(empty, "3m")
    mec.apply_dynamic_gap_exclusion(empty, "1h")
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.apply_dynamic_gap_exclusion(empty, "1m")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.apply_dynamic_gap_exclusion(empty, "5m")  # type: ignore[arg-type]


def test_sealed_inputs_require_explicit_3m() -> None:
    from pathlib import Path

    from src.core.data_provenance import mhs_input_layout_for_lake, resolve_required_mhs_input_paths

    paths = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(Path("/data")), panel_symbols=["AAA"], execution_symbols=["AAA"], execution_timeframe="3m",
    )
    assert Path("ohlcv/3m/AAA.parquet") in [Path(p.parent.name) / p.name for p in paths] or any(
        "ohlcv/3m" in p.as_posix() for p in paths
    )
