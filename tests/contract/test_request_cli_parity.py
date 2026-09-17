"""I3 REQUEST/CLI DECLARE-ONCE contract.

The argparse flag set exposed by the mhs CLI must equal the flag set derived
from ``MhsDiagnosticRequest`` field ``cli`` metadata. Every CLI-exposed request
field carries that metadata, so adding one execution option requires editing
exactly one field.
"""

from __future__ import annotations

import argparse
import dataclasses

from src.mhs.evaluation import MhsDiagnosticRequest
from src.cli.commands.research.mhs import add_mhs_commands
from src.cli.dataclass_args import build_parser_from_dataclass


def _flag_set(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in ("-h", "--help")
    }


def _mhs_cli_flags() -> set[str]:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    flags = _flag_set(parser)
    # The output tier is a persistence switch on the parser, not a request field.
    flags.discard("--output-tier")
    # SCENARIO_MHS_LEVERAGE_SCAN_07: diagnostic-only short-circuit switches,
    # never construct MhsDiagnosticRequest.
    flags.discard("--leverage-frontier-scan")
    flags.discard("--leverage-frontier-multiples")
    # 연구-라이브 seam 스위치: 완료된 리포트를 사후 소비할 뿐 요청 필드가 아니다.
    flags.discard("--emit-target-weights")
    flags.discard("--emit-signal-state")
    # 배포 후처리 사이드 이펙트 스위치: 완료된 리포트를 소비할 뿐 요청 필드가 아니다.
    flags.discard("--emit-deployment")
    flags.discard("--deploy-push")
    # 절차 등록 사이드 이펙트 스위치: 플래그 세트를 레지스트리에 동결할 뿐 요청 필드가 아니다.
    flags.discard("--register-procedure")
    return flags


def _metadata_flags() -> set[str]:
    parser = argparse.ArgumentParser()
    build_parser_from_dataclass(parser, MhsDiagnosticRequest)
    return _flag_set(parser)


def test_cli_flags_equal_metadata_derived_flags() -> None:
    cli_flags = _mhs_cli_flags()
    meta_flags = _metadata_flags()
    assert cli_flags == meta_flags, (
        f"CLI/metadata flag divergence: only_cli={sorted(cli_flags - meta_flags)} "
        f"only_metadata={sorted(meta_flags - cli_flags)}"
    )


def test_every_cli_exposed_field_carries_flag_metadata() -> None:
    for field in dataclasses.fields(MhsDiagnosticRequest):
        if "flag" in field.metadata:
            assert field.metadata["flag"], f"field {field.name} has empty flag metadata"


def test_pnl_vol_target_mode_choices_match_cli_and_metadata() -> None:
    """SCENARIO_MHS_CONSTANT_RISK_REQUEST_CLI_PARITY: the request contract's
    cli_param choices and the hand-written CLI argparse choices for
    --pnl-vol-target-mode stay exactly equal (4 registered values)."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    cli_action = next(a for a in parser._actions if a.dest == "pnl_vol_target_mode")
    field = next(
        f for f in dataclasses.fields(MhsDiagnosticRequest)
        if f.name == "pnl_vol_target_mode"
    )
    meta_choices = field.metadata["choices"]
    # 선언 순서는 계약/CLI 간 다를 수 있으므로 등록 값집합의 정확한 일치를 단언한다.
    assert sorted(cli_action.choices) == sorted(meta_choices)
    assert len(meta_choices) == 4
    assert "constant_risk" in meta_choices


def test_scenario_mhs_dd_brake_10_cli_request_parity() -> None:
    """SCENARIO_MHS_DD_BRAKE_10_CLI_REQUEST_PARITY: --exposure-drawdown-brake is declared exactly
    once on the request (cli_param metadata) and mirrored by the hand-written
    CLI; the MhsRunConfig no-arg parity stays intact with brake=False."""
    cli_flags = _mhs_cli_flags()
    meta_flags = _metadata_flags()
    assert "--exposure-drawdown-brake" in cli_flags
    assert "--exposure-drawdown-brake" in meta_flags

    from src.cli.main import build_root_parser
    from src.mhs.pipeline.config import MhsRunConfig

    args = build_root_parser().parse_args(
        ["research", "run", "portfolio", "mhs-horizon-diagnostic"],
    )
    assert args.exposure_drawdown_brake is False
    assert (
        dataclasses.asdict(MhsRunConfig.from_namespace(args))
        == dataclasses.asdict(MhsRunConfig())
    )
    assert dataclasses.asdict(MhsRunConfig())["exposure_drawdown_brake"] is False


def test_data_policy_choices_match_cli_and_metadata() -> None:
    import argparse
    import dataclasses

    from src.cli.commands.research.mhs import add_mhs_commands
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
    from src.mhs.evaluation import MhsDiagnosticRequest
    from src.mhs.panel import DATA_POLICIES

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    cli_action = next(a for a in parser._actions if a.dest == "data_policy")
    field = next(f for f in dataclasses.fields(MhsDiagnosticRequest) if f.name == "data_policy")

    assert cli_action.default == MHS_DATA_POLICY_DEFAULT
    assert MhsDiagnosticRequest().data_policy == MHS_DATA_POLICY_DEFAULT
    assert set(cli_action.choices) == set(DATA_POLICIES)
    assert set(field.metadata["choices"]) == set(DATA_POLICIES)


def test_request_default_execution_is_3m() -> None:
    import dataclasses

    from src.mhs.evaluation import MhsDiagnosticRequest

    request = MhsDiagnosticRequest()
    assert request.execution_timeframe == "3m"
    field = next(f for f in dataclasses.fields(MhsDiagnosticRequest) if f.name == "execution_timeframe")
    assert field.metadata["choices"] == ("3m",)
    assert dataclasses.asdict(request)["execution_timeframe"] == "3m"


def test_request_rejects_legacy_execution_intervals() -> None:
    import pytest

    from src.mhs.evaluation import MhsDiagnosticRequest

    for legacy in ("1m", "5m"):
        with pytest.raises(ValueError, match="execution_timeframe"):
            MhsDiagnosticRequest(execution_timeframe=legacy)  # type: ignore[arg-type]


def test_request_timeout_must_align_to_three_minutes() -> None:
    import pytest

    from src.mhs.evaluation import MhsDiagnosticRequest

    MhsDiagnosticRequest(passive_timeout_minutes=30)
    with pytest.raises(ValueError, match="multiple of 3"):
        MhsDiagnosticRequest(passive_timeout_minutes=31)


def test_deployment_policy_converts_3m_and_rejects_legacy() -> None:
    import pytest

    from src.mhs.deployment_policy import TargetWeightPolicy
    from src.mhs.evaluation import MhsDiagnosticRequest

    base = {
        "execution_universe_size": 8,
        "fast_book_mode": "single_horizon",
        "slow_book_mode": "single_horizon",
        "rebalance_filter": "per_symbol_deadband",
        "beta_neutralize": False,
        "ensemble_signal": "raw",
        "trend_efficiency_overlay": False,
        "trend_sleeve": False,
        "trend_sleeve_gross": 0.0,
        "crash_regime_tilt_alpha": None,
        "committee_capital": False,
        "committee_member_set": "risk_premia",
        "committee_tranche_smoothing": False,
        "committee_regime_adaptive_tranche": False,
        "committee_target_gross": None,
        "funding_carry_sleeve": False,
        "funding_carry_weight": 0.0,
        "fill_mark_parity_gate": True,
    }
    request = TargetWeightPolicy(execution_timeframe="3m", **base).to_request()  # type: ignore[arg-type]
    assert isinstance(request, MhsDiagnosticRequest)
    assert request.execution_timeframe == "3m"
    for legacy in ("1m", "5m"):
        with pytest.raises(ValueError, match="execution_timeframe"):
            TargetWeightPolicy(execution_timeframe=legacy, **base).to_request()  # type: ignore[arg-type]


def test_execution_grids_use_three_minute_steps() -> None:
    import pandas as pd

    from src.mhs.evaluation import MhsDiagnosticRequest
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    request = MhsDiagnosticRequest()
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01T01:00:00", tz="UTC")
    empty_weights = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    empty_signals = pd.DatetimeIndex([], tz="UTC")
    windows = list(
        _iter_mhs_execution_windows(
            empty_weights, empty_signals, "/nonexistent", "3m",
            start, end, {}, request.mark_mode, ExecutionSpec(),
        )
    )
    assert len(windows) == 1
    grid = windows[0].minute_grid
    assert len(grid) == 21
    assert (grid[1] - grid[0]) == pd.Timedelta(minutes=3)


def test_marks_align_on_three_minute_grid() -> None:
    import pandas as pd

    from src.mhs.marks import _align_minute_frames

    idx = pd.date_range("2025-01-01", periods=4, freq="3min", tz="UTC")
    frame = pd.DataFrame({"high": [1.0, 2.0, 3.0, 4.0], "low": [1.0, 2.0, 3.0, 4.0], "close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    result = _align_minute_frames({"AAA": frame}, "3m", idx[0], idx[-1])
    assert result is not None
    highs, _, _ = result
    assert (highs.index[1] - highs.index[0]) == pd.Timedelta(minutes=3)


def test_missing_execution_cache_rejected_without_fallback() -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.market_data.services import mhs_execution as mec

    assert mec._coverage("AAA", "3m", "2025-01-01", "2025-01-02", root="/nonexistent")["status"] == "MISSING"
    with pytest.raises(DataIntegrityError, match="incomplete"):
        mec.assert_execution_data_coverage(["AAA"], "3m", "2025-01-01", "2025-01-02", root="/nonexistent")
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


def test_execution_manifest_refresh_rejects_legacy_interval(tmp_path) -> None:
    import json

    import pytest

    from src.market_data.services import mhs_execution as mec

    manifest = tmp_path / "plan.json"
    manifest.write_text(
        json.dumps({"timeframe": "1m", "start": "2025-01-01", "end": "2025-01-02", "symbols": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.refresh_mhs_execution_manifest(manifest)


def test_hourly_and_generic_contracts_preserved() -> None:
    import pandas as pd
    import pytest

    import src.market_data.services.futures_collection as fc
    from src.market_data.services import mhs_execution as mec

    assert fc._TIMEFRAME_MS["1m"] == 60_000
    assert fc._TIMEFRAME_MS["5m"] == 300_000
    with pytest.raises(ValueError, match="hourly"):
        mec.assert_relevant_mark_price_coverage(pd.DataFrame(), timeframe="3m")
    empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    mec.apply_dynamic_gap_exclusion(empty, "3m")
    mec.apply_dynamic_gap_exclusion(empty, "1h")
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.apply_dynamic_gap_exclusion(empty, "1m")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_timeframe"):
        mec.apply_dynamic_gap_exclusion(empty, "5m")  # type: ignore[arg-type]


def test_sealed_inputs_require_explicit_3m() -> None:
    from pathlib import Path

    from src.mhs.data_provenance import resolve_required_mhs_input_paths

    paths = resolve_required_mhs_input_paths(
        data_root=Path("/data"), panel_symbols=["AAA"], execution_symbols=["AAA"], execution_timeframe="3m",
    )
    assert Path("ohlcv/3m/AAA.parquet") in [Path(p.parent.name) / p.name for p in paths] or any(
        "ohlcv/3m" in p.as_posix() for p in paths
    )
