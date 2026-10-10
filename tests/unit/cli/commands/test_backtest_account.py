"""Invariant scenarios for the strategy account CLI leaves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
from tests.unit.cli.commands._backtest_helpers import (
    _account_argv,
    _assemble_fixture,
    _install_account,
    _install_strategy,
    _parse,
    _run_dirs,
)
from tests.unit.application._strategy_account_helpers import (
    _fake_run_dir,
    _install_exposure_fakes as _install_exposure,
    _write_same_book_reference,
)


@pytest.fixture(autouse=True)
def _isolated_release_ledgers(tmp_path, monkeypatch):
    import shutil
    import src.strategy.release as release_mod

    source = release_mod.release_path("flow_mom_top20")
    root = tmp_path / "default_releases"
    root.mkdir()
    shutil.copy(source, root / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root_arg=None: root)

def test_account_command_defaults_to_growth_at_retail_capital(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No policy/capital flags select growth at the declared minimum retail start."""
    from src.core.params import (
        ACCOUNT_DEFAULT_CAPITAL_USDT,
        ACCOUNT_EXPOSURE_CAP,
        ACCOUNT_EXPOSURE_STEP,
        ACCOUNT_IMPACT_Y,
        ACCOUNT_INITIAL_MARGIN_CAP,
        ACCOUNT_MARGIN_RESERVE,
        ACCOUNT_MEAN_HAIRCUT,
        ACCOUNT_MIN_MOMENT_DAYS,
        ACCOUNT_PRIOR_DAYS,
        ACCOUNT_SHOCK_PER_UNIT,
        ACCOUNT_TAKER_FEE_BPS,
        ACCOUNT_UNIT_REFERENCE_CAPITAL,
    )

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert len(seen["replays"]) == 3
    unit_replay, main, stress = seen["replays"]
    assert stress["fee"] == 18.0
    assert stress["policy"] == main["policy"]
    unit_policy = unit_replay["policy"]
    assert unit_policy.kind == "fixed"
    assert unit_policy.exposure_max == 1.0
    assert unit_policy.impact_y == 0.0
    assert unit_replay["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert unit_replay["filters"] is False
    assert unit_replay["unit_equity"] is None
    policy = main["policy"]
    assert policy.kind == "growth"
    assert main["capital"] == ACCOUNT_DEFAULT_CAPITAL_USDT
    assert policy.mean_haircut == ACCOUNT_MEAN_HAIRCUT
    assert (policy.exposure_max, policy.exposure_step) == (ACCOUNT_EXPOSURE_CAP, ACCOUNT_EXPOSURE_STEP)
    assert (policy.prior_days, policy.min_moment_days) == (ACCOUNT_PRIOR_DAYS, ACCOUNT_MIN_MOMENT_DAYS)
    assert not hasattr(policy, "unit_daily_mean")
    assert (policy.shock_per_unit, policy.margin_reserve, policy.initial_margin_cap) == (
        ACCOUNT_SHOCK_PER_UNIT, ACCOUNT_MARGIN_RESERVE, ACCOUNT_INITIAL_MARGIN_CAP,
    )
    assert policy.impact_y == ACCOUNT_IMPACT_Y
    assert main["fee"] == ACCOUNT_TAKER_FEE_BPS
    assert main["filters"] is True
    assert main["unit_equity"] is not None
    pd.testing.assert_series_equal(main["unit_equity"], seen["equities"][0])

def test_account_fixed_policy_requires_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--policy fixed without --fixed-exposure exits before any load."""
    seen = _install_account(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=r"fixed-exposure"):
        backtest_mod.run_account_replay_command(_parse(_account_argv("--policy", "fixed")))
    assert "request" not in seen
    backtest_mod.run_account_replay_command(_parse(_account_argv("--policy", "fixed", "--fixed-exposure", "2.5")))
    assert seen["replays"][1]["policy"].kind == "fixed"
    assert seen["replays"][1]["policy"].exposure_max == 2.5

def test_account_missing_venue_snapshot_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty venue-rules dir exits naming the collection command."""
    seen = _install_account(monkeypatch, tmp_path, stub_venue=False)
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    monkeypatch.setattr(backtest_mod, "VENUE_RULES_DIR", empty)
    with pytest.raises(SystemExit, match=r"data collect venue-rules"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert "request" not in seen

def test_account_artifacts_and_disclosures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """account.json carries mandated disclosures and the daily parquet exists."""
    from src.core.params import (
        ACCOUNT_MIN_MOMENT_DAYS,
        ACCOUNT_PRIOR_DAYS,
        ACCOUNT_RECON_CAGR_TOLERANCE,
        ACCOUNT_RECON_MDD_TOLERANCE,
        ACCOUNT_UNIT_REFERENCE_CAPITAL,
    )

    seen = _install_account(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps({"kind": "mhs", "strategy_id": "x", "base_cagr": 0.1}) + "\n",
        encoding="utf-8",
    )
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(
        tmp_path, index, run_dir="runs/old", base_cagr=unit_cagr, base_mdd=0.05,
    )
    with caplog.at_level("INFO", logger="MhsBacktestCli"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert "[EVAL] account-replay" in caplog.text
    (run_dir,) = _run_dirs(tmp_path)
    assert run_dir.name.startswith("20250101_20250201_top20_account_growth_2100_")
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert "in_sample_moments" not in payload
    assert payload["moment_source"] == "bayesian_causal_unit_ledger"
    assert payload["venue_rules_applied_retroactively"] is True
    assert payload["entry_anchor"] == "submit_bar"
    assert payload["capital"] == 2100.0
    assert payload["policy"]["kind"] == "growth"
    assert payload["policy"]["prior_days"] == ACCOUNT_PRIOR_DAYS
    assert payload["policy"]["min_moment_days"] == ACCOUNT_MIN_MOMENT_DAYS
    assert "unit_daily_mean" not in payload["policy"]
    assert payload["unit_reference"]["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert payload["unit_reference"]["cagr"] == pytest.approx(unit_cagr)
    assert payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert payload["unit_reference"]["daily_mdd"] == pytest.approx(105000.0 / 110000.0 - 1.0)
    assert payload["cagr"] == pytest.approx((2205.0 / 2100.0) ** (365.0 / 3.0) - 1.0)
    assert payload["mdd"] == pytest.approx(-0.05)
    assert payload["daily_mdd"] == pytest.approx(2205.0 / 2310.0 - 1.0)
    assert payload["liquidated_at"] is None
    assert (payload["mean_exposure"], payload["min_exposure"], payload["last_exposure"]) == (1.5, 1.0, 1.5)
    recon = payload["reconciliation"]
    assert recon["status"] == "ok"
    assert recon["reference_canonical"]["base_cagr"] == pytest.approx(unit_cagr)
    assert recon["reference_canonical"]["name_clip"] == pytest.approx(0.05)
    assert recon["reference_canonical"]["exposure_multiplier"] == pytest.approx(1.0)
    assert recon["cagr_gap"] == pytest.approx(0.0)
    assert recon["mdd"] == pytest.approx(0.05)
    assert recon["mdd_convention"] == "magnitude"
    assert recon["mdd_definition"] == "3m_close_path"
    assert recon["mdd_gap"] == pytest.approx(0.0)
    assert recon["cagr_tolerance"] == ACCOUNT_RECON_CAGR_TOLERANCE
    assert recon["mdd_tolerance"] == ACCOUNT_RECON_MDD_TOLERANCE
    daily = pd.read_parquet(run_dir / "account_daily.parquet")
    assert list(daily.columns) == ["equity", "exposure"]
    assert len(daily) == 3
    rows = index.read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["kind"] == "mhs_frozen_account"
    assert last["base_max_drawdown"] == pytest.approx(0.05)

    index.unlink()
    backtest_mod.run_account_replay_command(_parse(_account_argv()))
    _, second = _run_dirs(tmp_path)
    again = json.loads((second / "account.json").read_text(encoding="utf-8"))
    assert again["reconciliation"]["status"] == "missing_reference"
    assert again["reconciliation"]["reference_canonical"] is None
    assert again["reconciliation"]["cagr_gap"] is None
    assert again["reconciliation"]["mdd_gap"] is None

def test_account_unit_reference_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed unit reference replay exits without persisting an account artifact."""
    _install_account(monkeypatch, tmp_path, unit_fail=True)
    with pytest.raises(SystemExit, match=r"unit reference"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []

def test_account_unit_reference_liquidation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A liquidated unit reference is a failed input and exits."""
    _install_account(monkeypatch, tmp_path, unit_liquidated=True)
    with pytest.raises(SystemExit, match=r"unit reference"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []

def test_account_growth_policy_constructed_via_shared_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The growth replay policy equals the shared account growth factory output."""
    from src.strategy.sizing import account_growth_policy

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_account_replay_command(_parse(_account_argv("--impact-y", "0.7")))
    assert seen["replays"][1]["policy"] == account_growth_policy(impact_y=0.7)

def test_account_invalid_execution_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown execution mode exits before loading any input."""
    import argparse

    seen = _install_account(monkeypatch, tmp_path)
    args = argparse.Namespace(
        source_start="2024-01-01", start="2025-01-01", end="2025-02-01",
        policy="growth", fixed_exposure=None, execution="limit",
    )
    with pytest.raises(SystemExit, match=r"execution must be"):
        backtest_mod.run_account_replay_command(args)
    assert "request" not in seen

def test_assemble_account_inputs_held_symbols_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only held symbols are read; funding is cumulated on 3m bars and sampled at each entry."""
    import src.engine.account_sources as sources_mod

    candidate, context, entries, grid = _assemble_fixture(tmp_path)
    seen: dict = {}
    real_assert = sources_mod.assert_mhs_stage_allocation

    def _spy(*, stage: str, **kwargs: object) -> None:
        seen["stage"] = stage
        real_assert(stage=stage, **kwargs)

    monkeypatch.setattr(sources_mod, "assert_mhs_stage_allocation", _spy)
    unit, marks, funding_cum, adv, daily_sigma, anchors = sources_mod.assemble_account_inputs(candidate, context)

    assert list(unit.columns) == ["AAA"]
    assert seen["stage"] == "account_marks"
    assert marks.close.dtypes.iloc[0] == np.dtype("float32")
    assert marks.close.index[0] == entries[0] - pd.Timedelta(hours=1)
    assert marks.close.index[-1] == entries[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    # 펀딩 누적은 진입 시각에서 표본화되어 replay_account의 일간 인덱스와 정렬돼야 한다.
    assert funding_cum.index.equals(unit.index)
    assert list(funding_cum.columns) == list(unit.columns)
    assert funding_cum.loc[entries[0], "AAA"] == pytest.approx(0.0)
    assert funding_cum.loc[entries[1], "AAA"] == pytest.approx(0.0001)
    assert adv.index.equals(unit.index)
    assert daily_sigma.index.equals(unit.index)
    assert adv.loc[entries[1], "AAA"] == pytest.approx(5_000_000.0)
    assert bool(np.isfinite(daily_sigma.loc[entries[1], "AAA"]))

def test_assemble_account_inputs_missing_held_source_fails(tmp_path: Path) -> None:
    """A held symbol without 3m source fails closed."""
    import src.engine.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, _, _ = _assemble_fixture(tmp_path)
    (tmp_path / "3m" / "AAA.parquet").unlink()
    with pytest.raises(DataIntegrityError, match=r"AAA"):
        sources_mod.assemble_account_inputs(candidate, context)

def test_account_rejects_invalid_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid dates, policy, and controls exit before any load."""
    import argparse

    seen = _install_account(monkeypatch, tmp_path)
    run = backtest_mod.run_account_replay_command
    with pytest.raises(SystemExit, match=r"source-start is required"):
        run(_parse(["backtest", "account", "--start", "2025-01-01", "--end", "2025-02-01"]))
    with pytest.raises(SystemExit, match=r"start is required"):
        run(_parse(["backtest", "account", "--source-start", "2024-01-01", "--end", "2025-02-01"]))
    with pytest.raises(SystemExit, match=r"end is required"):
        run(_parse(["backtest", "account", "--source-start", "2024-01-01", "--start", "2025-01-01"]))
    with pytest.raises(SystemExit, match=r"source-start < start < end"):
        run(_parse(["backtest", "account", "--source-start", "2025-03-01", "--start", "2025-01-01", "--end", "2025-02-01"]))
    base = {"source_start": "2024-01-01", "start": "2025-01-01", "end": "2025-02-01"}
    with pytest.raises(SystemExit, match=r"policy must be"):
        run(argparse.Namespace(**base, policy="turbo", fixed_exposure=None))
    with pytest.raises(SystemExit, match=r"invalid account controls"):
        run(argparse.Namespace(**base, policy="fixed", fixed_exposure="abc"))
    with pytest.raises(SystemExit, match=r"positive finite exposure"):
        run(_parse(_account_argv("--policy", "fixed", "--fixed-exposure", "0")))
    with pytest.raises(SystemExit, match=r"positive finite capital"):
        run(_parse(_account_argv("--capital", "0")))
    assert "request" not in seen

def test_assemble_account_inputs_malformed_source_fails(tmp_path: Path) -> None:
    """A 3m archive without OHLC columns fails closed."""
    import src.engine.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, _, _ = _assemble_fixture(tmp_path)
    grid = pd.date_range("2021-04-01", periods=8, freq="3min", tz="UTC")
    pd.DataFrame({"timestamp": np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64"), "open": 1.0}).to_parquet(
        tmp_path / "3m" / "AAA.parquet"
    )
    with pytest.raises(DataIntegrityError, match=r"malformed"):
        sources_mod.assemble_account_inputs(candidate, context)

def test_account_source_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data-integrity failure during source assembly exits instead of replaying garbage."""
    import src.engine.strategy_backtest as run_mod

    from src.common.errors import DataIntegrityError

    seen = _install_account(monkeypatch, tmp_path)

    def _boom(request: object) -> tuple:
        raise DataIntegrityError("source boom")

    monkeypatch.setattr(run_mod, "build_request_targets", _boom)
    with pytest.raises(SystemExit, match=r"account replay failed"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert "replays" not in seen

def test_account_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay failure exits without persisting an account artifact."""
    import src.engine.account_ledger as ledger_mod

    from src.common.errors import DataIntegrityError

    _install_account(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("replay boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _boom)
    with pytest.raises(SystemExit, match=r"account replay failed"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []

def test_assemble_account_inputs_source_outside_window_fails(tmp_path: Path) -> None:
    """A 3m archive with no bars inside the replay grid fails closed."""
    import numpy as np

    import src.engine.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, entries, _ = _assemble_fixture(tmp_path)
    stale = pd.date_range("2020-01-01", periods=10, freq="3min", tz="UTC")
    pd.DataFrame({
        "timestamp": np.array([int(ts.value // 1_000_000) for ts in stale], dtype="int64"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }).to_parquet(tmp_path / "3m" / "AAA.parquet")
    with pytest.raises(DataIntegrityError, match=r"AAA"):
        sources_mod.assemble_account_inputs(candidate, context)

def test_backtest_strategy_account_unit_variant_only_at_breadth_20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """account_unit selects the registered clip unit book at Top-20 and names its run directory."""
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    assert backtest_mod._strategy_policy(20, "account_unit") is FLOW_MOM_TOP20_ACCOUNT_UNIT
    with pytest.raises(SystemExit):
        backtest_mod._strategy_policy(40, "account_unit")
    name = backtest_mod._strategy_run_name(
        pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"),
        20, pd.Timestamp("2025-03-01T00:00:00Z"), variant="account_unit",
    )
    assert "_account_unit" in name
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(_parse([
        "backtest", "strategy",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--variant", "account_unit", "--output", str(tmp_path / "unit.json"),
    ]))
    assert seen["request"].strategy is FLOW_MOM_TOP20_ACCOUNT_UNIT

def test_account_service_failure_maps_to_verbatim_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A AccountReplayError from the service surfaces as a verbatim non-zero exit."""
    import src.application.strategy_account as app_mod
    from src.application.strategy_account import AccountReplayError

    _install_account(monkeypatch, tmp_path)

    def _boom(request: object) -> object:
        raise AccountReplayError("account replay failed: x")

    monkeypatch.setattr(app_mod, "run_account_replay", _boom)
    with pytest.raises(SystemExit) as exc_info:
        backtest_mod.run_account_replay_command(_parse(_account_argv()))
    assert exc_info.value.code == "account replay failed: x"
    assert isinstance(exc_info.value.__cause__, AccountReplayError)

def test_account_request_validation_maps_to_invalid_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ValueError from the service maps to an invalid-request exit."""
    import src.application.strategy_account as app_mod

    _install_account(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "run_account_replay", lambda request: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(SystemExit, match=r"invalid account replay request: bad"):
        backtest_mod.run_account_replay_command(_parse(_account_argv()))

def test_account_cli_passes_path_roots_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched run/venue roots and flags reach the service request verbatim."""
    import src.application.strategy_account as app_mod

    _install_account(monkeypatch, tmp_path)
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", tmp_path / "custom" / "runs")
    monkeypatch.setattr(backtest_mod, "VENUE_RULES_DIR", tmp_path / "custom" / "venue")
    captured: dict = {}
    real = app_mod.run_account_replay

    def _capture(request: object) -> object:
        captured["request"] = request
        return real(request)

    monkeypatch.setattr(app_mod, "run_account_replay", _capture)
    backtest_mod.run_account_replay_command(_parse(_account_argv("--no-order-filters", "--export-unit-returns", str(tmp_path / "u.parquet"))))
    request = captured["request"]
    assert request.runs_root == tmp_path / "custom" / "runs"
    assert request.venue_rules_root == tmp_path / "custom" / "venue"
    assert request.apply_order_filters is False
    assert request.export_unit_returns == tmp_path / "u.parquet"

def test_exposure_cli_maps_service_error_and_prints_only_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """An exposure service error maps to SystemExit with no stdout; success prints once."""
    import src.application.strategy_account as app_mod
    from src.application.strategy_account import AccountReplayError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)
    monkeypatch.setattr(app_mod, "run_exposure_scan", lambda request: (_ for _ in ()).throw(AccountReplayError("exposure scan failed: y")))
    with pytest.raises(SystemExit, match=r"exposure scan failed: y"):
        backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure", "--run-dir", str(run_dir)]))
    assert capsys.readouterr().out == ""

def test_exposure_cli_prints_path_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A successful exposure run prints the finalized path once."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)
    backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure", "--run-dir", str(run_dir)]))
    out = capsys.readouterr().out.strip()
    assert out == str(run_dir / "exposure.json")
