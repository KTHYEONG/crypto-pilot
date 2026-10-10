"""Local evaluation standard CLI: historical run directories in, verdict out."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.evaluate as evaluate_mod
import src.evaluation.standard as standard_mod
from src.strategy.release import load_release, strategy_signal_digest
from src.strategy.targets import FLOW_MOM_TOP20


def _current_signal() -> str:
    return strategy_signal_digest(FLOW_MOM_TOP20)

@pytest.fixture(autouse=True)
def _bootstrap_budget(monkeypatch):
    monkeypatch.setattr(standard_mod, "REPORT_BOOTSTRAP_PATHS", 200)


def _write_unit_run(
    run_dir: Path, n: int = 100, *, turnover: bool = True, funding: bool = True,
    start: str = "2025-01-01",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "base_return": np.linspace(0.001, 0.002, n),
            "stress_return": np.linspace(0.0008, 0.0018, n),
            **({"base_turnover": np.full(n, 0.004)} if turnover else {}),
        },
        index=index,
    )
    frame.to_parquet(run_dir / "daily.parquet")
    evaluation = {
        "base_cagr": 0.3,
        **(
            {"funding_by_symbol": {"stress": {"AAAUSDT": 0.01, "BBBUSDT": 0.005}}}
            if funding
            else {}
        ),
    }
    payload = {
        "strategy_id": "flow_mom_top20",
        "breadth": 20,
        "evaluation_start": index[0].isoformat(),
        "evaluation_end": (index[-1] + pd.Timedelta(days=1)).isoformat(),
        "base_valid": True,
        "stress_valid": True,
        "source_gap_excluded_count": 0,
        "roster_seat_days": 40000,
        "participation_scale": 100000.0,
        "participation_basis": "adv30_median_prior_day",
        "data_availability_withdrawals": [],
        "limitations": [],
        "signal_digest": _current_signal(),
        "report_periods": {"evaluation": evaluation},
        **(
            {"statistics": {"stress": {"funding_by_symbol": {"CCUSDT": 0.02}}}}
            if not funding
            else {}
        ),
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_account_run(run_dir: Path, n: int = 100, start: str = "2025-01-01") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    equity = 1000.0 * np.cumprod(1.0 + 0.001 + 0.002 * rng.standard_normal(n))
    stress_equity = 1000.0 * np.cumprod(1.0 + 0.0008 + 0.002 * rng.standard_normal(n))
    pd.DataFrame(
        {"equity": equity, "exposure": np.full(n, 1.2)},
        index=index,
    ).to_parquet(run_dir / "account_daily.parquet")
    pd.DataFrame(
        {"equity": stress_equity, "exposure": np.full(n, 1.1)},
        index=index,
    ).to_parquet(run_dir / "account_stress_daily.parquet")
    (run_dir / "account.json").write_text(json.dumps({
        "capital": 1000.0, "execution": "maker",
        "cagr": 0.3, "mdd": -0.1, "mean_exposure": 1.2,
        "liquidated_at": None, "initial_margin_breaches": 0,
        "signal_digest": _current_signal(),
        "evaluation_start": pd.Timestamp(start, tz="UTC").isoformat(),
        "evaluation_end": (pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=n)).isoformat(),
        "stress_execution": {"liquidated_at": None},
    }), encoding="utf-8")


def _args(**overrides: object) -> argparse.Namespace:
    params = {
        "strategy": "flow_mom_top20",
        "unit_run": "", "stress_from_unit": True, "account_run": "",
        "neighbors": [], "holdout_run": None, "accept": False,
    }
    params.update(overrides)
    return argparse.Namespace(**params)


def test_build_inputs_from_run_directories(tmp_path) -> None:
    """Run directories assemble pure evaluation inputs for the release."""
    unit, account, neighbor = tmp_path / "unit", tmp_path / "acct", tmp_path / "nb0"
    _write_unit_run(unit)
    daily = pd.read_parquet(unit / "daily.parquet")
    daily["fill_adv_participation"] = 0.004
    daily.to_parquet(unit / "daily.parquet")
    _write_account_run(account)
    _write_unit_run(neighbor, n=100)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        neighbor_runs=(neighbor,), holdout_run=None,
    )
    release = load_release("flow_mom_top20")
    assert inputs.strategy_id == release.strategy_id
    assert inputs.spec_digest == release.spec_digest
    assert len(inputs.base_returns) == 100
    assert len(inputs.neighbors) == 1
    assert inputs.ledger_certified is False
    assert inputs.causality_ok is True
    assert inputs.trial_population.n_trials >= 96
    assert inputs.participation.iloc[0] == 0.004
    assert evaluate_mod._declared_book({}, "unknown") is False


def test_build_inputs_funding_and_turnover_fallbacks(tmp_path) -> None:
    """Missing turnover and funding sections fall back without crashing."""
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit, turnover=False, funding=False)
    _write_account_run(account)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.participation.empty
    assert inputs.funding_by_symbol == {"CCUSDT": 0.02}
    naive = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2, freq="D"))
    localized = evaluate_mod._utc_series(naive, "probe")
    assert str(localized.index.tz) == "UTC"
    assert evaluate_mod._family_of("flow_mom_top20") == "flow_mom"
    assert evaluate_mod._family_of("other") == "other"
    assert evaluate_mod._causality_ok("flow_mom_b15_control") is False
    with pytest.raises(Exception, match=r".+"):
        evaluate_mod._utc_series(pd.Series(dtype="float64"), "probe")


def test_evaluate_command_reports_inconclusive_without_holdout(tmp_path, monkeypatch, capsys) -> None:
    """Short discovery without holdout exits 2 after printing the table."""
    import src.common.paths as paths_mod

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=1900, start="2021-06-01")
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    with pytest.raises(SystemExit) as excinfo:
        evaluate_mod.run_evaluate_strategy_command(
            _args(unit_run=str(unit), account_run=str(account))
        )
    assert excinfo.value.code == 4
    out = capsys.readouterr().out
    assert "verdict=invalid" in out
    assert "E1_DSR" in out
    written = list((tmp_path / "backtests" / "evaluation").glob("flow_mom_top20_*.json"))
    assert len(written) == 1
    envelope = json.loads(written[0].read_text(encoding="utf-8"))
    assert envelope["verdict"] == "invalid"


def test_evaluate_accept_refused_without_accept(tmp_path, monkeypatch) -> None:
    """--accept on a non-ACCEPT verdict refuses instead of writing."""
    import src.common.paths as paths_mod

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=1900, start="2021-06-01")
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    with pytest.raises(SystemExit, match="refused"):
        evaluate_mod.run_evaluate_strategy_command(
            _args(unit_run=str(unit), account_run=str(account), accept=True)
        )


def test_evaluate_accept_writes_release_on_matching_accept(tmp_path, monkeypatch, capsys) -> None:
    """A matching ACCEPT records the digest and exits 0."""
    import src.common.paths as paths_mod

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    release = load_release("flow_mom_top20")
    from src.strategy.release import criteria_digest

    check = types.SimpleNamespace(
        code="E1_DSR", group="edge", passed=True, value=0.99,
        threshold=">= 0.95", reason="ok",
    )
    evaluation = types.SimpleNamespace(
        strategy_id="flow_mom_top20", spec_digest=release.spec_digest,
        criteria_digest=criteria_digest(release.criteria), verdict="accept",
        digest="digest-0", checks=(check,), n_trials=97,
    )
    monkeypatch.setattr(evaluate_mod, "build_evaluation_inputs", lambda **kwargs: object())
    monkeypatch.setattr(standard_mod, "evaluate_strategy", lambda inputs, criteria: evaluation)
    seen: dict = {}
    import src.strategy.release as release_mod

    monkeypatch.setattr(
        release_mod, "record_acceptance",
        lambda strategy_id, **kwargs: seen.update(kwargs) or release,
    )
    with pytest.raises(SystemExit) as excinfo:
        evaluate_mod.run_evaluate_strategy_command(
            _args(unit_run=str(unit), account_run=str(account), accept=True)
        )
    assert excinfo.value.code == 0
    assert seen["evaluation_digest"] == "digest-0"
    assert "verdict=accept" in capsys.readouterr().out


def test_evaluate_command_maps_verdict_to_exit(tmp_path, monkeypatch) -> None:
    """REJECT exits 3 and INVALID exits 4."""
    import src.common.paths as paths_mod
    import src.evaluation.standard as standard_mod

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    release = load_release("flow_mom_top20")
    from src.strategy.release import criteria_digest

    for verdict, code in (("reject", 3), ("invalid", 4)):
        evaluation = types.SimpleNamespace(
            strategy_id="flow_mom_top20", spec_digest=release.spec_digest,
            criteria_digest=criteria_digest(release.criteria), verdict=verdict,
            digest=f"digest-{verdict}", checks=(), n_trials=97,
        )
        monkeypatch.setattr(
            standard_mod,
            "evaluate_strategy",
            lambda inputs, criteria, _evaluation=evaluation: _evaluation,
        )
        with pytest.raises(SystemExit) as excinfo:
            evaluate_mod.run_evaluate_strategy_command(
                _args(unit_run=str(unit), account_run=str(account))
            )
        assert excinfo.value.code == code


def test_evaluate_builder_guards_and_holdout(tmp_path, monkeypatch) -> None:
    """Indexer, funding-shape, and causality guards fail closed; holdout wires through."""
    import pytest

    from src.common.errors import DataIntegrityError

    with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
        evaluate_mod._utc_series(pd.Series([0.01, 0.02], index=[0, 1]), "probe")
    unit, account, holdout = tmp_path / "unit", tmp_path / "acct", tmp_path / "hold"
    _write_unit_run(unit)
    raw = json.loads((unit / "result.json").read_text(encoding="utf-8"))
    raw["report_periods"] = []
    (unit / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    _write_account_run(account)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.funding_by_symbol == {}
    _write_unit_run(holdout, n=70, start="2026-07-01")
    with pytest.raises(DataIntegrityError, match="one-look journal"):
        evaluate_mod.build_evaluation_inputs(strategy_id="flow_mom_top20", unit_run=unit,
                                            account_run=account, holdout_run=holdout)
    import src.strategy.release as release_mod
    from src.evaluation.holdout import consume_holdout_look

    release = load_release("flow_mom_top20")
    look_path = tmp_path / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", release.spec_digest,
                         (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-09", tz="UTC")), path=look_path,
                         signal_digest=_current_signal())
    original = release_mod.releases_dir
    import shutil
    shutil.copy(original() / "flow_mom_top20.json", tmp_path / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root=None: tmp_path)
    held = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        holdout_run=holdout,
    )
    assert held.holdout is not None
    assert held.holdout_returns is not None
    assert len(held.holdout_returns) == 70


def test_signal_digest_mismatch_fails_closed(tmp_path, monkeypatch) -> None:
    """A holdout result with another signal_digest raises."""
    from src.common.errors import DataIntegrityError
    from src.evaluation.holdout import consume_holdout_look

    import src.strategy.release as release_mod

    unit, account, holdout = tmp_path / "unit", tmp_path / "acct", tmp_path / "hold"
    _write_unit_run(unit)
    _write_account_run(account)
    _write_unit_run(holdout, n=70, start="2026-07-01")
    raw = json.loads((holdout / "result.json").read_text(encoding="utf-8"))
    raw["signal_digest"] = "other-signal"
    (holdout / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    release = load_release("flow_mom_top20")
    look_path = tmp_path / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", release.spec_digest,
                         (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-09", tz="UTC")), path=look_path,
                         signal_digest=_current_signal())
    import shutil
    shutil.copy(release_mod.releases_dir() / "flow_mom_top20.json", tmp_path / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root=None: tmp_path)
    with pytest.raises(DataIntegrityError, match="signal digest mismatch"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account, holdout_run=holdout,
        )
    raw = json.loads((holdout / "result.json").read_text(encoding="utf-8"))
    raw["signal_digest"] = _current_signal()
    (holdout / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    unit_raw = json.loads((unit / "result.json").read_text(encoding="utf-8"))
    unit_raw["signal_digest"] = "stale-signal"
    (unit / "result.json").write_text(json.dumps(unit_raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="signal digest mismatch"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account, holdout_run=holdout,
        )


def test_holdout_account_run_is_informational(tmp_path, monkeypatch) -> None:
    """Leveraged holdout slices land in informational only: verdict and digest are unchanged."""
    import src.common.paths as paths_mod

    unit, account, holdout = tmp_path / "unit", tmp_path / "acct", tmp_path / "hold"
    _write_unit_run(unit)
    _write_account_run(account)
    _write_unit_run(holdout, n=70, start="2026-07-01")
    from src.evaluation.holdout import consume_holdout_look

    import src.strategy.release as release_mod

    release = load_release("flow_mom_top20")
    look_path = tmp_path / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", release.spec_digest,
                         (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-09", tz="UTC")), path=look_path,
                         signal_digest=_current_signal())
    import shutil
    shutil.copy(release_mod.releases_dir() / "flow_mom_top20.json", tmp_path / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root=None: tmp_path)
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    holdout_account = tmp_path / "hold_acct"
    holdout_account.mkdir(parents=True)
    unit_index = pd.read_parquet(unit / "daily.parquet").index
    discovery_start = pd.Timestamp(str(json.loads((unit / "result.json").read_text())["evaluation_start"])).tz_convert("UTC")
    holdout_end = pd.Timestamp(str(json.loads((holdout / "result.json").read_text())["evaluation_end"])).tz_convert("UTC")
    full_index = pd.date_range(start=discovery_start, end=holdout_end - pd.Timedelta(days=1), freq="D", tz="UTC")
    n_full = len(full_index)
    equity = 1000.0 * np.cumprod(np.full(n_full, 1.001))
    pd.DataFrame({"equity": equity, "exposure": np.full(n_full, 1.2)}, index=full_index).to_parquet(
        holdout_account / "account_daily.parquet")
    pd.DataFrame({"equity": equity * 0.999, "exposure": np.full(n_full, 1.1)}, index=full_index).to_parquet(
        holdout_account / "account_stress_daily.parquet")
    (holdout_account / "account.json").write_text(json.dumps({
        "capital": 1000.0, "execution": {"mode": "maker"},
        "signal_digest": _current_signal(),
        "evaluation_start": discovery_start.isoformat(), "evaluation_end": holdout_end.isoformat(),
        "cagr": 0.3, "mdd": -0.1, "mean_exposure": 1.2,
        "liquidated_at": None, "initial_margin_breaches": 0,
        "stress_execution": {"liquidated_at": None},
    }), encoding="utf-8")
    assert unit_index is not None
    base_args = {"unit_run": str(unit), "account_run": str(account), "holdout_run": str(holdout)}
    with pytest.raises(SystemExit):
        evaluate_mod.run_evaluate_strategy_command(_args(**base_args))
    plain_outputs = list((tmp_path / "backtests" / "evaluation").glob("*.json"))
    assert len(plain_outputs) == 1
    plain = json.loads(plain_outputs[0].read_text())
    assert "holdout_deployed_return" not in plain["informational"]
    with pytest.raises(SystemExit):
        evaluate_mod.run_evaluate_strategy_command(_args(**base_args, holdout_account_run=str(holdout_account)))
    enriched_outputs = list((tmp_path / "backtests" / "evaluation").glob("*.json"))
    assert len(enriched_outputs) >= 1
    enriched = max(enriched_outputs, key=lambda path: path.stat().st_mtime_ns)
    info = json.loads(enriched.read_text())["informational"]
    assert info["holdout_days"] == 70.0
    assert info["discovery_days"] == 100.0
    assert set(info) >= {"holdout_deployed_return", "holdout_deployed_mdd", "holdout_deployed_stress_return",
                         "holdout_mean_exposure", "discovery_deployed_return", "discovery_deployed_mdd",
                         "discovery_deployed_stress_return", "discovery_mean_exposure"}
    assert json.loads(enriched.read_text())["digest"] == plain["digest"]
    assert json.loads(enriched.read_text())["verdict"] == plain["verdict"]


def _write_holdout_account(
    run_dir: Path, index: pd.DatetimeIndex, *, capital: float = 1000.0,
    signal: str | None = None, start: str | None = None, end: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(index)
    equity = capital * np.cumprod(np.full(n, 1.001))
    pd.DataFrame({"equity": equity, "exposure": np.full(n, 1.2)}, index=index).to_parquet(
        run_dir / "account_daily.parquet")
    pd.DataFrame({"equity": equity * 0.999, "exposure": np.full(n, 1.1)}, index=index).to_parquet(
        run_dir / "account_stress_daily.parquet")
    (run_dir / "account.json").write_text(json.dumps({
        "capital": capital, "execution": {"mode": "maker"},
        "signal_digest": _current_signal() if signal is None else signal,
        "evaluation_start": index[0].isoformat() if start is None else start,
        "evaluation_end": (index[-1] + pd.Timedelta(days=1)).isoformat() if end is None else end,
        "cagr": 0.3, "mdd": -0.1, "mean_exposure": 1.2,
        "liquidated_at": None, "initial_margin_breaches": 0,
        "stress_execution": {"liquidated_at": None},
    }), encoding="utf-8")


@pytest.mark.parametrize("override", [
    {"capital": 2100.0}, {"signal": "other"}, {"start": "2025-02-01T00:00:00+00:00"},
    {"end": "2026-01-01T00:00:00+00:00"},
])
def test_holdout_account_identity_mismatch_fails_closed(tmp_path, override) -> None:
    """A holdout account run with wrong capital, signal, or window raises."""
    from src.common.errors import DataIntegrityError

    discovery = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-04-11", tz="UTC"))
    holdout = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-09", tz="UTC"))
    full = pd.date_range(start=discovery[0], end=holdout[1] - pd.Timedelta(days=1), freq="D", tz="UTC")
    run_dir = tmp_path / "hold_acct"
    _write_holdout_account(run_dir, full, **override)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match=r"mismatch|must start|must end"):
        evaluate_mod._holdout_account_informational(
            run_dir, discovery, holdout, 1000.0, _current_signal())


def test_holdout_account_without_holdout_run_fails_closed(tmp_path, monkeypatch) -> None:
    """--holdout-account-run without --holdout-run raises instead of reporting."""
    import src.common.paths as paths_mod

    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    holdout_account = tmp_path / "hold_acct"
    _write_holdout_account(holdout_account, pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"))
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    with pytest.raises(DataIntegrityError, match="requires a holdout run"):
        evaluate_mod.run_evaluate_strategy_command(
            _args(unit_run=str(unit), account_run=str(account),
                  holdout_account_run=str(holdout_account))
        )


def test_slice_deployed_metrics_guards(tmp_path) -> None:
    """Degenerate account slices fail closed."""
    from src.common.errors import DataIntegrityError

    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame({"equity": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0],
                          "exposure": np.full(5, 1.2)}, index=index)
    stress = pd.DataFrame({"equity": [1000.0, 1009.0, 1018.0, 1027.0, 1036.0],
                           "exposure": np.full(5, 1.1)}, index=index)
    with pytest.raises(DataIntegrityError, match="daily index"):
        evaluate_mod._slice_deployed_metrics(
            pd.DataFrame({"equity": [1.0]}), stress, 1000.0, index[0], index[1])
    with pytest.raises(DataIntegrityError, match="no days"):
        evaluate_mod._slice_deployed_metrics(
            frame, stress, 1000.0, pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"))
    bad = frame.copy()
    bad.loc[index[0], "equity"] = float("nan")
    with pytest.raises(DataIntegrityError, match="finite"):
        evaluate_mod._slice_deployed_metrics(
            bad, stress, 1000.0, index[0], index[2])
    zeroed = frame.copy()
    zeroed.loc[:, "equity"] = 0.0
    with pytest.raises(DataIntegrityError, match="positive"):
        evaluate_mod._slice_deployed_metrics(
            zeroed, stress, 1000.0, index[1], index[3])
    metrics = evaluate_mod._slice_deployed_metrics(frame, stress, 1000.0, index[0], index[2])
    assert metrics["days"] == 2.0
    assert metrics["deployed_return"] == pytest.approx(0.01)


@pytest.mark.parametrize("defect", ["gap", "stress_shift", "duplicate", "reversed", "negative_exposure", "negative_equity", "resurrection"])
def test_holdout_account_evidence_fails_closed(tmp_path, defect) -> None:
    from src.common.errors import DataIntegrityError

    index = pd.date_range("2026-06-29", periods=4, freq="D", tz="UTC")
    account = tmp_path / "continuous"
    _write_holdout_account(account, index)
    frame = pd.read_parquet(account / "account_daily.parquet")
    stress = pd.read_parquet(account / "account_stress_daily.parquet")
    if defect == "gap":
        frame = frame.drop(index[1])
    elif defect == "stress_shift":
        stress.index = stress.index + pd.Timedelta(days=1)
    elif defect == "duplicate":
        frame.index = pd.DatetimeIndex([index[0], index[1], index[1], index[3]])
    elif defect == "reversed":
        frame = frame.iloc[::-1]
    elif defect == "negative_exposure":
        frame.loc[index[2], "exposure"] = -1.0
    elif defect == "negative_equity":
        stress.loc[index[2], "equity"] = -1.0
    else:
        frame.loc[index[1], "equity"] = 0.0
    frame.to_parquet(account / "account_daily.parquet")
    stress.to_parquet(account / "account_stress_daily.parquet")
    with pytest.raises(DataIntegrityError):
        evaluate_mod._holdout_account_informational(
            account, (index[0], index[2]), (index[2], index[-1] + pd.Timedelta(days=1)),
            1000.0, _current_signal(),
        )


def test_account_slice_stress_index_cannot_be_reinterpreted() -> None:
    from src.common.errors import DataIntegrityError

    index = pd.date_range("2026-07-01", periods=3, tz="UTC")
    base = pd.DataFrame({"equity": [1000.0, 900.0, 990.0], "exposure": [1.0, 1.0, 1.0]}, index=index)
    stress = base.copy()
    stress.index = index + pd.Timedelta(days=1)
    with pytest.raises(DataIntegrityError, match="indexes differ"):
        evaluate_mod._slice_deployed_metrics(base, stress, 1000.0, index[0], index[-1])


def test_account_holdout_slice_uses_prior_close_and_local_peak() -> None:
    index = pd.date_range("2026-06-29", periods=4, tz="UTC")
    base = pd.DataFrame({"equity": [1500.0, 1200.0, 900.0, 990.0], "exposure": [1.0, 2.0, 3.0, 4.0]}, index=index)
    stress = pd.DataFrame({"equity": [1400.0, 1000.0, 800.0, 880.0]}, index=index)
    metrics = evaluate_mod._slice_deployed_metrics(base, stress, 1000.0, index[2], index[-1] + pd.Timedelta(days=1))
    assert metrics == pytest.approx({
        "deployed_return": -0.175, "deployed_mdd": -0.25,
        "deployed_stress_return": -0.12, "mean_exposure": 3.5, "days": 2.0,
    })


def test_evaluate_causality_fails_on_unknown_feature(monkeypatch) -> None:
    """A member outside the registered feature set breaks causality."""
    monkeypatch.setattr(evaluate_mod, "FEATURE_REGISTRY", ())
    assert evaluate_mod._causality_ok("flow_mom_top20") is False


def test_evaluate_accept_refused_on_digest_mismatch(tmp_path, monkeypatch) -> None:
    """--accept with a stale spec digest refuses instead of recording."""
    import types

    import src.common.paths as paths_mod
    import src.evaluation.standard as standard_mod

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=1900, start="2021-06-01")
    monkeypatch.setattr(paths_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    release = load_release("flow_mom_top20")
    from src.strategy.release import criteria_digest

    evaluation = types.SimpleNamespace(
        strategy_id="flow_mom_top20", spec_digest="stale",
        criteria_digest=criteria_digest(release.criteria), verdict="accept",
        digest="digest-stale", checks=(), n_trials=97,
    )
    monkeypatch.setattr(
        standard_mod, "evaluate_strategy", lambda inputs, criteria, _ev=evaluation: _ev
    )
    with pytest.raises(SystemExit, match="digest mismatch"):
        evaluate_mod.run_evaluate_strategy_command(
            _args(unit_run=str(unit), account_run=str(account), accept=True)
        )


def _rewrite_unit_payload(unit: Path, **overrides: object) -> None:
    raw = json.loads((unit / "result.json").read_text(encoding="utf-8"))
    raw.update(overrides)
    (unit / "result.json").write_text(json.dumps(raw), encoding="utf-8")


def test_neighbor_run_predating_seat_accounting_fails_closed(tmp_path) -> None:
    from src.common.errors import DataIntegrityError

    unit, account, neighbor = tmp_path / "unit", tmp_path / "account", tmp_path / "neighbor"
    _write_unit_run(unit)
    _write_unit_run(neighbor)
    _write_account_run(account)
    payload = json.loads((neighbor / "result.json").read_text())
    del payload["roster_seat_days"]
    (neighbor / "result.json").write_text(json.dumps(payload))
    with pytest.raises(DataIntegrityError, match="run predates seat accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account, neighbor_runs=(neighbor,),
        )


def test_recorded_participation_requires_recorded_scale(tmp_path) -> None:
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit)
    _write_account_run(account)
    daily = pd.read_parquet(unit / "daily.parquet")
    daily["fill_adv_participation"] = 0.0002
    daily.to_parquet(unit / "daily.parquet")
    _rewrite_unit_payload(unit, participation_scale=None)
    with pytest.raises(DataIntegrityError, match="participation scale not recorded"):
        evaluate_mod.build_evaluation_inputs(strategy_id="flow_mom_top20", unit_run=unit, account_run=account)


def test_r4_reads_adv_column(tmp_path) -> None:
    """R4 scales the ADV column p95, ignoring the single-bar column."""
    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=800, start="2023-01-01")
    daily = pd.read_parquet(unit / "daily.parquet")
    daily["fill_adv_participation"] = 0.0002
    daily["fill_participation"] = 0.2
    daily.to_parquet(unit / "daily.parquet")
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert float(inputs.participation.max()) == pytest.approx(0.0002)
    import dataclasses

    inputs = dataclasses.replace(inputs, ledger_certified=True, book_identity_ok=True)
    evaluation = standard_mod.evaluate_strategy(inputs, load_release("flow_mom_top20").criteria)
    r4 = next(check for check in evaluation.checks if check.code == "R4_CAPACITY")
    assert inputs.participation_scale_to_deployed == pytest.approx(1000.0 * 1.2 / 100000.0)
    assert r4.value == pytest.approx(0.0002 * 1000.0 * 1.2 / 100000.0)
    assert r4.passed is True


def test_r4_scales_by_realized_leverage(tmp_path) -> None:
    """An account run peaking at 4.25x scales R4 by realized leverage."""
    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=800, start="2023-01-01")
    daily = pd.read_parquet(unit / "daily.parquet")
    daily["fill_adv_participation"] = 0.0002
    daily.to_parquet(unit / "daily.parquet")
    index = pd.date_range("2023-01-01", periods=800, freq="D", tz="UTC")
    frame = pd.read_parquet(account / "account_daily.parquet")
    frame["exposure"] = np.linspace(1.0, 4.25, 800)
    frame.index = index
    frame.to_parquet(account / "account_daily.parquet")
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.participation_scale_to_deployed == pytest.approx(1000.0 * 4.25 / 100000.0)


def test_missing_exposure_series_fails_closed(tmp_path) -> None:
    """An account frame without exposure raises instead of defaulting."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit)
    _write_account_run(account)
    for name in ("account_daily.parquet", "account_stress_daily.parquet"):
        frame = pd.read_parquet(account / name)
        frame = frame.drop(columns=["exposure"])
        frame.to_parquet(account / name)
    with pytest.raises(DataIntegrityError, match="account run lacks exposure series"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )
    with pytest.raises(DataIntegrityError, match="account run lacks exposure series"):
        evaluate_mod._account_informational(account)


@pytest.mark.parametrize("exposure", [[1.0, float("nan")], [1.0, float("inf")],
                                      [1.0, -0.25], [1.0, "invalid"]])
def test_invalid_exposure_series_fails_closed(tmp_path, exposure) -> None:
    from src.common.errors import DataIntegrityError

    account = tmp_path / "account"
    _write_account_run(account, n=2)
    frame = pd.read_parquet(account / "account_daily.parquet")
    frame["exposure"] = exposure
    if isinstance(exposure[-1], str):
        frame["exposure"] = frame["exposure"].astype(str)
    frame.to_parquet(account / "account_daily.parquet")
    for reader in (evaluate_mod._read_account_run, evaluate_mod._account_informational):
        with pytest.raises(DataIntegrityError, match="account exposure must be finite and non-negative"):
            reader(account)


def test_empty_exposure_series_fails_closed() -> None:
    from src.common.errors import DataIntegrityError

    with pytest.raises(DataIntegrityError, match="account exposure must be finite and non-negative"):
        evaluate_mod._account_max_leverage(pd.DataFrame({"exposure": []}))


def test_realized_leverage_allows_zero_exposure_days() -> None:
    assert evaluate_mod._account_max_leverage(pd.DataFrame({"exposure": [0.0, 4.25, 0.0]})) == 4.25


def test_old_run_without_adv_column_fails_closed(tmp_path) -> None:
    """A unit run without the ADV column records no participation."""
    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit, n=800, start="2023-01-01")
    _write_account_run(account, n=800, start="2023-01-01")
    daily = pd.read_parquet(unit / "daily.parquet")
    daily["fill_participation"] = 0.2
    daily.to_parquet(unit / "daily.parquet")
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.participation.empty
    import dataclasses

    inputs = dataclasses.replace(inputs, ledger_certified=True, book_identity_ok=True)
    evaluation = standard_mod.evaluate_strategy(inputs, load_release("flow_mom_top20").criteria)
    r4 = next(check for check in evaluation.checks if check.code == "R4_CAPACITY")
    assert r4.passed is False
    assert r4.reason == "participation not recorded"


@pytest.mark.parametrize("basis", [None, "trailing_24h"])
def test_basis_mismatch_rejected(tmp_path, basis) -> None:
    """A unit result without the recorded basis cannot be evaluated."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit)
    _write_account_run(account)
    raw = json.loads((unit / "result.json").read_text(encoding="utf-8"))
    del raw["participation_basis"]
    if basis is not None:
        raw["participation_basis"] = basis
    (unit / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="participation basis mismatch"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


def test_neighbor_exclusion_and_withdrawal_are_included(tmp_path) -> None:
    unit, account, neighbor = tmp_path / "unit", tmp_path / "account", tmp_path / "neighbor"
    _write_unit_run(unit)
    _write_unit_run(neighbor)
    _write_account_run(account)
    _rewrite_unit_payload(neighbor, source_gap_excluded_count=1,
                          data_availability_withdrawals=[{"days": 10}])
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account, neighbor_runs=(neighbor,),
    )
    assert inputs.lake_coverage_ok is False
    assert inputs.withdrawn_seat_fraction == pytest.approx(0.00025)


def test_i2_identical_withdrawals_across_neighbors_are_not_double_counted(tmp_path) -> None:
    """The same 10 withdrawn seat-days repeated in every neighbor run stay at the single-run fraction."""
    unit, account = tmp_path / "unit", tmp_path / "account"
    _write_unit_run(unit)
    _write_account_run(account)
    withdrawals = [{"symbol": "MANAUSDT", "extent": "INTERIOR", "days": 10}]
    _rewrite_unit_payload(unit, data_availability_withdrawals=withdrawals)
    neighbors = []
    for position in range(7):
        neighbor = tmp_path / f"neighbor{position}"
        _write_unit_run(neighbor)
        _rewrite_unit_payload(neighbor, data_availability_withdrawals=withdrawals)
        neighbors.append(neighbor)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account, neighbor_runs=tuple(neighbors),
    )
    assert inputs.lake_coverage_ok is True
    assert inputs.withdrawn_seat_fraction == pytest.approx(10 / 40000)


def test_i2_disclosed_tiny_withdrawal_passes(tmp_path) -> None:
    """Ten withdrawn seat-days of 40 000 pass I2 with value 0.00025."""
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _rewrite_unit_payload(
        unit,
        data_availability_withdrawals=[
            {"symbol": "MANAUSDT", "extent": "INTERIOR", "days": 5},
            {"symbol": "NEARUSDT", "extent": "INTERIOR", "days": 5},
        ],
        limitations=["DATA_AVAILABILITY_SELECTION"],
    )
    _write_account_run(account)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.lake_coverage_ok is True
    assert inputs.withdrawn_seat_fraction == pytest.approx(10 / 40000)


def test_i2_material_withdrawal_fails(tmp_path) -> None:
    """Withdrawals of 0.2 % of seats fail I2 while staying disclosed."""
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _rewrite_unit_payload(
        unit,
        data_availability_withdrawals=[{"symbol": "XRPUSDT", "extent": "INTERIOR", "days": 80}],
        limitations=["DATA_AVAILABILITY_SELECTION"],
    )
    _write_account_run(account)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.lake_coverage_ok is False
    assert inputs.withdrawn_seat_fraction == pytest.approx(0.002)


def test_i2_excluded_symbols_still_fail(tmp_path) -> None:
    """Any source-gap exclusion fails I2 regardless of the withdrawal fraction."""
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _rewrite_unit_payload(unit, source_gap_excluded_count=1)
    _write_account_run(account)
    inputs = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
    )
    assert inputs.lake_coverage_ok is False


def test_i2_old_run_without_seat_accounting_fails_closed(tmp_path) -> None:
    """A result lacking roster_seat_days raises instead of guessing coverage."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    raw = json.loads((unit / "result.json").read_text(encoding="utf-8"))
    del raw["roster_seat_days"]
    (unit / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    _write_account_run(account)
    with pytest.raises(DataIntegrityError, match="seat accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


def test_i2_degenerate_seat_and_scale_accounting_fails_closed(tmp_path) -> None:
    """Zero seats or a non-positive participation scale raise instead of dividing."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    _rewrite_unit_payload(unit, roster_seat_days=0)
    with pytest.raises(DataIntegrityError, match="seat accounting invalid"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )
    _rewrite_unit_payload(unit, roster_seat_days=40000, participation_scale=0.0)
    with pytest.raises(DataIntegrityError, match="participation scale"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


@pytest.mark.parametrize("withdrawals", [{}, [{"days": -1}], [{"days": 0.5}], [{"days": True}]])
def test_i2_malformed_withdrawal_fails_closed(tmp_path, withdrawals) -> None:
    """Malformed withdrawal accounting cannot certify coverage."""
    from src.common.errors import DataIntegrityError
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _rewrite_unit_payload(unit, data_availability_withdrawals=withdrawals)
    _write_account_run(account)
    with pytest.raises(DataIntegrityError, match="withdrawal accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


def test_old_account_run_predates_survival_accounting(tmp_path) -> None:
    """An account run without stress daily or survival keys fails closed."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    (account / "account_stress_daily.parquet").unlink()
    with pytest.raises(DataIntegrityError, match="survival accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


@pytest.mark.parametrize("payload", [
    {"liquidated_at": 0},
    {"stress_execution": {}},
    {"initial_margin_breaches": True},
    {"initial_margin_breaches": -1},
])
def test_malformed_survival_flags_fail_closed(tmp_path, payload) -> None:
    """Coerced liquidation or breach types never read as survival evidence."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    raw = json.loads((account / "account.json").read_text(encoding="utf-8"))
    if "stress_execution" in payload:
        raw["stress_execution"] = payload["stress_execution"]
    else:
        raw.update(payload)
    (account / "account.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="survival accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


def test_stress_liquidation_flag_type_fails_closed(tmp_path) -> None:
    """A non-string stress liquidation stamp fails closed."""
    from src.common.errors import DataIntegrityError

    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    raw = json.loads((account / "account.json").read_text(encoding="utf-8"))
    raw["stress_execution"] = {"liquidated_at": 0}
    (account / "account.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="survival accounting"):
        evaluate_mod.build_evaluation_inputs(
            strategy_id="flow_mom_top20", unit_run=unit, account_run=account,
        )


def test_informational_falls_back_to_frame_mean(tmp_path) -> None:
    """Missing mean_exposure still reports the frame mean outside the digest."""
    account = tmp_path / "acct"
    _write_account_run(account)
    raw = json.loads((account / "account.json").read_text(encoding="utf-8"))
    del raw["mean_exposure"]
    (account / "account.json").write_text(json.dumps(raw), encoding="utf-8")
    info = evaluate_mod._account_informational(account)
    assert info["deployed_mean_exposure"] == pytest.approx(1.2)
    assert set(info) == {"deployed_cagr", "deployed_mdd", "deployed_stress_mdd",
                         "deployed_max_leverage", "deployed_mean_exposure"}


def test_liquidated_account_has_finite_returns_and_initial_loss(tmp_path) -> None:
    account = tmp_path / "acct"
    _write_account_run(account, n=3)
    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    pd.DataFrame({"equity": [900.0, 0.0, 0.0], "exposure": [1.2, 0.0, 0.0]}, index=index).to_parquet(
        account / "account_stress_daily.parquet",
    )
    raw = json.loads((account / "account.json").read_text())
    raw["stress_execution"]["liquidated_at"] = index[1].isoformat()
    (account / "account.json").write_text(json.dumps(raw))
    _, stress, _, _, liquidated, _ = evaluate_mod._read_account_run(account)
    assert stress.to_numpy() == pytest.approx([-0.1, -1.0, 0.0])
    assert liquidated is True
    assert evaluate_mod._account_informational(account)["deployed_stress_mdd"] == -1.0


@pytest.mark.parametrize(("equity", "capital"), [
    ([1000.0, 0.0, 100.0], 1000.0), ([1000.0, -1.0, 0.0], 1000.0),
    ([1000.0, float("nan"), 0.0], 1000.0), ([1000.0], 0.0),
])
def test_invalid_account_equity_fails_closed(tmp_path, equity, capital) -> None:
    from src.common.errors import DataIntegrityError

    account = tmp_path / "acct"
    _write_account_run(account, n=len(equity))
    index = pd.date_range("2025-01-01", periods=len(equity), tz="UTC")
    pd.DataFrame({"equity": equity}, index=index).to_parquet(account / "account_stress_daily.parquet")
    raw = json.loads((account / "account.json").read_text())
    raw["capital"] = capital
    (account / "account.json").write_text(json.dumps(raw))
    with pytest.raises(DataIntegrityError, match=r"equity|capital"):
        evaluate_mod._read_account_run(account)


@pytest.mark.parametrize("missing", ["account.json", "capital", "liquidated_at", "initial_margin_breaches", "stress_execution"])
def test_missing_account_evidence_fails_closed(tmp_path, missing) -> None:
    from src.common.errors import DataIntegrityError

    account = tmp_path / "acct"
    _write_account_run(account)
    path = account / "account.json"
    if missing == "account.json":
        path.unlink()
    else:
        raw = json.loads(path.read_text())
        del raw[missing]
        path.write_text(json.dumps(raw))
    with pytest.raises(DataIntegrityError, match="survival accounting"):
        evaluate_mod._read_account_run(account)


def test_stress_drawdown_includes_initial_capital(tmp_path) -> None:
    account = tmp_path / "acct"
    _write_account_run(account, n=2)
    index = pd.date_range("2025-01-01", periods=2, tz="UTC")
    pd.DataFrame({"equity": [400.0, 600.0]}, index=index).to_parquet(account / "account_stress_daily.parquet")
    assert evaluate_mod._account_informational(account)["deployed_stress_mdd"] == pytest.approx(-0.6)


def test_informational_values_do_not_change_evaluation_digest(tmp_path, monkeypatch) -> None:
    unit, account = tmp_path / "unit", tmp_path / "acct"
    _write_unit_run(unit)
    _write_account_run(account)
    monkeypatch.setattr(evaluate_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    args = _args(unit_run=str(unit), account_run=str(account))
    with pytest.raises(SystemExit):
        evaluate_mod.run_evaluate_strategy_command(args)
    output = next((tmp_path / "backtests" / "evaluation").glob("*.json"))
    first = json.loads(output.read_text())
    raw = json.loads((account / "account.json").read_text())
    raw.update(cagr=0.9, mdd=-0.8, mean_exposure=5.0)
    (account / "account.json").write_text(json.dumps(raw))
    with pytest.raises(SystemExit):
        evaluate_mod.run_evaluate_strategy_command(args)
    outputs = list((tmp_path / "backtests" / "evaluation").glob("*.json"))
    second = max((json.loads(path.read_text()) for path in outputs), key=lambda item: item["informational"]["deployed_cagr"])
    assert first["informational"] != second["informational"]
    assert first["digest"] == second["digest"]
    assert first["verdict"] == second["verdict"]
