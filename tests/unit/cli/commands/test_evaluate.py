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
from src.strategy.release import load_release

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
        "limitations": [],
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
    equity = 2100.0 * np.cumprod(1.0 + 0.001 + 0.002 * rng.standard_normal(n))
    pd.DataFrame(
        {"equity": equity, "exposure": np.full(n, 1.2)},
        index=index,
    ).to_parquet(run_dir / "account_daily.parquet")
    (run_dir / "account.json").write_text(json.dumps({"capital": 2100.0, "execution": "maker"}), encoding="utf-8")


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
    daily["fill_participation"] = 0.004
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
                         (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-09", tz="UTC")), path=look_path)
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
