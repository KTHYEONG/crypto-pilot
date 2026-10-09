"""Invariant scenarios for the exposure scan service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.strategy_account import (
    ExposureScanRequest,
    derive_growth_exposure,
    load_strategy_run_artifacts,
    run_exposure_scan,
)
from tests.unit.application._strategy_account_helpers import (
    _PINNED_NOW,
    _PRE_REFACTOR,
    _fake_run_dir,
    _install_exposure_fakes,
)


def test_exposure_scan_unlevers_by_run_multiplier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The solver receives base returns and mean name weight divided by the run multiplier."""
    from src.core.params import COMMITTEE_GROWTH_HORIZON_YEARS, COMMITTEE_GROWTH_N_PATHS, EXPOSURE_SCAN_MEAN_HAIRCUT, NULL_BOOTSTRAP_MEAN_BLOCK_DAYS

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    report = run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    np.testing.assert_allclose(seen["unit_returns"].to_numpy(), 0.001 / 2.5)
    assert seen["max_name_weight"] == pytest.approx(0.05 / 2.5)
    assert seen["solver_params"]["mean_haircut"] == EXPOSURE_SCAN_MEAN_HAIRCUT
    assert seen["solver_params"]["n_paths"] == COMMITTEE_GROWTH_N_PATHS
    assert seen["solver_params"]["horizon_years"] == COMMITTEE_GROWTH_HORIZON_YEARS
    assert seen["solver_params"]["mean_block_days"] == NULL_BOOTSTRAP_MEAN_BLOCK_DAYS
    assert report.payload["execution_bound"] == "OHLCV_IMMEDIATE_TAKER"

def test_exposure_scan_gap_roster_ignores_trading_exclusions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap roster is built with no trading-exclusion filter."""
    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert seen["blocked_decisions"] is None
    assert seen["breadth"] == 20
    assert seen["selection_mode"] == "causal_history"

def test_exposure_scan_gap_population_restricted_to_delisted_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only DELISTED symbols feed the gap sampler."""
    import src.evaluation.exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset({"AAA"}))
    real_sample = growth_mod.roster_gap_sample
    captured: dict = {}

    def _spy(daily_close, roster, *, threshold):
        captured["columns"] = list(daily_close.columns)
        return real_sample(daily_close, roster, threshold=threshold)

    monkeypatch.setattr(growth_mod, "roster_gap_sample", _spy)
    report = run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert captured["columns"] == ["AAA"]
    assert report.payload["gap_symbols"] == ["AAA"]

def test_exposure_scan_empty_exclusion_registry_skips_sampler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered exclusion yields a zero-event gap sample without calling the sampler."""
    import src.evaluation.exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "roster_gap_sample", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    report = run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert seen["gaps"].events_per_year == 0.0
    assert report.payload["gap_symbols"] == []

def test_exclusion_registry_read_at_most_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exclusion registry is read once for a non-empty census and never for an empty one."""
    import src.evaluation.exposure as growth_mod
    import src.core.panel as panel_mod
    from src.application.strategy_account import AccountReplayError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    calls = []
    real = growth_mod.structurally_excluded_symbols
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert len(calls) == 1

    empty_base = tmp_path / "strategy_empty"
    empty_base.mkdir(parents=True, exist_ok=True)
    (tmp_path / "strategy_run" / "result.json").replace(empty_base / "result.json")
    (tmp_path / "strategy_run" / "daily.parquet").replace(empty_base / "daily.parquet")
    calls.clear()
    monkeypatch.setattr(panel_mod, "load_base_panel", lambda *a, **k: {"close": pd.DataFrame(index=pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")), "quote_vol": pd.DataFrame(index=pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC"))})
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda *a, **k: (calls.append(1), frozenset())[1])
    import contextlib

    with contextlib.suppress(AccountReplayError):
        run_exposure_scan(ExposureScanRequest(run_dir=empty_base))
    assert calls == []

def test_derive_growth_exposure_pure_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed in-memory inputs give equal mappings without a created_at key."""
    _install_exposure_fakes(monkeypatch)
    import src.evaluation.exposure as growth_mod

    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset())
    artifacts = load_strategy_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6, "BBB": 1e6}, index=idx, dtype="float64")
    first = derive_growth_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA", "BBB"), excluded_symbols=frozenset())
    second = derive_growth_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA", "BBB"), excluded_symbols=frozenset())
    assert first == second
    assert "created_at" not in first

def test_exposure_scan_fresh_only_before_any_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing exposure.json fails before any panel load."""
    import src.core.panel as panel_mod
    from src.application.strategy_account import AccountReplayError

    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "exposure.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(panel_mod, "load_base_panel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no load")))
    with pytest.raises(AccountReplayError, match=r"fresh"):
        run_exposure_scan(ExposureScanRequest(run_dir=run_dir))

def test_exposure_scan_solver_rejection_leaves_no_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver rejection raises without persisting exposure.json."""
    import src.evaluation.exposure as growth_mod
    from src.application.strategy_account import AccountReplayError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", lambda *a, **k: (_ for _ in ()).throw(ValueError("no rung")))
    with pytest.raises(AccountReplayError, match=r"exposure scan failed"):
        run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert not (run_dir / "exposure.json").exists()

def test_exposure_scan_invalid_artifacts_rejected(tmp_path: Path) -> None:
    """An empty run dir raises an invalid-artifacts error."""
    from src.application.strategy_account import AccountReplayError

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(AccountReplayError, match=r"invalid strategy run artifacts"):
        run_exposure_scan(ExposureScanRequest(run_dir=run_dir))

def test_exposure_payload_golden_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposure output carries the exact key set with no tmp file left behind."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    report = run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    assert set(report.payload) == {"run_dir", "strategy_id", "execution_bound", "exposure_multiplier", "grid", "growth", "ruin_probability", "argmax", "chosen", "gap_events_per_year", "gap_sample_size", "gap_symbols", "mean_haircut", "plateau_tolerance", "seed", "created_at", "unlever_assumption"}
    assert not (run_dir / "exposure.json.tmp").exists()

def test_exposure_payload_matches_pre_refactor_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """exposure.json is byte-identical to the pre-refactor CLI output (created_at aside)."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: _PINNED_NOW))
    report = run_exposure_scan(ExposureScanRequest(run_dir=run_dir))
    text = report.path.read_text(encoding="utf-8").replace(f'"created_at": "{_PINNED_NOW.isoformat()}", ', "")
    assert text == _PRE_REFACTOR["exposure"]

def test_derive_roster_integrity_error_is_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster DataIntegrityError surfaces as ValueError from the derivation."""
    import src.strategy.universe as universe_mod
    from src.common.errors import DataIntegrityError

    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(universe_mod, "build_pit_roster", lambda *a, **k: (_ for _ in ()).throw(DataIntegrityError("roster boom")))
    artifacts = load_strategy_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6}, index=idx, dtype="float64")
    with pytest.raises(ValueError, match=r"roster boom"):
        derive_growth_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA",), excluded_symbols=frozenset())

def test_derive_unexpected_solver_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ValueError solver failure propagates unwrapped."""
    import src.evaluation.exposure as growth_mod

    _install_exposure_fakes(monkeypatch)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("solver bug")))
    artifacts = load_strategy_run_artifacts(_fake_run_dir(tmp_path))
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx, dtype="float64")
    volume = pd.DataFrame({"AAA": 1e6}, index=idx, dtype="float64")
    with pytest.raises(RuntimeError, match=r"solver bug"):
        derive_growth_exposure(artifacts, run_name="r", daily_close=close, daily_quote_volume=volume, census=("AAA",), excluded_symbols=frozenset())
