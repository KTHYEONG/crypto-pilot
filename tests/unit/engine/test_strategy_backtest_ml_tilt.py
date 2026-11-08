"""Invariant scenarios for the ML tilt variant of the strategy historical runner."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.engine.strategy_backtest as run_mod
from src.common.errors import DataIntegrityError
from src.strategy.ml_tilt import train_missing_boundaries
from src.strategy.release import strategy_signal_digest
from src.strategy.targets import FLOW_MOM_TOP20, FLOW_MOM_TOP20_ACCOUNT_UNIT, FLOW_MOM_TOP20_GROWTH
from tests.fixtures.ml_tilt_source import EVAL_START, EVAL_END, request, synthetic_source, ml_test_values


@pytest.fixture(autouse=True)
def _ml_values_and_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_mod, "assert_strategy_execution_coverage", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "ml_tilt_values", lambda *a, **k: ml_test_values())


def _no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay must not start before the ML fail-closed checks")

    monkeypatch.setattr(run_mod, "_strategy_window_stream", _boom)


def _train(directory: Path) -> None:
    vals = ml_test_values()
    req = request(strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    src = synthetic_source(req)
    features = run_mod.build_ml_tilt_features(src, req.strategy, req.base_spec, vals)
    train_missing_boundaries(
        features,
        entry_marks=src.hourly_panels["close"].shift(freq=pd.Timedelta(hours=1)),
        funding_by_symbol=src.funding_by_symbol,
        hourly_close=src.hourly_panels["close"],
        until=EVAL_END,
        out_dir=directory,
        values=vals,
        strategy=req.strategy,
    )


def test_request_rejects_ml_on_levered_spec(tmp_path: Path) -> None:
    """The ML tilt is defined only on the unit clip book, never on the growth rung or an unclipped book."""
    with pytest.raises(DataIntegrityError, match="unit clip book"):
        request(strategy=FLOW_MOM_TOP20_GROWTH, ml_tilt_model_dir=tmp_path)
    with pytest.raises(DataIntegrityError, match="unit clip book"):
        request(strategy=FLOW_MOM_TOP20, ml_tilt_model_dir=tmp_path)


def test_no_model_dir_leaves_candidate_untouched() -> None:
    """Without a model directory the builder emits the baseline book and no ML digest."""
    req = request(strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    candidate, context = run_mod.build_request_targets(req, source=synthetic_source(req))
    again, _ = run_mod.build_request_targets(req, source=synthetic_source(req))
    pd.testing.assert_frame_equal(candidate.target_weights, again.target_weights)
    assert context.ml_tilt_digest is None


def test_ml_targets_keep_c0_gross_and_dollar_neutrality(tmp_path: Path) -> None:
    """Each tilted row keeps the C0 row gross and stays dollar neutral."""
    _train(tmp_path)
    base_req = request(strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    src = synthetic_source(base_req)
    base, _ = run_mod.build_request_targets(base_req, source=src)
    tilted, context = run_mod.build_request_targets(request(ml_tilt_model_dir=tmp_path), source=src)
    c0 = base.target_weights.to_numpy(dtype="float64")
    c1 = tilted.target_weights.to_numpy(dtype="float64")
    np.testing.assert_allclose(np.abs(c1).sum(axis=1), np.abs(c0).sum(axis=1), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(c1.sum(axis=1), 0.0, atol=1e-12)
    assert not np.allclose(c0, c1)
    assert context.ml_tilt_digest is not None
    assert context.ml_tilt_digest != strategy_signal_digest(FLOW_MOM_TOP20_ACCOUNT_UNIT)


def test_ml_window_before_anchor_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A decision before the schedule anchor raises before any replay starts."""
    late = dataclasses.replace(ml_test_values(), schedule_anchor=EVAL_START)
    monkeypatch.setattr(run_mod, "ml_tilt_values", lambda *a, **k: late)
    _no_replay(monkeypatch)
    req = request(ml_tilt_model_dir=tmp_path)
    with pytest.raises(DataIntegrityError, match="schedule anchor"):
        run_mod.run_strategy_backtest(req, source=synthetic_source(req))


def test_ml_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty model directory raises a missing-boundary error before any replay starts."""
    _no_replay(monkeypatch)
    req = request(ml_tilt_model_dir=tmp_path)
    with pytest.raises(DataIntegrityError, match="required boundary model is missing"):
        run_mod.run_strategy_backtest(req, source=synthetic_source(req))


def test_ml_run_reports_ml_digest(tmp_path: Path) -> None:
    """The runner exposes the ML digest on the run so persistence and trials can key on it."""
    _train(tmp_path)
    req = request(ml_tilt_model_dir=tmp_path)
    candidate, context = run_mod.build_request_targets(req, source=synthetic_source(req))
    assert context.ml_tilt_digest == run_mod.ml_tilt_model_digest(req.strategy, tmp_path)
    assert candidate.strategy.strategy_id == FLOW_MOM_TOP20_ACCOUNT_UNIT.strategy_id


def test_persist_records_ml_digest(tmp_path: Path) -> None:
    """Persisted signal_digest is the ML digest when present and the baseline digest otherwise."""
    import json

    from src.engine.backtest_persist import persist_strategy_backtest
    from tests.unit.engine.test_strategy_backtest_report import _run

    base = _run()
    baseline_digest = strategy_signal_digest(base.candidate.strategy)
    (tmp_path / "baseline").mkdir()
    persist_strategy_backtest(base, tmp_path / "baseline" / "result.json")
    baseline = json.loads((tmp_path / "baseline" / "result.json").read_text(encoding="utf-8"))
    assert baseline["signal_digest"] == baseline_digest
    assert baseline["ml_tilt_digest"] is None

    ml_run = dataclasses.replace(base, ml_tilt_digest="ml-digest-x")
    (tmp_path / "ml").mkdir()
    persist_strategy_backtest(ml_run, tmp_path / "ml" / "result.json")
    ml = json.loads((tmp_path / "ml" / "result.json").read_text(encoding="utf-8"))
    assert ml["signal_digest"] == "ml-digest-x"
    assert ml["ml_tilt_digest"] == "ml-digest-x"
    assert ml["signal_digest"] != baseline_digest


def test_ml_needs_no_model_after_last_scored_decision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A boundary after the last scored decision is not required (decisions past the evaluation are ignored)."""
    vals = dataclasses.replace(ml_test_values(), retrain_days=18)
    monkeypatch.setattr(run_mod, "ml_tilt_values", lambda *a, **k: vals)
    req = request(strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    src = synthetic_source(req)
    features = run_mod.build_ml_tilt_features(src, req.strategy, req.base_spec, vals)
    rows = train_missing_boundaries(
        features,
        entry_marks=src.hourly_panels["close"].shift(freq=pd.Timedelta(hours=1)),
        funding_by_symbol=src.funding_by_symbol,
        hourly_close=src.hourly_panels["close"],
        until=EVAL_END - pd.Timedelta(days=2),
        out_dir=tmp_path,
        values=vals,
        strategy=req.strategy,
    )
    assert [r.boundary for r in rows] == [vals.schedule_anchor]
    tilted, context = run_mod.build_request_targets(request(ml_tilt_model_dir=tmp_path), source=src)
    assert context.ml_tilt_digest is not None
    assert len(tilted.target_weights) > 0
