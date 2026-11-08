"""CLI scenarios for the ML tilt backtest flags and the ml-train boundary command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
import src.engine.strategy_backtest as run_mod
from src.common.errors import DataIntegrityError
from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT
from tests.fixtures.ml_tilt_source import ml_test_values, synthetic_source, request
from tests.unit.cli.commands._backtest_helpers import _parse, _strategy_argv


@pytest.fixture
def ml_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backtest_mod, "ml_tilt_values", lambda *a, **k: ml_test_values())
    monkeypatch.setattr(run_mod, "ml_tilt_values", lambda *a, **k: ml_test_values())


def _ml_train_argv(out: Path) -> list[str]:
    return ["backtest", "ml-train", "--until", "2021-03-20", "--out", str(out)]


def test_trial_override_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ML trial is appended under the ML signal digest, never the baseline spec digest."""
    import src.strategy.release as release_mod

    monkeypatch.setattr(release_mod, "releases_dir", lambda: tmp_path)
    start = pd.Timestamp("2021-03-05", tz="UTC")
    end = pd.Timestamp("2021-03-20", tz="UTC")
    backtest_mod._record_trial_for_window(
        FLOW_MOM_TOP20_ACCOUNT_UNIT, start, end, 0.5, 10, spec_digest_override="ml-digest-x",
    )
    rows = [json.loads(line) for line in next(tmp_path.glob("flow_mom.trials.jsonl")).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["spec_digest"] == "ml-digest-x"


def test_cli_ml_refuses_neighbors(tmp_path: Path, ml_values: None) -> None:
    """Discovery neighbors cannot be scored on an ML variant."""
    argv = _strategy_argv(tmp_path, "--ml-tilt-models", str(tmp_path / "models"), "--neighbors")
    args = _parse(argv)
    with pytest.raises(SystemExit, match="neighbors"):
        backtest_mod.run_strategy_backtest_command(args)


def test_cli_ml_refuses_growth_variant(tmp_path: Path, ml_values: None) -> None:
    """The ML tilt never runs on the levered growth rung."""
    argv = _strategy_argv(tmp_path, "--ml-tilt-models", str(tmp_path / "models"), "--variant", "growth")
    with pytest.raises(SystemExit, match="unit clip book"):
        backtest_mod.run_strategy_backtest_command(_parse(argv))


def test_ml_train_refuses_example_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training is refused while the public example store is active."""
    import types

    import src.strategy.store as store_mod

    monkeypatch.setattr(store_mod, "strategy_values", lambda *a, **k: types.SimpleNamespace(example=True))
    with pytest.raises(DataIntegrityError, match="example store"):
        backtest_mod.run_ml_train_command(_parse(_ml_train_argv(tmp_path / "models")))


@pytest.fixture
def train_source(monkeypatch: pytest.MonkeyPatch, ml_values: None) -> None:
    monkeypatch.setattr(run_mod, "load_strategy_source", lambda req: synthetic_source(request(strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)))
    monkeypatch.setattr(backtest_mod, "refuse_example_store", lambda action: None)


def test_ml_train_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture, train_source: None) -> None:
    """A rerun keeps valid models byte-identical and reports them as existing."""
    out = tmp_path / "models"
    backtest_mod.run_ml_train_command(_parse(_ml_train_argv(out)))
    first = capsys.readouterr().out.strip().splitlines()
    files = sorted(out.glob("*.json"))
    assert len(files) == 1
    assert first[0].endswith("trained")
    before = files[0].read_bytes()
    backtest_mod.run_ml_train_command(_parse(_ml_train_argv(out)))
    second = capsys.readouterr().out.strip().splitlines()
    assert second[0].endswith("exists")
    assert files[0].read_bytes() == before


def test_ml_train_invalid_existing_file(tmp_path: Path, train_source: None) -> None:
    """A corrupted model file fails the command non-zero and is never overwritten."""
    out = tmp_path / "models"
    out.mkdir()
    corrupted = out / "20210301.json"
    corrupted.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="ml-train failed"):
        backtest_mod.run_ml_train_command(_parse(_ml_train_argv(out)))
    assert corrupted.read_text(encoding="utf-8") == "{not json"


def test_cli_ml_refuses_non_top20_breadth(tmp_path: Path, ml_values: None) -> None:
    """The ML tilt is registered only on the Top-20 unit book."""
    argv = _strategy_argv(tmp_path, "--ml-tilt-models", str(tmp_path / "models"), "--breadth", "40")
    with pytest.raises(SystemExit, match="Top-20 unit clip book"):
        backtest_mod.run_strategy_backtest_command(_parse(argv))


def test_cli_ml_rejects_corrupted_model_dir(tmp_path: Path, ml_values: None) -> None:
    """A corrupted model file stops the strategy run before any replay or trial is recorded."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "20210301.json").write_text("{not json", encoding="utf-8")
    argv = _strategy_argv(tmp_path, "--ml-tilt-models", str(models))
    with pytest.raises(SystemExit, match="invalid ml tilt models"):
        backtest_mod.run_strategy_backtest_command(_parse(argv))


def test_ml_train_requires_ml_tilt_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training refuses to run when the store carries no ml_tilt values."""
    monkeypatch.setattr(backtest_mod, "refuse_example_store", lambda action: None)
    monkeypatch.setattr(backtest_mod, "ml_tilt_values", lambda *a, **k: None)
    with pytest.raises(SystemExit, match=r"\[ml_tilt\]"):
        backtest_mod.run_ml_train_command(_parse(_ml_train_argv(tmp_path / "models")))


def test_ml_train_skips_boundaries_past_observed_data(tmp_path: Path, capsys: pytest.CaptureFixture, train_source: None) -> None:
    """Boundaries after the observed source end are reported awaiting_data and never written."""
    out = tmp_path / "models"
    backtest_mod.run_ml_train_command(_parse(["backtest", "ml-train", "--until", "2022-12-31", "--out", str(out)]))
    lines = capsys.readouterr().out.strip().splitlines()
    trained = [line for line in lines if line.endswith("trained")]
    waiting = [line for line in lines if line.endswith("awaiting_data")]
    assert trained
    assert waiting
    assert len(list(out.glob("*.json"))) == len(trained)


def test_cli_ml_digest_change_during_run_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ml_values: None) -> None:
    """A run whose model digest differs from the journaled digest is never persisted."""
    import types

    import src.engine.backtest_persist as persist_mod

    monkeypatch.setattr(run_mod, "ml_tilt_model_digest", lambda strategy, model_dir: "journaled-digest")
    monkeypatch.setattr(run_mod, "run_strategy_backtest", lambda *a, **k: types.SimpleNamespace(ml_tilt_digest="other-digest"))
    monkeypatch.setattr(backtest_mod, "_record_trial_for_window", lambda *a, **k: None)

    def _never(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not persist a mismatched ML run")

    monkeypatch.setattr(persist_mod, "persist_strategy_backtest", _never)
    argv = _strategy_argv(tmp_path, "--ml-tilt-models", str(tmp_path / "models"))
    with pytest.raises(SystemExit, match="models changed"):
        backtest_mod.run_strategy_backtest_command(_parse(argv))
