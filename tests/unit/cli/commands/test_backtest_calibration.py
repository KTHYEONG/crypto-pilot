"""Invariant scenarios for the backtest fill-calibration CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.application._fill_calibration_helpers import (
    _args,
)



def test_post_design_calibration_journals_look_before_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import src.application.fill_calibration as cal
    import src.cli.commands.backtest_calibration as cli
    from tests.unit.cli.commands._backtest_helpers import _patch_releases_root

    root = _patch_releases_root(monkeypatch, tmp_path)
    journal = root / "flow_mom.holdout.jsonl"
    journaled_first: list[bool] = []

    def fake(request, *, output_dir):
        journaled_first.append(journal.exists())
        return cal.FillCalibrationReport({}, None, {}, tmp_path / "i", tmp_path / "r")

    monkeypatch.setattr(cal, "run_fill_calibration", fake)
    cli.run_fill_calibration_command(_args(start="2026-08-01", end="2026-09-01", output_dir=str(tmp_path / "out")))
    assert journaled_first == [True]
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [(r["window_start"], r["window_end"], r["purpose"]) for r in rows] == [
        ("2026-08-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00", "fill_calibration"),
    ]

def test_cli_preserves_zero_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    import src.cli.commands.backtest_calibration as cli

    captured = []
    def fake(request, *, output_dir):
        captured.append(request)
        return cal.FillCalibrationReport({}, None, {}, tmp_path / "i", tmp_path / "r")
    monkeypatch.setattr(cal, "run_fill_calibration", fake)
    cli.run_fill_calibration_command(_args(seed=0, queue_ahead_fraction=0.0))
    assert captured[0].seed == 0
    assert captured[0].queue_ahead_fraction == 0.0
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(touch_lookback_s=0.0))
    assert exc.value.code == 2


def test_cli_happy_and_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import src.application.fill_calibration as cal
    import src.cli.commands.backtest_calibration as cli

    seen = {}

    def _fake_run(request, *, output_dir):
        seen["request"] = request
        seen["output_dir"] = output_dir
        return cal.FillCalibrationReport(backtest={"periods": {"design": {"rules": {"ohlcv_strict@K1": {"n_intents": 2, "fill_fraction": 0.5}}, "marketable_anchor_share": 0.1}}}, paper=None, exclusions={}, intents_path=tmp_path / "i.parquet", report_path=tmp_path / "r.json")

    monkeypatch.setattr(cli, "run_fill_calibration_command", cli.run_fill_calibration_command)
    import src.application.fill_calibration as calmod

    monkeypatch.setattr(calmod, "run_fill_calibration", _fake_run)
    args = _args(output_dir=str(tmp_path / "out"))
    cli.run_fill_calibration_command(args)
    out = capsys.readouterr().out
    assert "fill-calibration period=design" in out
    assert seen["request"].k_bars == (1, 3, 5, 10)
    assert seen["request"].touch_lookback_s == 60.0
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(start="2023-02-01", end="2023-01-01"))
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(k_bars=[0]))
    assert exc.value.code == 2

    def _boom_run(request, *, output_dir):
        from src.common.errors import DataIntegrityError as _D

        raise _D("bad tape")

    monkeypatch.setattr(calmod, "run_fill_calibration", _boom_run)
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(output_dir=str(tmp_path / "out2")))
    assert exc.value.code == 2
    assert "bad tape" in capsys.readouterr().err
    monkeypatch.setattr(calmod, "run_fill_calibration", _fake_run)
    import src.evaluation.holdout as holdout_mod

    def _integrity(*args, **kwargs):
        from src.common.errors import DataIntegrityError as _D

        raise _D("journal unreadable")

    monkeypatch.setattr(holdout_mod, "record_holdout_look", _integrity)
    with pytest.raises(SystemExit, match="journal unreadable"):
        cli.run_fill_calibration_command(_args(start="2026-08-01", end="2026-09-01", output_dir=str(tmp_path / "out3")))

    def _nan_run(request, *, output_dir):
        return cal.FillCalibrationReport(backtest={"periods": {"design": {"rules": {}, "marketable_anchor_share": None}}}, paper=None, exclusions={}, intents_path=tmp_path / "i.parquet", report_path=tmp_path / "r.json")

    monkeypatch.setattr(calmod, "run_fill_calibration", _nan_run)
    cli.run_fill_calibration_command(_args(output_dir=str(tmp_path / "out4")))
    assert "nan" in capsys.readouterr().out
