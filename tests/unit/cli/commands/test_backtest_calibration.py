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


def _parity_args(**overrides):
    args = type("A", (), {})()
    args.day = None
    args.paper_run_dir = None
    args.archive_root = None
    args.live_capture_root = None
    args.output = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_fill_parity_run_dir_auto_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.cli.commands.backtest_calibration as cli
    import src.common.paths as paths

    runs = tmp_path / "state" / "runs"
    (runs / "a" / "audit").mkdir(parents=True)
    (runs / "b" / "audit").mkdir(parents=True)
    (runs / "a" / "audit" / "2023-01-02.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    resolved = cli._resolve_parity_run_dir(None, __import__("datetime").date(2023, 1, 2))
    assert resolved == runs / "a"
    (runs / "b" / "audit" / "2023-01-02.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli._resolve_parity_run_dir(None, __import__("datetime").date(2023, 1, 2))
    assert exc.value.code == 2
    for p in (runs / "a" / "audit" / "2023-01-02.jsonl", runs / "b" / "audit" / "2023-01-02.jsonl"):
        p.unlink()
    with pytest.raises(SystemExit) as exc:
        cli._resolve_parity_run_dir(None, __import__("datetime").date(2023, 1, 2))
    assert exc.value.code == 2


def test_fill_parity_breach_exits_3_after_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from datetime import date

    import src.cli.commands.backtest_calibration as cli
    import src.application.fill_parity as parity

    day = date(2023, 1, 2)
    run_dir = tmp_path / "run1"
    (run_dir / "audit").mkdir(parents=True)
    output = tmp_path / "fill_parity.jsonl"
    for i in range(7):
        parity.append_parity_row(
            parity.FillParityRow(
                day=date(2023, 1, 2 + i) if i else date(2023, 1, 3),
                run_id="run1", n_intents=30, n_posts=10,
                through_agreement=0.99, paper_is_bps=4.0, rule_is_bps=2.0,
                gap_bps=2.0, status="OK", detail=None,
            ),
            output,
        )
    fake_row = parity.FillParityRow(
        day=day, run_id="run1", n_intents=30, n_posts=10,
        through_agreement=0.99, paper_is_bps=4.0, rule_is_bps=2.0,
        gap_bps=2.0, status="OK", detail=None,
    )
    monkeypatch.setattr(parity, "measure_fill_parity", lambda *a, **k: fake_row)
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_parity_command(_parity_args(day="2023-01-02", paper_run_dir=str(run_dir), output=str(output)))
    assert exc.value.code == 3
    out = capsys.readouterr().out
    assert "fill-parity breach=" in out
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 8


def test_fill_parity_default_day_is_yesterday_utc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    import src.cli.commands.backtest_calibration as cli
    import src.application.fill_parity as parity

    seen: dict[str, object] = {}

    def _fake_measure(day, *, paper_run_dir, archive_root, live_capture_root):
        seen["day"] = day
        return parity.FillParityRow(
            day=day, run_id="run1", n_intents=0, n_posts=0,
            through_agreement=None, paper_is_bps=None, rule_is_bps=None,
            gap_bps=None, status="SKIPPED", detail="no_intents",
        )

    monkeypatch.setattr(parity, "measure_fill_parity", _fake_measure)
    monkeypatch.setattr(parity, "trailing_breach", lambda path, thresholds: None)
    real_now = pd.Timestamp.now

    class _Frozen:
        @staticmethod
        def now(tz=None):
            return real_now(tz="UTC").tz_convert("UTC").normalize() + pd.Timedelta(days=5)

    import datetime as _dt

    frozen_today = _dt.date(2023, 1, 10)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: pd.Timestamp(frozen_today, tz="UTC")))
    run_dir = tmp_path / "run1"
    (run_dir / "audit").mkdir(parents=True)
    cli.run_fill_parity_command(
        _parity_args(paper_run_dir=str(run_dir), output=str(tmp_path / "out.jsonl"))
    )
    assert seen["day"] == _dt.date(2023, 1, 9)


def test_fill_parity_invalid_day_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import src.cli.commands.backtest_calibration as cli

    with pytest.raises(SystemExit) as exc:
        cli.run_fill_parity_command(_parity_args(day="not-a-day", paper_run_dir=str(tmp_path), output=str(tmp_path / "o.jsonl")))
    assert exc.value.code == 2
    assert "invalid --day" in capsys.readouterr().err


def test_fill_parity_resolve_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import src.cli.commands.backtest_calibration as cli

    with pytest.raises(SystemExit) as exc:
        cli.run_fill_parity_command(_parity_args(day="2023-01-02", output=str(tmp_path / "o.jsonl")))
    assert exc.value.code == 2

    def _raise_value(raw, day):
        raise ValueError("resolve boom")

    monkeypatch.setattr(cli, "_resolve_parity_run_dir", _raise_value)
    with pytest.raises(SystemExit) as exc2:
        cli.run_fill_parity_command(_parity_args(day="2023-01-02", output=str(tmp_path / "o2.jsonl")))
    assert exc2.value.code == 2
    assert "resolve boom" in capsys.readouterr().err


def test_fill_parity_fmt_nan_and_integrity_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from datetime import date

    import src.cli.commands.backtest_calibration as cli
    import src.application.fill_parity as parity
    from src.common.errors import DataIntegrityError

    run_dir = tmp_path / "run1"
    (run_dir / "audit").mkdir(parents=True)
    weird = parity.FillParityRow(
        day=date(2023, 1, 2), run_id="run1", n_intents=1, n_posts=1,
        through_agreement=float("nan"), paper_is_bps=float("nan"), rule_is_bps=float("nan"),
        gap_bps=float("nan"), status="OK", detail=None,
    )
    monkeypatch.setattr(parity, "measure_fill_parity", lambda *a, **k: weird)
    monkeypatch.setattr(parity, "trailing_breach", lambda path, thresholds: None)
    cli.run_fill_parity_command(_parity_args(day="2023-01-02", paper_run_dir=str(run_dir), output=str(tmp_path / "o.jsonl")))
    assert "nan" in capsys.readouterr().out

    bad_row = parity.FillParityRow(
        day=date(2023, 1, 2), run_id="run1", n_intents=1, n_posts=1,
        through_agreement="bad", paper_is_bps="bad", rule_is_bps="bad",  # type: ignore[arg-type]
        gap_bps="bad", status="OK", detail=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(parity, "measure_fill_parity", lambda *a, **k: bad_row)
    cli.run_fill_parity_command(_parity_args(day="2023-01-02", paper_run_dir=str(run_dir), output=str(tmp_path / "o2.jsonl")))
    assert "nan" in capsys.readouterr().out

    def _boom(*args, **kwargs):
        raise DataIntegrityError("parity boom")

    monkeypatch.setattr(parity, "measure_fill_parity", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_parity_command(_parity_args(day="2023-01-02", paper_run_dir=str(run_dir), output=str(tmp_path / "o3.jsonl")))
    assert exc.value.code == 2
    assert "parity boom" in capsys.readouterr().err
