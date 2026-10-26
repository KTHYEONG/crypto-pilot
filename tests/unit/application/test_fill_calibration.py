"""Invariant scenarios for the fill calibration request, and persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import src.application.fill_calibration_backtest as fcb
from src.application.fill_calibration import FillCalibrationRequest, run_fill_calibration
from src.common.errors import DataIntegrityError
from tests.unit.application._fill_calibration_helpers import (
    _ts,
    _request,
    _toy_parts,
    _install_tape,
    _slice_frame,
    _coverage_row,
    _day_ms,
)


def test_existing_output_dir_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    monkeypatch.setattr(cal, "_utc_now_stamp", lambda: "20230101T000000Z")
    out = tmp_path / "out"
    (out / "20230101T000000Z").mkdir(parents=True)
    sentinel = out / "20230101T000000Z" / "report.json"
    sentinel.write_text("{}", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        run_fill_calibration(_request(tmp_path), output_dir=out)
    assert sentinel.read_text(encoding="utf-8") == "{}"


def test_request_validation_matrix(tmp_path: Path) -> None:
    good = {"evaluation_start": _ts("2023-01-02T00:00:00Z"), "evaluation_end": _ts("2023-01-04T00:00:00Z"), "archive_root": tmp_path}
    bads = [
        {"evaluation_start": pd.Timestamp("2023-01-02")},
        {"evaluation_start": pd.Timestamp("2023-01-02 09:00:00+09:00")},
        {"evaluation_start": _ts("2023-01-05T00:00:00Z")},
        {"k_bars": ()},
        {"k_bars": [1, 3]},
        {"k_bars": (3, 1)},
        {"k_bars": (1, 1)},
        {"k_bars": (0,)},
        {"k_bars": (True,)},
        {"touch_lookback_s": 0.0},
        {"touch_lookback_s": float("nan")},
        {"bootstrap_samples": 50},
        {"bootstrap_samples": True},
        {"paper_run_dir": tmp_path},
        {"live_capture_root": tmp_path},
        {"archive_root": "nope"},
        {"data_root": 3},
        {"seed": -1},
        {"touch_lookback_s": 0.0001},
        {"queue_ahead_fraction": 1.1},
    ]
    for overrides in bads:
        kwargs = {**good, **overrides}
        with pytest.raises(ValueError, match=r".+"):
            FillCalibrationRequest(**kwargs)


def test_helper_branches() -> None:
    import src.application.fill_calibration as cal

    assert fcb.kind_of(0.0, 1.0) == "open"
    assert fcb.kind_of(1.0, 0.0) == "close"
    assert fcb.kind_of(1.0, -1.0) == "flip"
    assert fcb.kind_of(1.0, 2.0) == "increase"
    assert fcb.kind_of(2.0, 1.0) == "reduce"
    assert cal._round_json(float("nan")) is None
    rounded = cal._round_json({"a": [1, 2.123456789, True, None], "b": (1.5,)})
    assert rounded == {"a": [1, 2.123457, True, None], "b": [1.5]}
    assert cal._round_json("x") == "x"


def test_run_guards_and_write_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")}
    frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    with pytest.raises(DataIntegrityError, match="FillCalibrationRequest"):
        run_fill_calibration("nope", output_dir=tmp_path / "out")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match="output_dir must be a Path"):
        run_fill_calibration(_request(tmp_path), output_dir="nope")  # type: ignore[arg-type]
    bad = _request(tmp_path)
    object.__setattr__(bad, "archive_root", "nope")
    with pytest.raises(DataIntegrityError, match="archive_root must be a Path"):
        run_fill_calibration(bad, output_dir=tmp_path / "out")
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (_ for _ in ()).throw(DataIntegrityError("boom")))
    with pytest.raises(DataIntegrityError, match="malformed calibration inputs"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out2")

    def _boom_inputs(request):
        raise DataIntegrityError("boom")

    monkeypatch.setattr(cal, "_load_backtest_inputs", _boom_inputs)
    with pytest.raises(DataIntegrityError, match="malformed calibration inputs"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out3")
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    import pathlib as _pl

    real_mkdir = _pl.Path.mkdir

    def _race(self, *args, **kwargs):
        raise FileExistsError("race")

    monkeypatch.setattr(_pl.Path, "mkdir", _race)
    with pytest.raises(DataIntegrityError, match="must be fresh"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out4")
    monkeypatch.setattr(_pl.Path, "mkdir", real_mkdir)
    monkeypatch.setattr(cal, "write_parquet_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(DataIntegrityError, match="intents write failed"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out5")
    import src.common.parquet_io as _pio

    monkeypatch.setattr(cal, "write_parquet_atomic", _pio.write_parquet_atomic)
    import os as _os

    real_replace = _os.replace

    def _report_only_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("report.json"):
            raise OSError("disk full")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(cal.os, "replace", _report_only_replace)
    with pytest.raises(DataIntegrityError, match="report write failed"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out6")
