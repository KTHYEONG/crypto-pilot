"""Tests for src/lab/mhs/telemetry.py — StageTelemetry and Tag."""

from __future__ import annotations

import json
import logging

import pytest

from src.lab.mhs.telemetry import StageTelemetry, Tag, _format_value


class TestFormatValue:
    def test_float_three_dp(self):
        assert _format_value(0.9231) == "0.923"

    def test_integer(self):
        assert _format_value(42) == "42"

    def test_string(self):
        assert _format_value("hello") == "hello"

    def test_short_tuple(self):
        result = _format_value(("a", "b", "c"))
        assert result == "(a, b, c)"

    def test_long_tuple_truncated(self):
        result = _format_value(("a", "b", "c", "d", "e", "f", "g"))
        assert result == "(a, b, c, d, e) truncated=2"

    def test_list_truncated(self):
        result = _format_value([1, 2, 3, 4, 5, 6])
        assert result == "(1, 2, 3, 4, 5) truncated=1"


class TestTag:
    def test_tag_members(self):
        assert Tag.SYS == "SYS"
        assert Tag.DATA == "DATA"
        assert Tag.ALGO == "ALGO"
        assert Tag.EVAL == "EVAL"
        assert {Tag.PORTFOLIO, Tag.RISK, Tag.EXEC} == {"PORTFOLIO", "RISK", "EXEC"}

    def test_tag_is_strenum(self):
        assert len(Tag) == 7


class TestStageTelemetry:
    def test_log_emits_message(self, caplog):
        """SCENARIO_ANALYSIS_ARCHITECTURE_02: StageTelemetry.log emits [TAG] stage=... k=v ..."""
        from io import StringIO

        from src.lab.mhs.telemetry import TELEMETRY_LOGGER_NAME

        telemetry = StageTelemetry(log_run=False)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        telemetry._logger.addHandler(handler)
        try:
            with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER_NAME):
                telemetry.log(Tag.ALGO, "committee_book", gross=0.9231, members=("a", "b", "c", "d", "e", "f", "g"))
            output = stream.getvalue()
            assert "[ALGO]" in output
            assert "stage=committee_book" in output
            assert "gross=0.923" in output
            assert "truncated=2" in output
            assert any(record.name == TELEMETRY_LOGGER_NAME and record.getMessage() in output for record in caplog.records)
        finally:
            telemetry._logger.removeHandler(handler)

    def test_construction_attaches_no_handler_and_opens_no_file(self):
        from src.common.logging import LOG_DIR
        from src.lab.mhs.telemetry import TELEMETRY_LOGGER_NAME

        logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
        handlers_before = list(logger.handlers)
        propagate_before = logger.propagate
        level_before = logger.level
        before = {p.relative_to(LOG_DIR): p.read_bytes() for p in LOG_DIR.rglob("*") if p.is_file()}
        telemetry = StageTelemetry()
        try:
            assert list(logger.handlers) == handlers_before
            assert logger.propagate == propagate_before
            assert logger.level == level_before
            telemetry.log(Tag.SYS, "probe", k=1)
            assert list(logger.handlers) == handlers_before
            after = {p.relative_to(LOG_DIR): p.read_bytes() for p in LOG_DIR.rglob("*") if p.is_file()}
            assert after == before
        finally:
            assert list(logger.handlers) == handlers_before

    def test_debug_streams_without_root_raise_before_io(self, tmp_path, monkeypatch):
        from src.common.logging import LOG_DIR

        monkeypatch.chdir(tmp_path)
        before = {p.relative_to(LOG_DIR) for p in LOG_DIR.rglob("*")}
        with pytest.raises(TypeError):
            StageTelemetry(debug_streams=True)
        with pytest.raises(TypeError):
            StageTelemetry(debug_streams=True, streams_root=None)
        with pytest.raises(TypeError):
            StageTelemetry(debug_streams=True, streams_root=str(tmp_path))  # type: ignore[arg-type]
        assert list(tmp_path.iterdir()) == []
        assert {p.relative_to(LOG_DIR) for p in LOG_DIR.rglob("*")} == before

    def test_log_rejects_invalid_tag(self):
        telemetry = StageTelemetry(log_run=False)
        with pytest.raises(ValueError, match="tag must be a Tag member"):
            telemetry.log("INVALID", "stage")  # type: ignore[arg-type]

    def test_stream_noop_when_disabled(self, tmp_path):
        """SCENARIO_ANALYSIS_ARCHITECTURE_03: (03a) stream() with sidecars disabled creates no file."""
        telemetry = StageTelemetry(log_run=False, debug_streams=False, streams_root=tmp_path / "mhs")
        telemetry.stream("panel", [{"stage": "panel", "key": "value"}])
        assert not (tmp_path / "mhs").exists()

    def test_stream_writes_when_enabled(self, tmp_path):
        """SCENARIO_ANALYSIS_ARCHITECTURE_03b: stream() with sidecars enabled writes JSONL."""
        streams_root = tmp_path / "mhs"
        telemetry = StageTelemetry(log_run=False, debug_streams=True, streams_root=streams_root)
        telemetry.stream("panel", [{"stage": "panel", "key": "value"}])
        path = streams_root / "panel.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["stage"] == "panel"
        assert row["key"] == "value"

    def test_stream_swallows_exception(self, tmp_path):
        """SCENARIO_ANALYSIS_ARCHITECTURE_03c: stream() swallows exceptions (I-OBSERVE)."""
        telemetry = StageTelemetry(log_run=False, debug_streams=True, streams_root=tmp_path / "mhs")

        def bad_rows():
            raise RuntimeError("boom")
            yield  # type: ignore[misc]

        # Should not raise
        telemetry.stream("panel", bad_rows())

    def test_record_returns_measurement(self):
        telemetry = StageTelemetry(log_run=False)
        m = telemetry.record("base_1h_panel", grid_bars=100, n_symbols=8)
        assert m.stage == "base_1h_panel"
        assert m.grid_bars == 100
        assert m.n_symbols == 8
        assert len(telemetry.records) == 1

    def test_record_tracks_peak_rss(self):
        telemetry = StageTelemetry(log_run=False)
        telemetry.record("stage1")
        telemetry.record("stage2")
        assert len(telemetry.records) == 2
        # peak_rss should be non-decreasing
        assert telemetry.records[1].peak_rss_bytes >= telemetry.records[0].peak_rss_bytes
