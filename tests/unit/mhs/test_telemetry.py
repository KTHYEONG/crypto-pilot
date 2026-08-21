"""Tests for src/mhs/telemetry.py — StageTelemetry and Tag."""

from __future__ import annotations

import json
import logging

import pytest

from src.mhs.telemetry import StageTelemetry, Tag, _format_value


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

    def test_tag_is_strenum(self):
        assert len(Tag) == 4


class TestStageTelemetry:
    def test_log_emits_message(self):
        """SCENARIO_ANALYSIS_ARCHITECTURE_02: StageTelemetry.log emits [TAG] stage=... k=v ..."""
        from io import StringIO
        telemetry = StageTelemetry(log_run=False)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        telemetry._logger.addHandler(handler)
        try:
            telemetry.log(Tag.ALGO, "committee_book", gross=0.9231, members=("a", "b", "c", "d", "e", "f", "g"))
            output = stream.getvalue()
            assert "[ALGO]" in output
            assert "stage=committee_book" in output
            assert "gross=0.923" in output
            assert "truncated=2" in output
        finally:
            telemetry._logger.removeHandler(handler)

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
