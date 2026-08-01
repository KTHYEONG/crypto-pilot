from __future__ import annotations

import json

import pytest

import tools.benchmark_library_admission as benchmark


def test_benchmark_emits_complete_schema_without_absolute_time_assertions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """LAE-07C: the benchmark emits comparable records and never asserts wall time."""
    calls: list[object] = []

    def fake_run(request) -> None:
        calls.append(request.admission.max_workers)

    monkeypatch.setattr(benchmark, "run_technical_library_admission", fake_run)
    monkeypatch.setattr(benchmark, "_measure_panel_bytes", lambda request: 12345)
    monkeypatch.setattr(benchmark, "HARDWARE_MAX_WORKERS", 3)
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_library_admission", "--symbols", "BTCUSDT", "ETHUSDT"],
    )

    benchmark.main()

    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "process"
    records = out["records"]
    assert {r["effective_workers"] for r in records} == {1, 2}
    for record in records:
        assert record["backend"] == "process"
        assert isinstance(record["median_wall_seconds"], (int, float))
        assert isinstance(record["peak_rss_bytes"], int)
        assert isinstance(record["panel_bytes"], int)
        assert record["panel_bytes"] == 12345
    assert len(records) == 2
    assert calls.count(1) == 3
    assert calls.count(2) == 3
