"""Fast cache invariants for the shared MHS diagnostic report cache."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.mhs_requests import research_baseline
from tests.lab.mhs.integration._report_cache import (
    GOLDEN_PROFILE,
    MHS_GOLDEN_BASELINE_GROUP,
    MHS_LATE_MARKET_GROUP,
    MHS_SYNTHETIC_DEFAULT_GROUP,
    SYNTHETIC_DEFAULT_PROFILE,
    CacheStats,
    DiagnosticReportCache,
    DiagnosticRunProfile,
    DiagnosticRunSpec,
    market_fingerprint,
    report_cache_group_violations,
    report_mentions,
    report_state_digest,
    request_digest,
)

_NO_OBSERVE: DiagnosticRunProfile = DiagnosticRunProfile(20, 24, 20260807, None, False)


@dataclass(frozen=True, slots=True, eq=False)
class _FakeReport:
    label: str
    frame: pd.DataFrame
    series: pd.Series
    array: np.ndarray
    bundle: dict[str, tuple[pd.Series, ...]]


def _tiny_frame(seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame({"a": [1.0 + seed, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}, index=idx)


def _tiny_report(label: str = "ok", seed: int = 0) -> _FakeReport:
    frame = _tiny_frame(seed)
    series = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3), name="s")
    return _FakeReport(label, frame, series, np.array([1.0, 2.0, 3.0]), {"k": (series,)})


def _write_market(root: Path, payload: bytes = b"market-bytes") -> Path:
    (root / "funding").mkdir(parents=True, exist_ok=True)
    (root / "markPriceKlines" / "1h").mkdir(parents=True, exist_ok=True)
    (root / "funding" / "BTCUSDT.parquet").write_bytes(payload)
    (root / "markPriceKlines" / "1h" / "BTCUSDT.parquet").write_bytes(payload + b"-mark")
    (root / "manifest.json").write_bytes(b'{"v":1}')
    return root


def _spec(root: Path, profile: DiagnosticRunProfile = _NO_OBSERVE, **overrides: Any) -> DiagnosticRunSpec:
    return DiagnosticRunSpec(research_baseline(data_root=str(root), **overrides), profile)


def _install_fake(monkeypatch: pytest.MonkeyPatch, report: Any, calls: dict[str, int]) -> None:
    import src.lab.mhs.pipeline.orchestrator as orchestrator

    def _fake(request: Any) -> Any:
        calls["n"] = calls.get("n", 0) + 1
        return report

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake)


def test_second_lease_reuses_one_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_market(tmp_path / "mkt")
    calls: dict[str, int] = {}
    _install_fake(monkeypatch, _tiny_report(), calls)
    cache = DiagnosticReportCache()
    spec = _spec(root)
    with cache.lease(spec, consumer="a") as first, cache.lease(spec, consumer="b") as second:
        assert second.report is first.report
    assert calls["n"] == 1
    assert cache.stats == CacheStats(1, 1, 0)


def test_data_root_excluded_but_market_bytes_included(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_a = _write_market(tmp_path / "a")
    root_b = tmp_path / "b"
    root_b.mkdir()
    import shutil

    shutil.copytree(root_a, root_b, dirs_exist_ok=True)
    calls: dict[str, int] = {}
    _install_fake(monkeypatch, _tiny_report(), calls)
    cache = DiagnosticReportCache()
    with cache.lease(_spec(root_a), consumer="a"):
        pass
    with cache.lease(_spec(root_b), consumer="b"):
        pass
    assert calls["n"] == 1
    (root_b / "funding" / "BTCUSDT.parquet").write_bytes(b"changed")
    with cache.lease(_spec(root_b), consumer="c"):
        pass
    assert calls["n"] == 2
    assert cache.stats == CacheStats(1, 2, 1)


def test_request_field_change_is_new_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_market(tmp_path / "mkt")
    calls: dict[str, int] = {}
    _install_fake(monkeypatch, _tiny_report(), calls)
    cache = DiagnosticReportCache()
    with cache.lease(_spec(root), consumer="a"):
        pass
    with cache.lease(_spec(root, touch_diagnostic=True), consumer="b"):
        pass
    assert calls["n"] == 2


def test_profile_change_is_new_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_market(tmp_path / "mkt")
    calls: dict[str, int] = {}
    _install_fake(monkeypatch, _tiny_report(), calls)
    cache = DiagnosticReportCache()
    with cache.lease(_spec(root, SYNTHETIC_DEFAULT_PROFILE), consumer="a"):
        pass
    with cache.lease(_spec(root, GOLDEN_PROFILE), consumer="b"):
        pass
    assert calls["n"] == 2


def test_module_globals_patched_only_during_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.core.marks as marks
    import src.lab.mhs.pipeline.stages.fold as fold_stage
    import src.lab.mhs.statistics as statistics
    from src.lab.mhs.pipeline import orchestrator

    root = _write_market(tmp_path / "mkt")
    before = {
        "funding": marks.funding_path,
        "rep": statistics._BOOTSTRAP_REPLICATES,
        "block": statistics._BOOTSTRAP_MEAN_BLOCK,
        "seed": statistics._BOOTSTRAP_SEED,
        "trials": fold_stage.derive_trials_attempted,
    }
    recorded: dict[str, Any] = {}

    def _fake(request: Any) -> Any:
        recorded["funding"] = marks.funding_path("X")
        recorded["rep"] = statistics._BOOTSTRAP_REPLICATES
        recorded["block"] = statistics._BOOTSTRAP_MEAN_BLOCK
        recorded["seed"] = statistics._BOOTSTRAP_SEED
        recorded["trials"] = fold_stage.derive_trials_attempted()
        return _tiny_report()

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake)
    cache = DiagnosticReportCache()
    with cache.lease(_spec(root, GOLDEN_PROFILE), consumer="a"):
        pass
    assert str(recorded["funding"]).startswith(str(root))
    assert (recorded["rep"], recorded["block"], recorded["seed"]) == (20, 24, 20260807)
    assert recorded["trials"] == (80, "constant_plus_ledger")
    assert marks.funding_path is before["funding"]
    assert statistics._BOOTSTRAP_REPLICATES is before["rep"]
    assert statistics._BOOTSTRAP_MEAN_BLOCK is before["block"]
    assert statistics._BOOTSTRAP_SEED is before["seed"]
    assert fold_stage.derive_trials_attempted is before["trials"]


def test_patches_restored_when_run_raises_and_failure_memoised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.core.marks as marks
    import src.lab.mhs.statistics as statistics
    from src.lab.mhs.pipeline import orchestrator

    root = _write_market(tmp_path / "mkt")
    before = (marks.funding_path, statistics._BOOTSTRAP_REPLICATES)
    calls: dict[str, int] = {}
    boom = ValueError("boom")

    def _fake(request: Any) -> Any:
        calls["n"] = calls.get("n", 0) + 1
        raise boom

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake)
    cache = DiagnosticReportCache()
    spec = _spec(root)
    with pytest.raises(ValueError, match="boom"), cache.lease(spec, consumer="a"):
        pass
    with pytest.raises(RuntimeError, match="cached diagnostic failure") as error, cache.lease(spec, consumer="b"):
        pass
    assert error.value.__cause__ is boom
    assert calls["n"] == 1
    assert marks.funding_path is before[0]
    assert statistics._BOOTSTRAP_REPLICATES is before[1]


def test_consumer_mutation_detected_and_evicted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_market(tmp_path / "mkt")
    calls: dict[str, int] = {}
    _install_fake(monkeypatch, _tiny_report(), calls)
    cache = DiagnosticReportCache()
    spec = _spec(root)
    with cache.lease(spec, consumer="a") as entry:
        pass
    with pytest.raises(AssertionError, match="consumer-mut"), cache.lease(spec, consumer="consumer-mut") as entry2:
        entry2.report.frame.iloc[0, 0] = -999.0
    assert calls["n"] == 1
    with cache.lease(spec, consumer="c"):
        pass
    assert calls["n"] == 2
    assert cache.stats.misses == 2
    assert cache.stats.evictions == 1


def test_read_only_consumers_keep_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_market(tmp_path / "mkt")
    _install_fake(monkeypatch, _tiny_report(), {})
    cache = DiagnosticReportCache()
    spec = _spec(root)
    with cache.lease(spec, consumer="a") as entry:
        report = entry.report
        assert report_state_digest(report) == entry.state_digest
        _ = report.frame.head(2)
        assert report.frame.equals(report.frame.copy())
        _ = report_state_digest(report)


def test_state_digest_detects_nested_container_mutation() -> None:
    series = pd.Series([1.0, 2.0], index=pd.RangeIndex(2), name="v")
    before = _FakeReport("n", _tiny_frame(), series, np.array([1.0, 2.0]), {"k": (series,)})
    assert report_state_digest(before) == report_state_digest(before)
    mutated_array = _FakeReport("n", _tiny_frame(), series, np.array([9.0, 2.0]), {"k": (series,)})
    assert report_state_digest(mutated_array) != report_state_digest(before)
    mutated_series = pd.Series([9.0, 2.0], index=pd.RangeIndex(2), name="v")
    mutated = _FakeReport("n", _tiny_frame(), mutated_series, np.array([1.0, 2.0]), {"k": (mutated_series,)})
    assert report_state_digest(mutated) != report_state_digest(before)
    renamed = _FakeReport("n", _tiny_frame(), series, np.array([1.0, 2.0]), {"other": (series,)})
    assert report_state_digest(renamed) != report_state_digest(before)
    nan_frame = pd.DataFrame({"a": [float("nan"), 1.0]}, index=pd.RangeIndex(2))
    first = _FakeReport("n", nan_frame, series, np.array([1.0, 2.0]), {"k": (series,)})
    assert report_state_digest(first) == report_state_digest(first)


def test_data_root_leak_refuses_sharing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.lab.mhs.pipeline.orchestrator as orchestrator

    root = _write_market(tmp_path / "mkt")
    calls: dict[str, int] = {}

    def _fake(request: Any) -> Any:
        calls["n"] = calls.get("n", 0) + 1
        return _FakeReport(str(root), _tiny_frame(), pd.Series([1.0], name="s"), np.array([1.0]), {})

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake)
    cache = DiagnosticReportCache()
    with pytest.raises(AssertionError, match="data_root"), cache.lease(_spec(root), consumer="a"):
        pass
    with pytest.raises(AssertionError, match="data_root"), cache.lease(_spec(root), consumer="b"):
        pass
    assert calls["n"] == 2


def test_missing_data_root_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.lab.mhs.pipeline import orchestrator

    calls: dict[str, int] = {}

    def _fake(request: Any) -> Any:
        calls["n"] = calls.get("n", 0) + 1
        return _tiny_report()

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake)
    cache = DiagnosticReportCache()
    spec = DiagnosticRunSpec(research_baseline(), _NO_OBSERVE)
    with pytest.raises(ValueError, match="data_root"), cache.lease(spec, consumer="a"):
        pass
    assert calls.get("n", 0) == 0


def test_observation_spies_restored_and_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.lab.mhs import scaling
    from src.lab.mhs.evaluation import books as eval_books
    from src.lab.mhs.pipeline import orchestrator
    from src.lab.mhs.pipeline.stages import replay as replay_stage

    root = _write_market(tmp_path / "mkt")
    before = (
        eval_books._book_weights,
        scaling._regime_cash_scale,
        scaling.regime_cash_scale_1h,
        scaling._apply_rebalance_deadband,
        replay_stage.run_replays,
    )
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda request: _tiny_report())
    cache = DiagnosticReportCache()
    with cache.lease(_spec(root, SYNTHETIC_DEFAULT_PROFILE), consumer="a") as entry:
        obs = entry.observations
        assert obs is not None
        assert obs.replay_entry == {}
        assert isinstance(obs.ema_spans, MappingProxyType)
        assert isinstance(obs.regime_callers, tuple)
        assert isinstance(obs.deadband_callers, tuple)
        captures = obs.calibration_captures()
        with pytest.raises(TypeError):
            captures["ema_spans"] = {}  # type: ignore[index]
    assert eval_books._book_weights is before[0]
    assert scaling._regime_cash_scale is before[1]
    assert scaling.regime_cash_scale_1h is before[2]
    assert scaling._apply_rebalance_deadband is before[3]
    assert replay_stage.run_replays is before[4]
    assert multiprocessing.active_children() == []


class _FakeMark:
    def __init__(self, *args: str) -> None:
        self.args = args


class _FakeItem:
    def __init__(
        self,
        nodeid: str,
        path: Path,
        fixturenames: list[str],
        groups: list[str] | None = None,
        params: dict[str, str] | None = None,
    ) -> None:
        self.nodeid = nodeid
        self.path = path
        self.fixturenames = fixturenames
        self._groups = groups or []
        self.callspec = type("C", (), {"params": params})() if params is not None else None

    def iter_markers(self, name: str | None = None):
        return iter([_FakeMark(g) for g in self._groups])


def test_group_guard_flags_split_shares(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    in_suite = suite / "test_x.py"
    outside = tmp_path / "other" / "test_y.py"
    items = [
        _FakeItem("a::report-no-group", in_suite, ["report"]),
        _FakeItem("b::report-two-groups", in_suite, ["report"], [MHS_SYNTHETIC_DEFAULT_GROUP, "other"]),
        _FakeItem("c::late-wrong-group", in_suite, ["late_market_report"], [MHS_SYNTHETIC_DEFAULT_GROUP]),
        _FakeItem("d::baseline-no-group", in_suite, ["matrix_market"], None, {"matrix_market": "baseline"}),
        _FakeItem(
            "e::committee-wrong-group",
            in_suite,
            ["matrix_market"],
            [MHS_GOLDEN_BASELINE_GROUP],
            {"matrix_market": "committee"},
        ),
        _FakeItem("f::outside", outside, ["report"]),
    ]
    violations = report_cache_group_violations(items, suite)  # type: ignore[arg-type]
    assert len(violations) == 5
    for nodeid in ("a::report-no-group", "b::report-two-groups", "c::late-wrong-group", "d::baseline-no-group", "e::committee-wrong-group"):
        assert any(nodeid in line for line in violations)
    assert not any("f::outside" in line for line in violations)
    ok_items = [
        _FakeItem("g1", in_suite, ["report"], [MHS_SYNTHETIC_DEFAULT_GROUP]),
        _FakeItem("g2", in_suite, ["late_market_report"], [MHS_LATE_MARKET_GROUP]),
        _FakeItem("g3", in_suite, ["matrix_market"], [MHS_GOLDEN_BASELINE_GROUP], {"matrix_market": "baseline"}),
        _FakeItem("g4", in_suite, ["matrix_market"], None, {"matrix_market": "committee"}),
        _FakeItem("g5", in_suite, ["touch_report"]),
    ]
    assert report_cache_group_violations(ok_items, suite) == []  # type: ignore[arg-type]


def test_request_digest_excludes_data_root_and_fingerprint_sorts(tmp_path: Path) -> None:
    root = _write_market(tmp_path / "mkt")
    first = research_baseline(data_root=str(root))
    second = research_baseline(data_root=str(root / "elsewhere"))
    assert request_digest(first) == request_digest(second)
    assert market_fingerprint(root) == market_fingerprint(root)


@pytest.mark.parametrize("mutation", ["series_index", "series_index_name", "columns_name", "multiindex_name"])
def test_axis_mutation_is_detected_and_evicted(tmp_path, monkeypatch, mutation):
    root = _write_market(tmp_path / "market")
    report = _tiny_report()
    if mutation == "multiindex_name":
        report.frame.index = pd.MultiIndex.from_tuples([(i, "X") for i in range(4)], names=["time", "symbol"])
    calls = {}
    _install_fake(monkeypatch, report, calls)
    cache = DiagnosticReportCache()
    def mutate():
        if mutation == "series_index":
            report.series.index = pd.Index([4, 5, 6])
        elif mutation == "series_index_name":
            report.series.index.name = "changed"
        elif mutation == "columns_name":
            report.frame.columns.name = "changed"
        else:
            report.frame.index.names = ["changed", "symbol"]
    with pytest.raises(AssertionError, match="axis-mutator"), cache.lease(_spec(root), consumer="axis-mutator"):
        mutate()
    with cache.lease(_spec(root), consumer="next"):
        pass
    assert calls["n"] == 2
    assert cache.stats.evictions == 1


@pytest.mark.parametrize("location", ["series_index", "frame_index", "tuple_column", "path_key", "multiindex"])
def test_market_path_in_labels_refuses_sharing(tmp_path, monkeypatch, location):
    root = _write_market(tmp_path / "market")
    needle = str(root)
    reports = {
        "series_index": pd.Series([1.0], index=[needle]),
        "frame_index": pd.DataFrame({"a": [1.0]}, index=[needle]),
        "tuple_column": pd.DataFrame([[1.0]], columns=pd.MultiIndex.from_tuples([("a", needle)])),
        "path_key": {Path(root): 1.0},
        "multiindex": pd.Series([1.0], index=pd.MultiIndex.from_tuples([(needle, 1)])),
    }
    report = reports[location]
    assert report_mentions(report, needle)
    _install_fake(monkeypatch, report, {})
    with pytest.raises(AssertionError, match="data_root"), DiagnosticReportCache().lease(_spec(root), consumer="leak"):
        pass
    assert not report_mentions(pd.Series([needle], name="value"), needle)


def test_deep_state_is_not_silently_truncated():
    leaf = [0]
    report = leaf
    for _ in range(110):
        report = [report]
    before = report_state_digest(report)
    leaf[0] = 1
    assert report_state_digest(report) != before
    leaf[0] = Path("/market/deep")
    assert report_mentions(report, "/market")


def test_missing_patch_target_leaves_globals_untouched(tmp_path, monkeypatch):
    import src.core.marks as marks
    import src.lab.mhs.pipeline.stages.fold as fold_stage
    import src.lab.mhs.statistics as statistics

    root = _write_market(tmp_path / "market")
    originals = (marks.funding_path, statistics._BOOTSTRAP_REPLICATES,
                 statistics._BOOTSTRAP_MEAN_BLOCK, statistics._BOOTSTRAP_SEED)
    monkeypatch.delattr(fold_stage, "derive_trials_attempted")
    with pytest.raises(AttributeError, match="derive_trials_attempted"), DiagnosticReportCache().lease(
        _spec(root, GOLDEN_PROFILE), consumer="missing-target"
    ):
        pass
    current = (marks.funding_path, statistics._BOOTSTRAP_REPLICATES,
               statistics._BOOTSTRAP_MEAN_BLOCK, statistics._BOOTSTRAP_SEED)
    assert all(a is b for a, b in zip(current, originals, strict=True))


@pytest.mark.parametrize("fallback", [False, True], ids=["manager", "queue"])
def test_observed_failure_restores_spies_and_closes_resources(tmp_path, monkeypatch, fallback):
    from src.lab.mhs import scaling
    from src.lab.mhs.evaluation import books
    from src.lab.mhs.pipeline import orchestrator
    from src.lab.mhs.pipeline.stages import replay
    from threading import enumerate as threads

    root = _write_market(tmp_path / "market")
    originals = (books._book_weights, scaling._regime_cash_scale, scaling.regime_cash_scale_1h,
                 scaling._apply_rebalance_deadband, replay.run_replays)
    children = {child.pid for child in multiprocessing.active_children()}
    before_threads = {thread.ident for thread in threads()}
    if fallback:
        def unavailable():
            raise PermissionError("manager unavailable")
        monkeypatch.setattr(multiprocessing, "Manager", unavailable)
    boom = ValueError("observed failure")
    calls = []

    def fail(request):
        calls.append(request)
        scaling._regime_cash_scale(pd.Series([1.0, 2.0]))
        raise boom

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", fail)
    cache = DiagnosticReportCache()
    spec = _spec(root, SYNTHETIC_DEFAULT_PROFILE)
    with pytest.raises(ValueError, match="observed failure"), cache.lease(spec, consumer="first"):
        pass
    with pytest.raises(RuntimeError) as error, cache.lease(spec, consumer="second"):
        pass
    assert error.value.__cause__ is boom
    assert len(calls) == 1
    current = (books._book_weights, scaling._regime_cash_scale, scaling.regime_cash_scale_1h,
               scaling._apply_rebalance_deadband, replay.run_replays)
    assert all(a is b for a, b in zip(current, originals, strict=True))
    assert {child.pid for child in multiprocessing.active_children()} == children
    assert {thread.ident for thread in threads()} == before_threads


@pytest.mark.parametrize("fallback", [False, True], ids=["manager", "queue"])
def test_fork_observations_are_complete_pass_through_and_reset(tmp_path, monkeypatch, fallback):
    from types import SimpleNamespace
    from src.lab.mhs import scaling
    from src.lab.mhs.evaluation import books
    from src.lab.mhs.pipeline import orchestrator
    from src.lab.mhs.pipeline.stages import replay

    root = _write_market(tmp_path / "market")
    if fallback:
        def unavailable():
            raise OSError("manager unavailable")
        monkeypatch.setattr(multiprocessing, "Manager", unavailable)
    frame = _tiny_frame()
    eligible = frame.notna()
    book_spec = SimpleNamespace(band=SimpleNamespace(name="fast_reversal"))
    grid = frame.index
    seen = []

    def real_weights(log_close, actual_eligible, spec, step_grid, ema_span=None):
        assert log_close is frame
        assert actual_eligible is eligible
        assert spec is book_spec
        assert step_grid is grid
        assert ema_span == 7
        seen.append("weights")
        return frame

    monkeypatch.setattr(books, "_book_weights", real_weights)
    monkeypatch.setattr(scaling, "_regime_cash_scale", lambda *a, **kw: frame)
    monkeypatch.setattr(scaling, "regime_cash_scale_1h", lambda *a, **kw: frame)
    monkeypatch.setattr(scaling, "_apply_rebalance_deadband", lambda *a, **kw: frame)
    monkeypatch.setattr(replay, "run_replays", lambda *a, **kw: frame)

    def fork_caller():
        for _ in range(300):
            assert books._book_weights(frame, eligible, spec=book_spec, step_grid=grid, ema_span=7) is frame
        assert scaling._regime_cash_scale(frame) is frame
        assert scaling.regime_cash_scale_1h(frame) is frame
        assert scaling._apply_rebalance_deadband(frame) is frame

    def fake_run(request):
        assert books._book_weights(frame, eligible, spec=book_spec, step_grid=grid, ema_span=7) is frame
        child = multiprocessing.get_context("fork").Process(target=fork_caller)
        child.start()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join()
            pytest.fail("observation transport blocked fork worker")
        assert child.exitcode == 0
        ctx = SimpleNamespace(signal_48h=frame)
        assert replay.run_replays(ctx, object()) is frame
        return _tiny_report()

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", fake_run)
    cache = DiagnosticReportCache()
    spec = _spec(root, SYNTHETIC_DEFAULT_PROFILE)
    for consumer in ("first", "recomputed"):
        with cache.lease(spec, consumer=consumer) as entry:
            obs = entry.observations
            assert obs.ema_spans["fast_reversal"] == (7,) * 301
            assert obs.regime_callers == ("fork_caller", "fork_caller")
            assert obs.deadband_callers == ("fork_caller",)
            assert obs.replay_entry == {"log_close_released": True, "signal_empty": False}
        cache.clear()
    assert seen == ["weights", "weights"]
