from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

import psutil

from src.domain.futures.optimization.observability.l2_runtime_probe import (
    L2RuntimeProbe,
    RuntimeProbeSnapshot,
    RuntimeSpanSummary,
    _clamp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_proc(
    pid: int = 1000,
    ppid: int = 999,
    rss: int = 500 * 1024 * 1024,
    pss: int = 400 * 1024 * 1024,
    name: str = "python",
    children: list[MagicMock] | None = None,
) -> MagicMock:
    proc = MagicMock(spec=psutil.Process)
    proc.pid = pid
    proc.ppid.return_value = ppid
    proc.name.return_value = name
    mem_info = MagicMock()
    mem_info.rss = rss
    proc.memory_info.return_value = mem_info
    full_info = MagicMock()
    full_info.pss = pss
    proc.memory_full_info.return_value = full_info
    if children is not None:
        proc.children.return_value = children
    else:
        proc.children.return_value = []
    return proc


_TMPDIR = Path(tempfile.gettempdir())

def _make_probe(
    enabled: bool = True,
    jsonl_enabled: bool = False,
    sample_ms: int = 250,
    hot_ms: int = 50,
) -> L2RuntimeProbe:
    return L2RuntimeProbe(
        enabled=enabled,
        sample_interval_ms=sample_ms,
        hot_sample_interval_ms=hot_ms,
        jsonl_enabled=jsonl_enabled,
        jsonl_path=_TMPDIR / "test_l2_probe.jsonl",
    )


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — nested span & tree peak
# ---------------------------------------------------------------------------

class TestScenario1HappyPath:
    def test_nested_span_returns_correct_elapsed_and_peak(self) -> None:
        child = _fake_proc(pid=1001, ppid=1000, rss=200 * 1024 * 1024, pss=150 * 1024 * 1024)
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, pss=400 * 1024 * 1024, children=[child])

        with (
            patch.object(os, "getpid", return_value=1000),
            patch.object(os, "getppid", return_value=999),
            patch("psutil.Process", side_effect=lambda pid=None: parent if pid is None or pid == 1000 else child),
        ):
            probe = _make_probe(enabled=True, jsonl_enabled=False)
            probe.start_run(stage="l2")
            with probe.span("bridge"):
                time.sleep(0.005)
                with probe.span("l1_nested"):
                    time.sleep(0.005)
            summaries = probe.stop_run(outcome="completed")

        assert len(summaries) >= 1
        bridge = next((s for s in summaries if "bridge" in s.stage_path), None)
        assert bridge is not None
        assert bridge.calls >= 1
        assert bridge.elapsed_ms >= 5.0
        assert bridge.tree_rss_peak_mb > 0.0
        assert bridge.peak_pid in (1000, 1001)

    def test_snapshot_now_returns_valid_snapshot(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, pss=400 * 1024 * 1024)

        with (
            patch.object(os, "getpid", return_value=1000),
            patch.object(os, "getppid", return_value=999),
            patch("psutil.Process", return_value=parent),
            patch(
                "src.domain.futures.optimization.observability.l2_runtime_probe._parent_vmhwm_mib",
                return_value=480.0,
            ),
        ):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            snap = probe.snapshot_now(reason="test")
            probe.stop_run(outcome="completed")

        assert snap is not None
        assert snap.role == "parent"
        assert snap.rss_mb == 500.0
        assert snap.pss_mb == 400.0
        assert snap.parent_vmhwm_mb == 480.0
        assert snap.status == "ok"

    def test_disabled_probe_returns_empty(self) -> None:
        probe = _make_probe(enabled=False)
        probe.start_run(stage="l2")
        with probe.span("test"):
            pass
        assert probe.snapshot_now(reason="test") is None
        assert len(probe.stop_run(outcome="completed")) == 0


# ---------------------------------------------------------------------------
# Scenario 2: Edge / Performance — PSS degradation
# ---------------------------------------------------------------------------

class TestScenario2EdgePerformance:
    def test_pss_unavailable_sets_neg_one(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, pss=400 * 1024 * 1024)
        full_info = MagicMock()
        del full_info.pss
        parent.memory_full_info.return_value = full_info

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            snap = probe.snapshot_now(reason="test_pss")
            probe.stop_run(outcome="completed")

        assert snap is not None
        assert snap.pss_mb == -1.0
        assert snap.rss_mb == 500.0

    def test_pss_access_denied_returns_neg_one(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)
        parent.memory_full_info.side_effect = psutil.AccessDenied()

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            snap = probe.snapshot_now(reason="test_denied")
            probe.stop_run(outcome="completed")

        assert snap is not None
        assert snap.pss_mb == -1.0

    def test_degraded_state_set_on_slow_samples(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        original = L2RuntimeProbe._do_sample

        def _slow(self: L2RuntimeProbe, *, reason: str, stage_path: str) -> None:
            time.sleep(0.06)
            original(self, reason=reason, stage_path=stage_path)

        with (
            patch.object(os, "getpid", return_value=1000),
            patch("psutil.Process", return_value=parent),
            patch.object(L2RuntimeProbe, "_do_sample", _slow),
        ):
            probe = _make_probe(enabled=True, sample_ms=50, hot_ms=50)
            probe.start_run(stage="l2")
            time.sleep(0.3)
            probe.stop_run(outcome="completed")

        assert probe._degraded
        assert probe._slow_sample_count >= 2

    def test_disabled_probe_no_span_summary(self) -> None:
        probe = _make_probe(enabled=False)
        probe.start_run(stage="l2")
        with probe.span("test"):
            pass
        assert len(probe.stop_run(outcome="completed")) == 0


# ---------------------------------------------------------------------------
# Scenario 3: Error handling — JSONL & process access errors
# ---------------------------------------------------------------------------

class TestScenario3ErrorHandling:
    def test_jsonl_writes_detail_and_aggregate_samples(self, tmp_path: Path) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, pss=400 * 1024 * 1024)
        jsonl_path = tmp_path / "probe.jsonl"

        with (
            patch.object(os, "getpid", return_value=1000),
            patch("psutil.Process", return_value=parent),
        ):
            probe = L2RuntimeProbe(
                enabled=True,
                sample_interval_ms=250,
                hot_sample_interval_ms=50,
                jsonl_enabled=True,
                jsonl_path=jsonl_path,
                pss_interval_samples=100,
            )
            probe.start_run(stage="l2")
            probe._last_child_pids = frozenset()
            probe._last_pss_timestamp = time.time()
            probe._peak_tree_rss = 10_000.0
            probe._do_sample(reason="run_end", stage_path="l2")
            probe._do_sample(reason="steady", stage_path="l2")
            probe.stop_run(outcome="completed")

        records = [line for line in jsonl_path.read_text().splitlines() if line]
        assert records
        assert any('"reason": "steady"' in line and '"tree_rss_mb"' in line for line in records)
        assert any('"pss_mb"' in line and '"run_end"' in line for line in records)

    def test_jsonl_oserror_suppressed(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, pss=400 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True, jsonl_enabled=True)
            with patch.object(Path, "open", side_effect=OSError("permission denied")):
                probe.start_run(stage="l2")
                with probe.span("test"):
                    pass
                summaries = probe.stop_run(outcome="completed")

        assert len(summaries) >= 1

    def test_access_denied_child_does_not_crash(self) -> None:
        child = _fake_proc(pid=1001, ppid=1000, rss=200 * 1024 * 1024)
        child.memory_info.side_effect = psutil.AccessDenied()
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, children=[child])

        with (
            patch.object(os, "getpid", return_value=1000),
            patch("psutil.Process", side_effect=lambda pid=None: parent if pid is None or pid == 1000 else child),
        ):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            with probe.span("test"):
                pass
            probe.stop_run(outcome="completed")

    def test_span_continuation_after_child_error(self) -> None:
        child = _fake_proc(pid=1001, ppid=1000, rss=200 * 1024 * 1024)
        child.children.side_effect = psutil.AccessDenied()
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024, children=[child])

        with (
            patch.object(os, "getpid", return_value=1000),
            patch("psutil.Process", side_effect=lambda pid=None: parent if pid is None or pid == 1000 else child),
        ):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            with probe.span("bridge"):
                pass
            with probe.span("l1_nested"):
                pass
            summaries = probe.stop_run(outcome="completed")
            assert len(summaries) >= 1


# ---------------------------------------------------------------------------
# Scenario 4: Integration — span stages in order & EVAL records
# ---------------------------------------------------------------------------

class TestScenario4Integration:
    def test_span_stages_called_in_order(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            order: list[str] = []
            with probe.span("l2_prepare"):
                order.append("prepare")
                with probe.span("cache_build"):
                    order.append("cache")
            with probe.span("l2_optuna_batch"):
                order.append("batch")
            with probe.span("l2_champion_selection"):
                order.append("champion")
            summaries = probe.stop_run(outcome="completed")

        assert order == ["prepare", "cache", "batch", "champion"]
        paths = [s.stage_path for s in summaries]
        assert "l2_prepare" in paths
        assert "l2_prepare/cache_build" in paths
        assert "l2_optuna_batch" in paths
        assert "l2_champion_selection" in paths

    def test_eval_record_does_not_break(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            probe.record("EVAL", "l2_trial", trial=17, worker_pid=1001, queue_ms=2.5, eval_ms=4500.0)
            probe.stop_run(outcome="completed")

    def test_disabled_probe_ignores_span(self) -> None:
        probe = _make_probe(enabled=False)
        probe.start_run(stage="l2")
        with probe.span("test"):
            pass
        assert len(probe.stop_run(outcome="completed")) == 0


# ---------------------------------------------------------------------------
# Scenario 5: Feature cache attribution
# ---------------------------------------------------------------------------

class TestScenario5FeatureCache:
    def test_multiple_cache_spans_aggregated(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            for _ in range(5):
                with probe.span("l1_feature_cache", tf="15m", cache_status="hit"):
                    pass
            summaries = probe.stop_run(outcome="completed")

        cache_spans = [s for s in summaries if "l1_feature_cache" in s.stage_path]
        assert len(cache_spans) >= 1
        assert cache_spans[0].calls == 5

    def test_cache_tf_isolation(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            with probe.span("l1_feature_cache", tf="15m", cache_status="miss"):
                pass
            with probe.span("l1_feature_cache", tf="1h", cache_status="hit"):
                pass
            summaries = probe.stop_run(outcome="completed")

        cache_spans = [s for s in summaries if "l1_feature_cache" in s.stage_path]
        assert len(cache_spans) >= 1


# ---------------------------------------------------------------------------
# Scenario 6: Replay attribution
# ---------------------------------------------------------------------------

class TestScenario6ReplayAttribution:
    def test_replay_candidates_recorded(self) -> None:
        parent = _fake_proc(pid=1000, ppid=999, rss=500 * 1024 * 1024)

        with (patch.object(os, "getpid", return_value=1000), patch("psutil.Process", return_value=parent)):
            probe = _make_probe(enabled=True)
            probe.start_run(stage="l2")
            with probe.span("l2_champion_selection"):
                probe.record("EVAL", "l2_champion_replay", trial=101, rank=1, elapsed_ms=1200.0,
                             cache_status="hit", promotion="true", crisis_status="passed",
                             outcome="selected", blocker="none")
                probe.record("EVAL", "l2_champion_replay", trial=102, rank=2, elapsed_ms=3500.0,
                             cache_status="miss", promotion="false", crisis_status="failed",
                             outcome="rejected", blocker="mdd_breach")
            probe.stop_run(outcome="completed")


# ---------------------------------------------------------------------------
# from_environment
# ---------------------------------------------------------------------------

class TestFromEnvironment:
    def test_disabled_when_not_debug(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = False
        probe = L2RuntimeProbe.from_environment(logger=logger, base_dir=_TMPDIR)
        assert not probe.enabled

    def test_enabled_when_debug_and_env_set(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True
        with patch.dict(os.environ, {"L2_RUNTIME_PROBE_ENABLED": "true"}, clear=False):
            probe = L2RuntimeProbe.from_environment(logger=logger, base_dir=_TMPDIR)
        assert probe.enabled

    def test_disabled_when_env_not_set_even_if_debug(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True
        with patch.dict(os.environ, {}, clear=False):
            probe = L2RuntimeProbe.from_environment(logger=logger, base_dir=_TMPDIR)
        assert not probe.enabled

    def test_clamps_invalid_sample_ms(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True
        with patch.dict(os.environ, {
            "L2_RUNTIME_PROBE_ENABLED": "true",
            "L2_RUNTIME_PROBE_SAMPLE_MS": "999999",
        }, clear=False):
            probe = L2RuntimeProbe.from_environment(logger=logger, base_dir=_TMPDIR)
        assert probe._sample_interval_ms == 1000


# ---------------------------------------------------------------------------
# _clamp utility
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self) -> None:
        assert _clamp(250, 50, 1000, "test", "unit") == 250

    def test_below_range(self) -> None:
        assert _clamp(-5, 50, 1000, "test", "unit") == 50

    def test_above_range(self) -> None:
        assert _clamp(5000, 50, 1000, "test", "unit") == 1000


# ---------------------------------------------------------------------------
# Data class contracts
# ---------------------------------------------------------------------------

class TestDataClassContracts:
    def test_runtime_probe_snapshot_frozen(self) -> None:
        s = RuntimeProbeSnapshot(
            run_id="test", sample_seq=1, stage_path="l2/test",
            pid=1000, ppid=999, role="parent",
            rss_mb=500.0, pss_mb=400.0, tree_rss_mb=700.0, tree_pss_mb=550.0,
            parent_vmhwm_mb=480.0, sample_elapsed_ms=0.5, status="ok",
        )
        with pytest.raises(AttributeError):
            s.rss_mb = 600.0

    def test_runtime_span_summary_frozen(self) -> None:
        s = RuntimeSpanSummary(
            stage_path="l2/test", calls=3, elapsed_ms=1500.0,
            rss_delta_mb=100.0, tree_rss_peak_mb=1200.0, tree_pss_peak_mb=900.0,
            peak_pid=1001, peak_role="child",
        )
        with pytest.raises(AttributeError):
            s.elapsed_ms = 2000.0

    def test_runtime_probe_snapshot_has_slots(self) -> None:
        s = RuntimeProbeSnapshot(
            run_id="test", sample_seq=1, stage_path="l2/test",
            pid=1000, ppid=999, role="parent",
            rss_mb=500.0, pss_mb=400.0, tree_rss_mb=700.0, tree_pss_mb=550.0,
            parent_vmhwm_mb=480.0, sample_elapsed_ms=0.5, status="ok",
        )
        with pytest.raises((AttributeError, TypeError)):
            s.extra = "no_slots"

    def test_runtime_span_summary_has_slots(self) -> None:
        s = RuntimeSpanSummary(
            stage_path="l2/test", calls=3, elapsed_ms=1500.0,
            rss_delta_mb=100.0, tree_rss_peak_mb=1200.0, tree_pss_peak_mb=900.0,
            peak_pid=1001, peak_role="child",
        )
        with pytest.raises((AttributeError, TypeError)):
            s.extra = "no_slots"
