"""Unit-level isolation for src.mhs.pipeline.orchestrator.run_mhs_diagnostic.

The six-stage pipeline itself is exercised end-to-end by
tests/integration/mhs/test_golden_identity.py; this module isolates the
orchestrator's own composition logic (partition/holdout guards, and the
_TreeMemorySampler wiring added for measurement correctness) without running
a real pipeline pass, so it stays fast per the repo's test-speed directive.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.mhs.pipeline.config import MhsRunConfig
from src.research.evaluation.policy import HOLDOUT_CUTOFF


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeReport:
    marker: str
    tree_memory: object | None = None


def test_orchestrator_rejects_non_dev_partition() -> None:
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    with pytest.raises(RuntimeError, match="dev-only"):
        run_mhs_diagnostic(MhsRunConfig(partition="holdout"))


def test_orchestrator_rejects_end_past_holdout_cutoff() -> None:
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    past = str(HOLDOUT_CUTOFF + pd.Timedelta(days=1))
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        run_mhs_diagnostic(MhsRunConfig(end=past))


def test_orchestrator_attaches_tree_memory_from_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_mhs_diagnostic wraps run_stages with a _TreeMemorySampler and
    attaches its stats to the returned report as `tree_memory`."""
    import src.mhs.pipeline.orchestrator as orchestrator

    fake_report = _FakeReport(marker="stub")
    monkeypatch.setattr(orchestrator, "run_stages", lambda ctx, telemetry: fake_report)

    started: list[bool] = []
    stopped: list[bool] = []

    class _FakeSampler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            started.append(True)

        def stop(self) -> str:
            stopped.append(True)
            return "fake-tree-stats"

    monkeypatch.setattr(orchestrator, "_TreeMemorySampler", _FakeSampler)

    result = orchestrator.run_mhs_diagnostic(MhsRunConfig())

    assert started == [True]
    assert stopped == [True]
    assert result.marker == "stub"
    assert result.tree_memory == "fake-tree-stats"


def test_orchestrator_stops_sampler_even_if_run_stages_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sampler is stopped via `finally` even when a stage raises, so a
    failed run never leaves the background sampler thread running."""
    import src.mhs.pipeline.orchestrator as orchestrator

    def _boom(ctx: object, telemetry: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(orchestrator, "run_stages", _boom)

    stopped: list[bool] = []

    class _FakeSampler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            stopped.append(True)
            return None

    monkeypatch.setattr(orchestrator, "_TreeMemorySampler", _FakeSampler)

    with pytest.raises(ValueError, match="boom"):
        orchestrator.run_mhs_diagnostic(MhsRunConfig())

    assert stopped == [True]


# SCENARIO_MHS_SELECTION_EXEC_BOUNDED_CEILING_02
def test_scenario_mhs_selection_exec_bounded_ceiling_02(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """final_oos_2026h1=True bounds end at MHS_FINAL_OOS_CUTOFF_2026H1: a
    defaulted end resolves to exactly 2026-06-30 23:59:59 UTC without raising,
    an explicit later end fails closed naming both dates, and the flag-off
    default keeps rejecting ends past HOLDOUT_CUTOFF (I1 preserved)."""
    from src.mhs.pipeline.context import PipelineContext

    import src.mhs.pipeline.orchestrator as orchestrator

    fake_report = _FakeReport(marker="stub")
    captured: list[PipelineContext] = []

    def _fake_run_stages(ctx: PipelineContext, telemetry: object) -> _FakeReport:
        captured.append(ctx)
        return fake_report

    class _FakeSampler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            return None

    monkeypatch.setattr(orchestrator, "run_stages", _fake_run_stages)
    monkeypatch.setattr(orchestrator, "_TreeMemorySampler", _FakeSampler)

    orchestrator.run_mhs_diagnostic(MhsRunConfig(final_oos_2026h1=True))
    assert captured[-1].end == pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

    with pytest.raises(RuntimeError, match="Holdout sealed") as excinfo:
        orchestrator.run_mhs_diagnostic(
            MhsRunConfig(final_oos_2026h1=True, end="2026-07-15")
        )
    message = str(excinfo.value)
    assert "2026-07-15" in message
    assert "2026-06-30" in message

    with pytest.raises(RuntimeError) as sealed_excinfo:
        orchestrator.run_mhs_diagnostic(MhsRunConfig(end="2026-01-15"))
    assert "2025-12-31" in str(sealed_excinfo.value)
