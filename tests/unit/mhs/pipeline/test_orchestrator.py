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
